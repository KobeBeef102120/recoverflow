from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(slots=True)
class HAMSMArtifacts:
    transition_model: Pipeline
    direct_recovery_model: Pipeline | None
    feature_columns: list[str]
    label_column: str


# Columns that are metadata and must never be used as model features.
_META_COLS: frozenset[str] = frozenset({
    "next_state",
    "episode_id",
    "problem_id",
    "model_id",
    "final_outcome",
})


def build_transition_dataset(episodes: list[dict[str, Any]], history_length: int = 3) -> pd.DataFrame:
    """Create one row per observed transition X_t -> X_{t+1}.

    This is the model-ready representation of Z_{n,t}: current state, recent ordered history,
    cumulative counts, dwell time, attempt index, feedback type, and pass-rate trajectory.
    """
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        logs = ep["trajectory"]
        for idx in range(len(logs) - 1):
            current = logs[idx]
            nxt = logs[idx + 1]
            history = list(current.get("ordered_history", []))[-history_length:]
            history = ["START"] * (history_length - len(history)) + history

            row = {
                "episode_id": ep["episode_id"],
                "problem_id": ep["problem_id"],
                "model_id": ep["model_id"],
                "current_state": current["state"],
                "next_state": nxt["state"],
                "attempt": current["attempt"],
                "attempt_normalized": current["attempt"] / max(current.get("max_attempts", 1), 1),
                "dwell_time_current_state": current.get("dwell_time_current_state", 1),
                "feedback_type": current.get("feedback_type", "NONE"),
                "feedback_pass_rate": current.get("feedback_pass_rate", 0.0),
                "delta_feedback_pass_rate": _delta_pass_rate(logs, idx),
                "final_outcome": ep.get("final_outcome"),
            }
            for h_i, state in enumerate(history, start=1):
                row[f"history_{h_i}"] = state

            counts = current.get("cumulative_state_counts", {}) or {}
            for state, count in counts.items():
                row[f"count_{state}"] = count
            rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def _feature_columns(transitions: pd.DataFrame, feature_set: str) -> list[str]:
    """Return the column names to use as features for a given model variant."""
    all_cols = [c for c in transitions.columns if c not in _META_COLS]
    history_cols = [c for c in all_cols if c.startswith("history_")]
    count_cols = [c for c in all_cols if c.startswith("count_")]

    if feature_set == "first_order":
        return ["current_state"]
    if feature_set == "higher_order":
        return ["current_state"] + history_cols
    if feature_set == "count_augmented":
        return ["current_state"] + count_cols
    if feature_set == "duration_dependent":
        return ["current_state", "dwell_time_current_state"]
    # "full" — all non-metadata features (HAMSM)
    return all_cols


def _fit_pipeline(transitions: pd.DataFrame, feature_set: str) -> Pipeline:
    """Fit a multinomial logistic transition model for the given feature set."""
    y = transitions["next_state"]
    cols = _feature_columns(transitions, feature_set)
    x = transitions[cols]
    categorical = [c for c in cols if x[c].dtype == "object"]
    numeric = [c for c in cols if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(x, y)
    return pipe


def fit_transition_model(transitions: pd.DataFrame) -> Pipeline:
    """Fit the full HAMSM multinomial transition model: Pr(X_{t+1} | Z_t)."""
    if transitions.empty:
        raise ValueError("No transitions available. Need at least one episode with two states.")
    return _fit_pipeline(transitions, "full")


def fit_first_order_model(transitions: pd.DataFrame) -> Pipeline:
    """Baseline: Pr(X_{t+1} | X_t) — first-order Markov."""
    return _fit_pipeline(transitions, "first_order")


def fit_higher_order_model(transitions: pd.DataFrame) -> Pipeline:
    """Baseline: Pr(X_{t+1} | X_t, X_{t-1}, ...) — higher-order Markov."""
    return _fit_pipeline(transitions, "higher_order")


def fit_count_augmented_model(transitions: pd.DataFrame) -> Pipeline:
    """Baseline: Pr(X_{t+1} | X_t, C_t) — count-augmented Markov."""
    return _fit_pipeline(transitions, "count_augmented")


def fit_duration_dependent_model(transitions: pd.DataFrame) -> Pipeline:
    """Baseline: Pr(X_{t+1} | X_t, D_t) — duration-dependent Markov."""
    return _fit_pipeline(transitions, "duration_dependent")


def compare_baseline_models(transitions: pd.DataFrame) -> pd.DataFrame:
    """Fit all HAMSM baselines and return a training-accuracy comparison table.

    Corresponds to Section 9 of the RecoverFlow paper (baseline suite for RQ3).

    Note: accuracy is computed on the same data used for fitting. For the final
    paper, use cross-validation (sklearn cross_val_score) on a larger dataset to
    get unbiased estimates.
    """
    if transitions.empty or transitions["next_state"].nunique() < 2:
        return pd.DataFrame()

    y = transitions["next_state"]
    configs = [
        ("First-Order Markov",       "first_order"),
        ("Higher-Order Markov",      "higher_order"),
        ("Count-Augmented Markov",   "count_augmented"),
        ("Duration-Dependent Markov","duration_dependent"),
        ("HAMSM (Full)",             "full"),
    ]

    rows = []
    for name, feature_set in configs:
        try:
            pipe = _fit_pipeline(transitions, feature_set)
            cols = _feature_columns(transitions, feature_set)
            preds = pipe.predict(transitions[cols])
            accuracy = float((preds == y).mean())
            rows.append({
                "model": name,
                "feature_set": feature_set,
                "train_accuracy": round(accuracy, 4),
                "n_transitions": len(transitions),
                "n_classes": int(y.nunique()),
            })
        except Exception as exc:
            rows.append({
                "model": name,
                "feature_set": feature_set,
                "train_accuracy": float("nan"),
                "n_transitions": len(transitions),
                "n_classes": int(y.nunique()),
                "error": str(exc),
            })

    return pd.DataFrame(rows)


def fit_direct_recovery_model(transitions: pd.DataFrame) -> Pipeline | None:
    """Fit binary logistic regression for Pr(X_{t+1}=FEEDBACK_PASS | Z_t)."""
    if transitions.empty:
        return None
    y = (transitions["next_state"] == "FEEDBACK_PASS").astype(int)
    if y.nunique() < 2:
        return None

    cols = _feature_columns(transitions, "full")
    x = transitions[cols]
    categorical = [c for c in cols if x[c].dtype == "object"]
    numeric = [c for c in cols if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(x, y)
    return pipe


def predict_next_state_probabilities(model: Pipeline, z_rows: pd.DataFrame) -> pd.DataFrame:
    probs = model.predict_proba(z_rows)
    classes = model.named_steps["model"].classes_
    return pd.DataFrame(probs, columns=[f"Pr_next_{c}" for c in classes])


def save_models(
    out_dir: str | Path,
    *,
    transition_model: Pipeline,
    direct_recovery_model: Pipeline | None,
    first_order_model: Pipeline | None = None,
    higher_order_model: Pipeline | None = None,
    count_augmented_model: Pipeline | None = None,
    duration_dependent_model: Pipeline | None = None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(transition_model, out / "hamsm_transition_model.joblib")
    if direct_recovery_model is not None:
        joblib.dump(direct_recovery_model, out / "direct_recovery_model.joblib")
    if first_order_model is not None:
        joblib.dump(first_order_model, out / "first_order_markov_model.joblib")
    if higher_order_model is not None:
        joblib.dump(higher_order_model, out / "higher_order_markov_model.joblib")
    if count_augmented_model is not None:
        joblib.dump(count_augmented_model, out / "count_augmented_model.joblib")
    if duration_dependent_model is not None:
        joblib.dump(duration_dependent_model, out / "duration_dependent_model.joblib")


def _delta_pass_rate(logs: list[dict[str, Any]], idx: int) -> float:
    current = logs[idx].get("feedback_pass_rate", 0.0)
    if idx == 0:
        return 0.0
    previous = logs[idx - 1].get("feedback_pass_rate", 0.0)
    return current - previous
