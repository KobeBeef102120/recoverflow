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


def fit_transition_model(transitions: pd.DataFrame) -> Pipeline:
    """Fit multinomial logistic regression for Pr(X_{t+1}=j | Z_t)."""
    if transitions.empty:
        raise ValueError("No transitions available. Need at least one episode with two states.")
    y = transitions["next_state"]
    x = transitions.drop(columns=["next_state", "episode_id"], errors="ignore")
    categorical = [c for c in x.columns if x[c].dtype == "object"]
    numeric = [c for c in x.columns if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    model = LogisticRegression(
        max_iter=2000,
        multi_class="auto",
        class_weight="balanced",
    )
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(x, y)
    return pipe


def fit_direct_recovery_model(transitions: pd.DataFrame) -> Pipeline | None:
    """Fit binary logistic regression for Pr(X_{t+1}=FEEDBACK_PASS | Z_t)."""
    if transitions.empty:
        return None
    y = (transitions["next_state"] == "FEEDBACK_PASS").astype(int)
    if y.nunique() < 2:
        return None
    x = transitions.drop(columns=["next_state", "episode_id"], errors="ignore")
    categorical = [c for c in x.columns if x[c].dtype == "object"]
    numeric = [c for c in x.columns if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
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
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(transition_model, out / "hamsm_transition_model.joblib")
    if direct_recovery_model is not None:
        joblib.dump(direct_recovery_model, out / "direct_recovery_model.joblib")


def _delta_pass_rate(logs: list[dict[str, Any]], idx: int) -> float:
    current = logs[idx].get("feedback_pass_rate", 0.0)
    if idx == 0:
        return 0.0
    previous = logs[idx - 1].get("feedback_pass_rate", 0.0)
    return current - previous
