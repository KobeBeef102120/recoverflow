from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from agent_repair_eval.states import NON_RUNNABLE_STATES, RUNNABLE_FAILURE_STATES, State


def pass_at_k_summary(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    """Sequential pass-at-k table (Section 9.1 baseline metric).

    For each k, reports the fraction of episodes where the model passed all
    feedback tests within the first k attempts, and the overall hidden-test
    success rate (which is fixed — it reflects the final code after the full
    attempt budget, not a per-k value).

    Note: this is sequential repair pass@k, not the independent-sampling
    estimator used in the original HumanEval Pass@k paper.
    """
    if not episodes:
        return pd.DataFrame()

    n = len(episodes)
    # Exclude TERMINAL_UNRESOLVED sentinel from trajectory length
    real_attempts = lambda ep: [
        log for log in ep["trajectory"]
        if log["state"] != State.TERMINAL_UNRESOLVED.value
    ]
    max_k = max((len(real_attempts(ep)) for ep in episodes), default=0)
    rows = []
    for k in range(1, max_k + 1):
        feedback_pass = sum(
            1 for ep in episodes
            if any(
                log["state"] == State.FEEDBACK_PASS.value
                for log in real_attempts(ep)[:k]
            )
        ) / n
        rows.append({"k": k, "feedback_pass_at_k": round(feedback_pass, 4)})

    hidden_pass = sum(
        1 for ep in episodes if ep["final_outcome"] == "FINAL_PASS"
    ) / n
    df = pd.DataFrame(rows)
    df["hidden_pass_rate"] = round(hidden_pass, 4)
    return df


def feedback_loop_success_rate(episodes: list[dict[str, Any]]) -> float:
    """Fraction of episodes where the model passes all feedback tests within K attempts."""
    if not episodes:
        return 0.0
    return sum(
        1 for ep in episodes
        if any(log["state"] == State.FEEDBACK_PASS.value for log in ep["trajectory"])
    ) / len(episodes)


def flatten_attempts(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ep in episodes:
        for log in ep["trajectory"]:
            row = dict(log)
            row["final_outcome"] = ep["final_outcome"]
            row["final_hidden_pass_rate"] = ep["final_hidden_pass_rate"]
            rows.append(row)
    return pd.DataFrame(rows)


def state_frequencies(attempts: pd.DataFrame) -> pd.DataFrame:
    counts = attempts["state"].value_counts().rename_axis("state").reset_index(name="frequency")
    counts["percentage"] = counts["frequency"] / counts["frequency"].sum()
    return counts


def transition_table(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    counts: Counter[tuple[str, str]] = Counter()
    source_counts: Counter[str] = Counter()
    for ep in episodes:
        states = [log["state"] for log in ep["trajectory"]]
        for a, b in zip(states, states[1:]):
            counts[(a, b)] += 1
            source_counts[a] += 1

    rows = []
    for (source, dest), count in sorted(counts.items()):
        rows.append(
            {
                "source_state": source,
                "destination_state": dest,
                "count": count,
                "probability": count / source_counts[source],
            }
        )
    return pd.DataFrame(rows)


def transition_matrix(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    from agent_repair_eval.states import STATE_PRIORITY
    table = transition_table(episodes)
    if table.empty:
        return pd.DataFrame()
    matrix = table.pivot(index="source_state", columns="destination_state", values="probability")
    all_states = [s.value for s in STATE_PRIORITY]
    matrix = matrix.reindex(index=all_states, columns=all_states, fill_value=0.0)
    return matrix


def recovery_by_state(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    denominators: Counter[str] = Counter()
    feedback_recovered: Counter[str] = Counter()
    hidden_recovered: Counter[str] = Counter()
    one_step_recovered: Counter[str] = Counter()
    transitions_from: Counter[str] = Counter()
    persisted: Counter[str] = Counter()

    _non_error = {State.FEEDBACK_PASS.value, State.TERMINAL_UNRESOLVED.value}

    for ep in episodes:
        states = [log["state"] for log in ep["trajectory"]]
        hidden_success = ep["final_outcome"] == "FINAL_PASS"

        for i, s in enumerate(states):
            if s in _non_error:
                continue
            # Only count recovery if FEEDBACK_PASS appears AFTER this error state
            recovered_after = State.FEEDBACK_PASS.value in states[i + 1:]
            denominators[s] += 1
            if recovered_after:
                feedback_recovered[s] += 1
            if hidden_success:
                hidden_recovered[s] += 1

        for a, b in zip(states, states[1:]):
            if a in _non_error:
                continue
            transitions_from[a] += 1
            if b == State.FEEDBACK_PASS.value:
                one_step_recovered[a] += 1
            if a == b:
                persisted[a] += 1

    rows = []
    for s in sorted(denominators):
        rows.append(
            {
                "state": s,
                "episodes_with_state": denominators[s],
                "feedback_recovery_rate_within_k": feedback_recovered[s] / denominators[s],
                "hidden_success_rate_given_state": hidden_recovered[s] / denominators[s],
                "unresolved_rate": 1.0 - feedback_recovered[s] / denominators[s],
                "one_step_recovery_rate": (
                    one_step_recovered[s] / transitions_from[s] if transitions_from[s] else 0.0
                ),
                "persistence_rate": persisted[s] / transitions_from[s] if transitions_from[s] else 0.0,
            }
        )
    return pd.DataFrame(rows)


def dwell_time_summary(attempts: pd.DataFrame) -> pd.DataFrame:
    """Summarize how long the model stays in each state once it enters.

    Reports true *run lengths* (number of consecutive attempts in a state before
    leaving it), not the per-attempt cumulative dwell counter. A single run of
    length 5 reports mean/median/max dwell = 5, and attempt_count = 5.
    """
    if attempts.empty:
        return pd.DataFrame()

    import statistics

    run_lengths: dict[str, list[int]] = defaultdict(list)
    occurrences: Counter[str] = Counter()

    # Preserve trajectory order within each episode (flatten_attempts emits rows in order).
    for _, group in attempts.groupby("episode_id", sort=False):
        prev_state: str | None = None
        run_len = 0
        for state in group["state"]:
            occurrences[state] += 1
            if state == prev_state:
                run_len += 1
            else:
                if prev_state is not None:
                    run_lengths[prev_state].append(run_len)
                prev_state = state
                run_len = 1
        if prev_state is not None:
            run_lengths[prev_state].append(run_len)

    rows = []
    for state in sorted(occurrences):
        lengths = run_lengths[state]
        rows.append(
            {
                "state": state,
                "attempt_count": occurrences[state],
                "median_dwell_time": float(statistics.median(lengths)),
                "mean_dwell_time": sum(lengths) / len(lengths),
                "max_observed_dwell": max(lengths),
            }
        )
    return pd.DataFrame(rows)


def regression_summary(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ep in episodes:
        logs = ep["trajectory"]
        test_pass_regressions = 0
        runnability_regressions = 0
        transitions = 0
        for a, b in zip(logs, logs[1:]):
            transitions += 1
            if b["feedback_pass_rate"] < a["feedback_pass_rate"]:
                test_pass_regressions += 1
            a_state = State(a["state"])
            b_state = State(b["state"])
            if a_state in RUNNABLE_FAILURE_STATES and b_state in NON_RUNNABLE_STATES:
                runnability_regressions += 1
        rows.append(
            {
                "episode_id": ep["episode_id"],
                "problem_id": ep["problem_id"],
                "model_id": ep["model_id"],
                "transitions": transitions,
                "test_pass_regressions": test_pass_regressions,
                "runnability_regressions": runnability_regressions,
                "test_pass_regression_rate": test_pass_regressions / transitions if transitions else 0.0,
                "runnability_regression_rate": runnability_regressions / transitions if transitions else 0.0,
            }
        )
    return pd.DataFrame(rows)


def edit_distance_summary(attempts: pd.DataFrame) -> pd.DataFrame:
    """Summarize normalized edit distance between consecutive attempts, grouped by state.

    edit_distance_from_previous is None for attempt 1 (no prior code).
    Buckets: identical (0), cosmetic (0–0.05), targeted (0.05–0.20),
             rewrite (0.20–0.50), full_replacement (>0.50).
    """
    repair = attempts[attempts["edit_distance_from_previous"].notna()].copy()
    if repair.empty:
        return pd.DataFrame()

    def _bucket(d: float) -> str:
        if d == 0.0:
            return "identical"
        if d < 0.05:
            return "cosmetic"
        if d < 0.20:
            return "targeted"
        if d < 0.50:
            return "rewrite"
        return "full_replacement"

    repair["edit_bucket"] = repair["edit_distance_from_previous"].apply(_bucket)

    summary = (
        repair.groupby("state")["edit_distance_from_previous"]
        .agg(count="count", mean="mean", median="median", min="min", max="max")
        .reset_index()
        .round(4)
    )

    bucket_counts = (
        repair.groupby(["state", "edit_bucket"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["identical", "cosmetic", "targeted", "rewrite", "full_replacement"]:
        if col not in bucket_counts.columns:
            bucket_counts[col] = 0

    return summary.merge(bucket_counts, on="state", how="left")


def hidden_generalization(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    groups = defaultdict(lambda: {"count": 0, "hidden_pass": 0})
    for ep in episodes:
        states = [log["state"] for log in ep["trajectory"]]
        feedback_pass = State.FEEDBACK_PASS.value in states
        key = "FEEDBACK_PASS" if feedback_pass else "NO_FEEDBACK_PASS"
        groups[key]["count"] += 1
        groups[key]["hidden_pass"] += int(ep["final_outcome"] == "FINAL_PASS")

    rows = []
    for key, values in groups.items():
        rows.append(
            {
                "feedback_loop_group": key,
                "episodes": values["count"],
                "hidden_pass_count": values["hidden_pass"],
                "hidden_success_rate": values["hidden_pass"] / values["count"],
            }
        )
    return pd.DataFrame(rows)
