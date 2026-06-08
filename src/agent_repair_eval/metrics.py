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


def recovery_by_attempt(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    """When does the model first pass — and when does recovery actually happen?

    For each attempt index k, reports how many episodes first reached
    FEEDBACK_PASS exactly at attempt k. Attempt 1 is a first-try success (not a
    recovery); attempts >= 2 are recoveries. The recovery columns are computed
    only over episodes that needed at least one repair (i.e. excluding first-try
    successes), so they answer "given the model recovers, when does it happen?"

    Columns:
        attempt                  – the attempt index k
        first_pass_count         – episodes whose first FEEDBACK_PASS was at k
        is_recovery              – False for k == 1, True otherwise
        pct_of_recoveries        – share of all recoveries that happened at k
        cumulative_pct_recoveries– running share of recoveries by attempt k
    """
    if not episodes:
        return pd.DataFrame()

    first_pass_attempt: list[int] = []
    for ep in episodes:
        for log in ep["trajectory"]:
            if log["state"] == State.FEEDBACK_PASS.value:
                first_pass_attempt.append(log["attempt"])
                break

    if not first_pass_attempt:
        return pd.DataFrame()

    counts: Counter[int] = Counter(first_pass_attempt)
    max_k = max(counts)
    total_recoveries = sum(c for k, c in counts.items() if k >= 2)

    rows = []
    running = 0
    for k in range(1, max_k + 1):
        c = counts.get(k, 0)
        is_recovery = k >= 2
        if is_recovery:
            running += c
        rows.append({
            "attempt": k,
            "first_pass_count": c,
            "is_recovery": is_recovery,
            "pct_of_recoveries": round(c / total_recoveries, 4) if (is_recovery and total_recoveries) else 0.0,
            "cumulative_pct_recoveries": round(running / total_recoveries, 4) if total_recoveries else 0.0,
        })
    return pd.DataFrame(rows)


def _per_problem_metric(episodes: list[dict[str, Any]], metric: str) -> dict[str, float]:
    """Map problem_id -> scalar metric for one run, for paired comparison."""
    out: dict[str, float] = {}
    for ep in episodes:
        pid = ep["problem_id"]
        if metric == "final_pass":
            out[pid] = 1.0 if ep["final_outcome"] == "FINAL_PASS" else 0.0
        elif metric == "feedback_success":
            states = [log["state"] for log in ep["trajectory"]]
            out[pid] = 1.0 if State.FEEDBACK_PASS.value in states else 0.0
        elif metric == "hidden_pass_rate":
            out[pid] = float(ep.get("final_hidden_pass_rate", 0.0))
        elif metric == "recovered":
            # 1 if the model recovered (first pass at attempt >= 2), else 0;
            # problems solved on attempt 1 or never solved are 0.
            first_pass = next(
                (log["attempt"] for log in ep["trajectory"]
                 if log["state"] == State.FEEDBACK_PASS.value), None
            )
            out[pid] = 1.0 if (first_pass is not None and first_pass >= 2) else 0.0
        else:
            raise ValueError(f"Unknown metric {metric!r}")
    return out


def _paired_bootstrap(diffs: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Return (ci_low, ci_high, two_sided_p) for the mean of paired differences."""
    import random as _random

    rng = _random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        resample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    # Two-sided bootstrap p-value: how often the resampled mean is on the
    # opposite side of zero from the observed mean (doubled).
    frac_le0 = sum(1 for m in means if m <= 0) / n_boot
    frac_ge0 = sum(1 for m in means if m >= 0) / n_boot
    p = min(1.0, 2.0 * min(frac_le0, frac_ge0))
    return lo, hi, p


def compare_two_runs(
    episodes_a: list[dict[str, Any]],
    episodes_b: list[dict[str, Any]],
    *,
    metric: str = "final_pass",
    label_a: str = "A",
    label_b: str = "B",
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired significance test comparing two runs on the same problems.

    Pairs episodes by problem_id (both runs must cover the same problems) and
    reports: the two means, their difference, a paired-bootstrap 95% CI and
    p-value on the difference, a Wilcoxon signed-rank p-value, and — for binary
    metrics — McNemar's exact p-value.

    metric: "final_pass" | "feedback_success" | "recovered" | "hidden_pass_rate".
    The first three are binary (McNemar applies); the last is continuous.

    Produces the defensible sentence a reviewer wants, e.g.:
        "A passes 5.0% more than B (95% CI [1.2%, 8.8%], paired bootstrap p=0.01)."
    """
    a_map = _per_problem_metric(episodes_a, metric)
    b_map = _per_problem_metric(episodes_b, metric)
    common = sorted(set(a_map) & set(b_map))
    if len(common) < 2:
        return {"error": "Need at least 2 shared problems to compare."}

    a = [a_map[p] for p in common]
    b = [b_map[p] for p in common]
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    mean_diff = mean_a - mean_b

    ci_low, ci_high, boot_p = _paired_bootstrap(diffs, n_boot=n_boot, seed=seed)

    wilcoxon_p = None
    try:
        from scipy.stats import wilcoxon
        if any(d != 0 for d in diffs):
            wilcoxon_p = float(wilcoxon(a, b).pvalue)
    except Exception:
        wilcoxon_p = None

    mcnemar_p = None
    is_binary = metric in {"final_pass", "feedback_success", "recovered"}
    if is_binary:
        # Discordant pairs: b_count = A solved & B not; c_count = B solved & A not.
        b_count = sum(1 for ai, bi in zip(a, b) if ai == 1 and bi == 0)
        c_count = sum(1 for ai, bi in zip(a, b) if ai == 0 and bi == 1)
        try:
            from scipy.stats import binomtest
            n_disc = b_count + c_count
            if n_disc > 0:
                mcnemar_p = float(binomtest(min(b_count, c_count), n_disc, 0.5).pvalue)
        except Exception:
            mcnemar_p = None

    significant = ci_low > 0 or ci_high < 0  # CI excludes zero

    return {
        "metric": metric,
        "label_a": label_a,
        "label_b": label_b,
        "n_paired_problems": len(common),
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "mean_difference": round(mean_diff, 4),
        "bootstrap_ci_low": round(ci_low, 4),
        "bootstrap_ci_high": round(ci_high, 4),
        "bootstrap_p_value": round(boot_p, 4),
        "wilcoxon_p_value": round(wilcoxon_p, 4) if wilcoxon_p is not None else None,
        "mcnemar_p_value": round(mcnemar_p, 4) if mcnemar_p is not None else None,
        "significant_at_95": bool(significant),
        "summary": (
            f"{label_a} {'>' if mean_diff >= 0 else '<'} {label_b} on '{metric}': "
            f"{mean_a:.3f} vs {mean_b:.3f}, diff {mean_diff:+.3f} "
            f"(95% CI [{ci_low:+.3f}, {ci_high:+.3f}], paired bootstrap p={boot_p:.3f}"
            + (f", Wilcoxon p={wilcoxon_p:.3f}" if wilcoxon_p is not None else "")
            + (f", McNemar p={mcnemar_p:.3f}" if mcnemar_p is not None else "")
            + ")"
        ),
    }


def summarize_seed_runs(per_seed: list[dict[str, float]]) -> pd.DataFrame:
    """Aggregate per-seed scalar metrics into mean ± sample-σ across seeds.

    per_seed: one dict per seed, mapping metric name -> value (e.g. feedback
    success rate, hidden pass rate, pass@k). Returns a table with mean, sample
    standard deviation, min, max, n_seeds, and a preformatted "mean ± σ" string
    suitable for paper tables.
    """
    import statistics

    if not per_seed:
        return pd.DataFrame()

    # Use the union of keys, preserving the order of the first seed.
    keys = list(per_seed[0].keys())
    for d in per_seed[1:]:
        for k in d:
            if k not in keys:
                keys.append(k)

    rows = []
    for key in keys:
        vals = [d[key] for d in per_seed if key in d]
        mean = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append({
            "metric": key,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "n_seeds": len(vals),
            "mean_pm_std": f"{mean:.3f} ± {std:.3f}",
        })
    return pd.DataFrame(rows)


def recovery_source_states(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    """Which error state was the model in right before it recovered?

    For every recovery (an episode that first reached FEEDBACK_PASS at attempt
    >= 2), looks at the state on the immediately preceding attempt — the error
    the model successfully fixed in one step. Reports the distribution of those
    pre-recovery error states.

    Columns:
        source_state       – the error state immediately before FEEDBACK_PASS
        recoveries         – number of recoveries that came from this state
        pct_of_recoveries  – share of all recoveries that came from this state
    """
    if not episodes:
        return pd.DataFrame()

    source_counts: Counter[str] = Counter()
    for ep in episodes:
        traj = ep["trajectory"]
        for i, log in enumerate(traj):
            if log["state"] == State.FEEDBACK_PASS.value:
                if i >= 1:  # recovered (something preceded the pass)
                    source_counts[traj[i - 1]["state"]] += 1
                break

    total = sum(source_counts.values())
    if total == 0:
        return pd.DataFrame()

    rows = [
        {
            "source_state": state,
            "recoveries": count,
            "pct_of_recoveries": round(count / total, 4),
        }
        for state, count in source_counts.most_common()
    ]
    return pd.DataFrame(rows)


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
