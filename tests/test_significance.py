"""Tests for paired significance comparison between two runs."""
from agent_repair_eval.metrics import compare_two_runs


def _ep(pid, passed):
    outcome = "FINAL_PASS" if passed else "FINAL_FAIL"
    states = ["WRONG_ALGORITHM", "FEEDBACK_PASS"] if passed else ["WRONG_ALGORITHM", "TERMINAL_UNRESOLVED"]
    return {
        "problem_id": pid,
        "final_outcome": outcome,
        "final_hidden_pass_rate": 1.0 if passed else 0.0,
        "trajectory": [{"state": s, "attempt": i + 1} for i, s in enumerate(states)],
    }


def test_large_clear_difference_is_significant():
    # A solves 18/20, B solves 6/20 on the same problems.
    a = [_ep(f"p{i}", i < 18) for i in range(20)]
    b = [_ep(f"p{i}", i < 6) for i in range(20)]
    r = compare_two_runs(a, b, metric="final_pass", n_boot=5000)
    assert r["mean_difference"] > 0
    assert r["bootstrap_ci_low"] > 0          # CI excludes zero
    assert r["significant_at_95"] is True
    assert r["wilcoxon_p_value"] < 0.05


def test_identical_runs_not_significant():
    a = [_ep(f"p{i}", i < 10) for i in range(20)]
    b = [_ep(f"p{i}", i < 10) for i in range(20)]
    r = compare_two_runs(a, b, metric="final_pass", n_boot=5000)
    assert r["mean_difference"] == 0
    assert r["bootstrap_ci_low"] <= 0 <= r["bootstrap_ci_high"]
    assert r["significant_at_95"] is False


def test_continuous_metric_runs():
    a = [_ep(f"p{i}", i < 15) for i in range(20)]
    b = [_ep(f"p{i}", i < 10) for i in range(20)]
    r = compare_two_runs(a, b, metric="hidden_pass_rate", n_boot=3000)
    assert r["mcnemar_p_value"] is None  # not a binary metric
    assert "bootstrap_ci_low" in r


def test_requires_shared_problems():
    a = [_ep("x", True)]
    b = [_ep("y", True)]
    r = compare_two_runs(a, b, metric="final_pass")
    assert "error" in r
