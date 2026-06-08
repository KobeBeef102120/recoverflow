"""High-level Colab-friendly API for RecoverFlow.

Usage in Google Colab
---------------------
    !pip install git+https://github.com/YOUR_USER/agent-repair-eval.git

    from agent_repair_eval.colab import run_eval
    results = run_eval("Qwen/Qwen2.5-Coder-0.5B-Instruct", n_problems=20)
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eval(
    model_id: str | None = None,
    *,
    n_problems: int = 20,
    max_attempts: int = 5,
    feedback_policy: str = "structured",
    dataset: str | None = None,
    dataset_split: str = "test",
    column_map: dict[str, str] | None = None,
    timeout_seconds: float = 5.0,
    memory_limit_mb: int = 512,
    split_seed: int = 42,
    max_tests_per_problem: int = 80,
    temperature: float = 0.7,
    max_new_tokens: int = 1200,
) -> dict[str, Any]:
    """Run the full RecoverFlow evaluation loop and display results inline.

    Parameters
    ----------
    model_id:
        Any HuggingFace model ID that supports text-generation with a chat
        template, e.g. ``"Qwen/Qwen2.5-Coder-0.5B-Instruct"``.
    n_problems:
        Number of benchmark problems to evaluate (default 20).
    max_attempts:
        Maximum feedback-repair attempts per problem (default 5).
    feedback_policy:
        One of ``"structured"``, ``"counterexample"``, ``"binary"``,
        ``"error_category"``, ``"raw_terminal"`` (default ``"structured"``).
    dataset:
        ``"humaneval"`` or ``"mbpp"`` (default ``"humaneval"``).
    timeout_seconds:
        Per-attempt execution time limit in seconds (default 5.0).
    memory_limit_mb:
        Per-attempt memory limit in MB (default 512).
    split_seed:
        Random seed for the feedback/hidden test split (default 42).
    max_tests_per_problem:
        Cap on tests loaded per problem (default 80).
    temperature:
        Sampling temperature for the model (default 0.7). Higher values make
        retries produce different code, which is needed to observe recovery;
        set to 0.0–0.1 for near-deterministic single-shot evaluation.
    max_new_tokens:
        Max new tokens to generate per attempt (default 1200).

    Returns
    -------
    dict with keys:
        ``episodes``      – raw episode dicts
        ``attempts``      – flat DataFrame of all attempt logs
        ``state_freq``    – state frequency DataFrame
        ``recovery``      – recovery-by-state DataFrame
        ``transitions``   – first-order transition table
        ``dwell``         – dwell-time summary DataFrame
        ``regression``    – regression summary DataFrame
        ``hidden``        – hidden-test generalization DataFrame
        ``pass_at_k``     – pass@k summary DataFrame
        ``hamsm_data``    – HAMSM transition dataset DataFrame
    """
    _check_colab_deps()

    from agent_repair_eval.llm import LocalHuggingFaceClient, validate_model
    from agent_repair_eval.loaders import load_evalplus_benchmark, load_huggingface_dataset
    from agent_repair_eval.runner import run_problem_episode
    from agent_repair_eval.schemas import SandboxConfig, to_jsonable
    from agent_repair_eval.metrics import (
        dwell_time_summary, edit_distance_summary, feedback_loop_success_rate,
        flatten_attempts, hidden_generalization, pass_at_k_summary,
        recovery_by_state, regression_summary, state_frequencies,
        transition_matrix, transition_table,
    )
    from agent_repair_eval.hamsm import build_transition_dataset

    # ── Interactive prompts for missing arguments ────────────────────────────
    if model_id is None:
        model_id = input(
            "Enter a HuggingFace model ID to evaluate\n"
            "  (e.g. Qwen/Qwen2.5-Coder-0.5B-Instruct): "
        ).strip()

    if dataset is None:
        print("\nDataset options:")
        print("  [1] humaneval  — HumanEval+ (built-in, recommended)")
        print("  [2] mbpp       — MBPP+ (built-in)")
        print("  [3] custom     — any HuggingFace dataset ID")
        choice = input("Choose [1/2/3] or type a HF dataset ID directly: ").strip()
        if choice in ("1", "humaneval", ""):
            dataset = "humaneval"
        elif choice in ("2", "mbpp"):
            dataset = "mbpp"
        elif choice == "3":
            dataset = input("Enter HuggingFace dataset ID (e.g. openai/openai_humaneval): ").strip()
        else:
            dataset = choice  # user typed a dataset ID directly

    _print_header(model_id, dataset, n_problems, max_attempts)

    # ── Load problems ────────────────────────────────────────────────────────
    _BUILTIN = {"humaneval", "humaneval+", "human_eval", "mbpp", "mbpp+"}
    print("Loading benchmark problems...")
    if dataset.lower() in _BUILTIN:
        problems = load_evalplus_benchmark(
            dataset,
            max_problems=n_problems,
            max_tests_per_problem=max_tests_per_problem,
        )
    else:
        problems = load_huggingface_dataset(
            dataset,
            split=dataset_split,
            max_problems=n_problems,
            max_tests_per_problem=max_tests_per_problem,
            column_map=column_map,
        )
    print(f"  Loaded {len(problems)} problems.\n")

    validate_model(model_id)

    print(f"Loading model {model_id!r} (first call downloads weights)...")
    llm = LocalHuggingFaceClient(
        model_id=model_id,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    llm._get_pipeline()  # trigger download before the loop
    print("  Model ready.\n")

    config = SandboxConfig(
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
    )

    episodes: list[dict[str, Any]] = []
    outcome_counts: dict[str, int] = {}
    for i, problem in enumerate(problems, 1):
        episode = run_problem_episode(
            problem,
            llm,
            max_attempts=max_attempts,
            sandbox_config=config,
            feedback_ratio=0.2,
            split_seed=split_seed,
            feedback_policy=feedback_policy,
        )
        ep_dict = to_jsonable(episode)
        episodes.append(ep_dict)
        outcome = episode.final_outcome.value
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        bar = _progress_bar(i, len(problems))
        print(f"\r{bar}  {i}/{len(problems)}  latest: {problem.problem_id} -> {outcome}", end="", flush=True)

    print("\n")

    # Compute all metrics
    attempts_df    = flatten_attempts(episodes)
    state_freq_df  = state_frequencies(attempts_df)
    recovery_df    = recovery_by_state(episodes)
    trans_table_df = transition_table(episodes)
    trans_matrix_df = transition_matrix(episodes)
    dwell_df       = dwell_time_summary(attempts_df)
    regression_df  = regression_summary(episodes)
    hidden_df      = hidden_generalization(episodes)
    pass_at_k_df   = pass_at_k_summary(episodes)
    hamsm_df       = build_transition_dataset(episodes, history_length=3)
    edit_dist_df   = edit_distance_summary(attempts_df)
    fb_success     = feedback_loop_success_rate(episodes)
    hidden_pass    = sum(1 for ep in episodes if ep["final_outcome"] == "FINAL_PASS") / len(episodes)

    results = {
        "episodes":    episodes,
        "attempts":    attempts_df,
        "state_freq":  state_freq_df,
        "recovery":    recovery_df,
        "transitions": trans_table_df,
        "transition_matrix": trans_matrix_df,
        "dwell":       dwell_df,
        "regression":  regression_df,
        "hidden":      hidden_df,
        "pass_at_k":   pass_at_k_df,
        "hamsm_data":  hamsm_df,
        "edit_distance": edit_dist_df,
        "fb_success":  fb_success,
        "hidden_pass": hidden_pass,
    }

    display_results(results, model_id=model_id)
    return results


def display_results(results: dict[str, Any], *, model_id: str = "") -> None:
    """Re-display all tables and charts from a results dict returned by run_eval."""
    _check_display()
    from IPython.display import display, HTML

    fb  = results["fb_success"]
    hid = results["hidden_pass"]
    n   = len(results["episodes"])

    display(HTML(_section("Summary")))
    display(HTML(
        f"<table style='font-size:15px;border-collapse:collapse'>"
        f"<tr><td style='padding:6px 16px'><b>Model</b></td><td>{model_id or 'N/A'}</td></tr>"
        f"<tr><td style='padding:6px 16px'><b>Episodes</b></td><td>{n}</td></tr>"
        f"<tr><td style='padding:6px 16px'><b>Feedback-loop success rate (B=1)</b></td>"
        f"<td>{fb:.1%}</td></tr>"
        f"<tr><td style='padding:6px 16px'><b>Hidden-test pass rate (Y=1)</b></td>"
        f"<td>{hid:.1%}</td></tr>"
        f"</table>"
    ))

    _show_table(results["state_freq"],    "State Frequencies")
    _show_table(results["recovery"],      "Recovery by State")
    _show_table(results["dwell"],         "Dwell-Time Summary")
    _show_table(results["edit_distance"], "Edit Distance Between Attempts")
    _show_table(results["pass_at_k"],     "Pass@k")
    _show_table(results["hidden"],        "Hidden-Test Generalization")

    _plot_state_frequencies(results["state_freq"])
    _plot_recovery_rates(results["recovery"])
    _plot_dwell_time(results["dwell"])
    _plot_edit_distance(results["edit_distance"])
    _show_transition_table(results["transition_matrix"])
    _plot_transition_matrix(results["transition_matrix"])
    _plot_hidden_generalization(results["hidden"])
    _plot_pass_at_k(results["pass_at_k"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_colab_deps() -> None:
    missing = []
    for pkg in ("transformers", "torch", "evalplus"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise RuntimeError(
            f"Missing packages: {missing}.\n"
            "Run:  !pip install " + " ".join(missing)
        )


def _check_display() -> None:
    try:
        from IPython.display import display  # noqa: F401
    except ImportError:
        raise RuntimeError("IPython is not available. Are you running in a notebook?")


def _print_header(model_id: str, dataset: str, n: int, k: int) -> None:
    line = "=" * 60
    print(line)
    print("  RecoverFlow Evaluation")
    print(f"  Model:    {model_id}")
    print(f"  Dataset:  {dataset}  |  Problems: {n}  |  Max attempts: {k}")
    print(line + "\n")


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _section(title: str) -> str:
    return (
        f"<h3 style='margin-top:24px;margin-bottom:4px;"
        f"border-bottom:2px solid #4A90D9;padding-bottom:4px'>{title}</h3>"
    )


def _show_table(df: pd.DataFrame, title: str) -> None:
    if df is None or df.empty:
        return
    from IPython.display import display, HTML
    display(HTML(_section(title)))
    html = df.to_html(index=False, border=0)
    display(HTML(f"""
    <style>
      .rf-table {{ border-collapse: collapse; font-size: 14px; width: 100%; }}
      .rf-table th {{ background-color: #4A90D9; color: white; padding: 7px 12px; text-align: left; }}
      .rf-table td {{ padding: 6px 12px; border-bottom: 1px solid #eee; }}
      .rf-table tr:hover td {{ background-color: #f0f7ff; }}
    </style>
    {html.replace('<table', '<table class="rf-table"')}
    """))


def _plot_state_frequencies(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    from IPython.display import display, HTML
    display(HTML(_section("State Frequencies")))
    fig, ax = plt.subplots(figsize=(11, 4))
    colors = ["#e74c3c" if s != "FEEDBACK_PASS" else "#2ecc71" for s in df["state"]]
    ax.bar(df["state"], df["frequency"], color=colors)
    ax.set_xlabel("State")
    ax.set_ylabel("Count")
    ax.set_title("How often each execution state occurred")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    plt.show()


def _plot_recovery_rates(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    from IPython.display import display, HTML
    display(HTML(_section("Recovery Rates by State")))
    x = range(len(df))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - w for i in x], df["feedback_recovery_rate_within_k"], width=w, label="Feedback Recovery", color="#3498db")
    ax.bar(x,                  df["one_step_recovery_rate"],           width=w, label="One-Step Recovery",  color="#2ecc71")
    ax.bar([i + w for i in x], df["unresolved_rate"],                  width=w, label="Unresolved Rate",    color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(df["state"], rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Recovery, one-step recovery, and unresolved rates per error state")
    ax.legend()
    fig.tight_layout()
    plt.show()


def _plot_dwell_time(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    from IPython.display import display, HTML
    display(HTML(_section("Mean Dwell Time per State")))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(df["state"], df["mean_dwell_time"], color="#9b59b6")
    ax.set_xlabel("State")
    ax.set_ylabel("Mean consecutive attempts in state")
    ax.set_title("How many attempts the model stays stuck in each state on average")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    plt.show()


def _show_transition_table(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    from IPython.display import display, HTML
    display(HTML(_section("State Transition Probabilities (%)")))

    pct = (df * 100).round(1)
    pct.insert(0, "From \\ To", pct.index)

    header_cells = "".join(f"<th>{c}</th>" for c in pct.columns)
    rows_html = ""
    for _, row in pct.iterrows():
        cells = f"<td><b>{row.iloc[0]}</b></td>"
        for val in row.iloc[1:]:
            if val >= 50:
                bg = "#1a6fad"; fg = "white"
            elif val >= 20:
                bg = "#aed6f1"; fg = "black"
            elif val > 0:
                bg = "#eaf4fb"; fg = "black"
            else:
                bg = "white"; fg = "#ccc"
            cells += f"<td style='background:{bg};color:{fg};text-align:center'>{val if val > 0 else '—'}</td>"
        rows_html += f"<tr>{cells}</tr>"

    display(HTML(f"""
    <style>
      .rf-trans {{ border-collapse: collapse; font-size: 12px; width: 100%; }}
      .rf-trans th {{ background-color: #4A90D9; color: white; padding: 6px 8px; text-align: center; }}
      .rf-trans td {{ padding: 5px 8px; border: 1px solid #eee; }}
    </style>
    <table class="rf-trans"><thead><tr>{header_cells}</tr></thead><tbody>{rows_html}</tbody></table>
    <p style="font-size:11px;color:#888">Values are % of transitions from that row state. Darker blue = more likely.</p>
    """))


def _plot_transition_matrix(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    from IPython.display import display, HTML
    display(HTML(_section("State Transition Heatmap")))
    n = len(df)
    fig, ax = plt.subplots(figsize=(max(14, n * 0.85), max(10, n * 0.7)))
    im = ax.imshow(df.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(df.columns, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df.index, fontsize=7)
    ax.set_xlabel("Next State")
    ax.set_ylabel("Current State")
    ax.set_title("Empirical transition probabilities (all states)")
    for i in range(n):
        for j in range(n):
            v = df.iloc[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Probability")
    fig.tight_layout()
    plt.show()


def _plot_hidden_generalization(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    from IPython.display import display, HTML
    display(HTML(_section("Hidden-Test Generalization")))
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71" if g == "FEEDBACK_PASS" else "#e74c3c" for g in df["feedback_loop_group"]]
    ax.bar(df["feedback_loop_group"], df["hidden_success_rate"], color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Hidden-test success rate")
    ax.set_title("Does passing feedback tests generalize to hidden tests?")
    fig.tight_layout()
    plt.show()


def _plot_edit_distance(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    from IPython.display import display, HTML
    display(HTML(_section("Edit Distance Between Attempts (by State)")))

    buckets = ["identical", "cosmetic", "targeted", "rewrite", "full_replacement"]
    colors  = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
    present = [b for b in buckets if b in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean edit distance per state
    axes[0].bar(df["state"], df["mean"], color="#9b59b6")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Mean normalized edit distance")
    axes[0].set_title("How much the model changes its code per state")
    axes[0].tick_params(axis="x", rotation=40)

    # Right: stacked bar of edit buckets per state
    bottom = [0] * len(df)
    for bucket, color in zip(present, colors):
        vals = df[bucket].tolist()
        axes[1].bar(df["state"], vals, bottom=bottom, label=bucket, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    axes[1].set_ylabel("Number of repair attempts")
    axes[1].set_title("Edit size distribution per error state")
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()


def _plot_pass_at_k(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    from IPython.display import display, HTML
    display(HTML(_section("Feedback-Loop Pass@k")))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["k"], df["feedback_pass_at_k"], marker="o", label="Feedback pass@k", color="#3498db")
    ax.axhline(df["hidden_pass_rate"].iloc[0], linestyle="--", color="#e74c3c", label="Hidden-test pass rate")
    ax.set_xlabel("Attempt k")
    ax.set_ylabel("Cumulative pass rate")
    ax.set_title("Fraction of problems solved by attempt k")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    plt.show()
