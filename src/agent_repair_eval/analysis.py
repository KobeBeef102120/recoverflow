from __future__ import annotations

import argparse
from pathlib import Path

from agent_repair_eval.hamsm import (
    build_transition_dataset,
    compare_baseline_models,
    fit_count_augmented_model,
    fit_direct_recovery_model,
    fit_duration_dependent_model,
    fit_first_order_model,
    fit_higher_order_model,
    fit_transition_model,
    save_models,
)
from agent_repair_eval.jsonl import read_jsonl
from agent_repair_eval.metrics import (
    dwell_time_summary,
    feedback_loop_success_rate,
    flatten_attempts,
    hidden_generalization,
    pass_at_k_summary,
    recovery_by_state,
    regression_summary,
    state_frequencies,
    transition_matrix,
    transition_table,
)
from agent_repair_eval.plots import (
    plot_dwell_time_mean,
    plot_hidden_generalization,
    plot_recovery_rates,
    plot_state_frequencies,
    plot_transition_matrix_heatmap,
)


def analyze_episodes(episodes_path: str | Path, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    graphs_dir = out / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    episodes = read_jsonl(episodes_path)
    attempts = flatten_attempts(episodes)

    # --- Process-level metrics (Sections 7 and 11) ---
    state_freq_df = state_frequencies(attempts)
    transition_table_df = transition_table(episodes)
    transition_matrix_df = transition_matrix(episodes)
    recovery_df = recovery_by_state(episodes)
    dwell_df = dwell_time_summary(attempts)
    regression_df = regression_summary(episodes)
    hidden_df = hidden_generalization(episodes)

    # --- Pass@k summary (Section 9.1 baseline) ---
    pass_at_k_df = pass_at_k_summary(episodes)
    fb_success = feedback_loop_success_rate(episodes)
    hidden_pass = (
        sum(1 for ep in episodes if ep["final_outcome"] == "FINAL_PASS") / len(episodes)
        if episodes else 0.0
    )

    # --- Save CSVs ---
    attempts.to_csv(out / "attempts_flat.csv", index=False)
    state_freq_df.to_csv(out / "state_frequencies.csv", index=False)
    transition_table_df.to_csv(out / "transition_table.csv", index=False)
    transition_matrix_df.to_csv(out / "transition_matrix.csv")
    recovery_df.to_csv(out / "recovery_by_state.csv", index=False)
    dwell_df.to_csv(out / "dwell_time_summary.csv", index=False)
    regression_df.to_csv(out / "regression_summary.csv", index=False)
    hidden_df.to_csv(out / "hidden_generalization.csv", index=False)
    pass_at_k_df.to_csv(out / "pass_at_k.csv", index=False)

    # Print top-level summary
    n = len(episodes)
    print(f"Episodes:                   {n}")
    print(f"Feedback-loop success rate: {fb_success:.3f}  (B_n=1)")
    print(f"Hidden-test pass rate:      {hidden_pass:.3f}  (Y_n=1)")

    # --- HAMSM + baseline models (Section 9, RQ3) ---
    transitions = build_transition_dataset(episodes, history_length=3)
    transitions.to_csv(out / "hamsm_transition_dataset.csv", index=False)

    has_enough_data = (
        not transitions.empty and transitions["next_state"].nunique() >= 2
    )

    if has_enough_data:
        transition_model = fit_transition_model(transitions)
        direct_model = fit_direct_recovery_model(transitions)
        first_order = fit_first_order_model(transitions)
        higher_order = fit_higher_order_model(transitions)
        count_aug = fit_count_augmented_model(transitions)
        duration_dep = fit_duration_dependent_model(transitions)

        baseline_df = compare_baseline_models(transitions)
        baseline_df.to_csv(out / "baseline_model_comparison.csv", index=False)
        print("\nBaseline model comparison (training accuracy):")
        print(baseline_df.to_string(index=False))
        print("  Note: use cross-validation on a larger dataset for paper results.\n")

        save_models(
            out / "models",
            transition_model=transition_model,
            direct_recovery_model=direct_model,
            first_order_model=first_order,
            higher_order_model=higher_order,
            count_augmented_model=count_aug,
            duration_dependent_model=duration_dep,
        )
    else:
        print(
            "\nNot enough transition diversity to fit models "
            "(need at least 2 distinct next-state values). "
            "Run more problems or use a model that fails more often."
        )

    # --- Plots ---
    plot_state_frequencies(state_freq_df, graphs_dir / "state_frequencies.png")
    plot_transition_matrix_heatmap(transition_matrix_df, graphs_dir / "transition_matrix_heatmap.png")
    plot_recovery_rates(recovery_df, graphs_dir / "recovery_rates.png")
    plot_dwell_time_mean(dwell_df, graphs_dir / "dwell_time_mean.png")
    plot_hidden_generalization(hidden_df, graphs_dir / "hidden_generalization.png")

    print(f"Wrote CSV reports to {out}")
    print(f"Wrote graph files to {graphs_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RecoverFlow episode logs.")
    parser.add_argument("--episodes", required=True, help="Path to episodes.jsonl")
    parser.add_argument("--out", default="reports", help="Output directory for CSVs, models, and graphs")
    args = parser.parse_args()
    analyze_episodes(args.episodes, args.out)


if __name__ == "__main__":
    main()
