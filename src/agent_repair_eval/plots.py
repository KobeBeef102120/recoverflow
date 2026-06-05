from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_state_frequencies(state_freq: pd.DataFrame, out_path: str | Path) -> None:
    if state_freq.empty:
        return

    out_path = _ensure_parent(out_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(state_freq["state"], state_freq["frequency"])
    ax.set_title("State Frequencies")
    ax.set_xlabel("State")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_transition_matrix_heatmap(matrix: pd.DataFrame, out_path: str | Path) -> None:
    if matrix.empty:
        return

    out_path = _ensure_parent(out_path)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix.values, aspect="auto")
    ax.set_title("Transition Matrix Heatmap")
    ax.set_xlabel("Next State")
    ax.set_ylabel("Current State")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Transition Probability")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_recovery_rates(recovery_df: pd.DataFrame, out_path: str | Path) -> None:
    if recovery_df.empty:
        return

    out_path = _ensure_parent(out_path)

    x = list(range(len(recovery_df)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        [i - width for i in x],
        recovery_df["feedback_recovery_rate_within_k"],
        width=width,
        label="Feedback Recovery",
    )
    ax.bar(
        x,
        recovery_df["one_step_recovery_rate"],
        width=width,
        label="One-Step Recovery",
    )
    ax.bar(
        [i + width for i in x],
        recovery_df["unresolved_rate"],
        width=width,
        label="Unresolved Rate",
    )

    ax.set_title("Recovery Metrics by State")
    ax.set_xlabel("State")
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(recovery_df["state"], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_dwell_time_mean(dwell_df: pd.DataFrame, out_path: str | Path) -> None:
    if dwell_df.empty:
        return

    out_path = _ensure_parent(out_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(dwell_df["state"], dwell_df["mean_dwell_time"])
    ax.set_title("Mean Dwell Time by State")
    ax.set_xlabel("State")
    ax.set_ylabel("Mean Dwell Time")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_hidden_generalization(hidden_df: pd.DataFrame, out_path: str | Path) -> None:
    if hidden_df.empty:
        return

    out_path = _ensure_parent(out_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(hidden_df["feedback_loop_group"], hidden_df["hidden_success_rate"])
    ax.set_title("Hidden-Test Success by Feedback-Loop Outcome")
    ax.set_xlabel("Feedback Loop Group")
    ax.set_ylabel("Hidden Success Rate")
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
