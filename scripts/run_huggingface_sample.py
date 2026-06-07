"""Run a small HumanEval+ sample with a local HuggingFace model (no API credits needed).

Requires:
    pip install transformers torch evalplus

Usage:
    python scripts/run_huggingface_sample.py
"""
from __future__ import annotations

from pathlib import Path

from agent_repair_eval.jsonl import write_jsonl
from agent_repair_eval.llm import LocalHuggingFaceClient
from agent_repair_eval.loaders import load_evalplus_benchmark
from agent_repair_eval.runner import run_problem_episode
from agent_repair_eval.schemas import SandboxConfig, to_jsonable


def main() -> None:
    problems = load_evalplus_benchmark(
        "humaneval",
        max_problems=20,
        max_tests_per_problem=80,
    )

    # 0.5B model — small enough to run on CPU, weak enough to produce
    # interesting failure trajectories for the evaluator.
    llm = LocalHuggingFaceClient(
        model_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        temperature=0.1,
        max_new_tokens=1200,
    )
    config = SandboxConfig(timeout_seconds=3.0, memory_limit_mb=512)

    episodes = []
    for problem in problems:
        print(f"Running {problem.problem_id}...")
        episode = run_problem_episode(
            problem,
            llm,
            max_attempts=5,
            sandbox_config=config,
            feedback_ratio=0.2,
            split_seed=42,
            feedback_policy="structured",
        )
        episodes.append(to_jsonable(episode))
        print(f"  -> {episode.final_outcome.value}")

    out_path = Path("outputs/huggingface_sample_episodes.jsonl")
    write_jsonl(out_path, episodes)
    print(f"\nWrote {len(episodes)} episode logs to {out_path}")


if __name__ == "__main__":
    main()
