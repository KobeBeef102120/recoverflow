from __future__ import annotations

import os
from pathlib import Path

from agent_repair_eval.jsonl import write_jsonl
from agent_repair_eval.llm import OpenAIChatClient
from agent_repair_eval.loaders import load_jsonl_benchmark
from agent_repair_eval.runner import run_problem_episode
from agent_repair_eval.schemas import SandboxConfig, to_jsonable


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. In Git Bash, run:\n"
            'export OPENAI_API_KEY="your_api_key_here"'
        )

    problems = load_jsonl_benchmark("data/demo_benchmark.jsonl")

    llm = OpenAIChatClient(
        model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        max_tokens=1400,
    )

    config = SandboxConfig(timeout_seconds=3.0, memory_limit_mb=512)

    episodes = []

    for problem in problems:
        print(f"Running real model on problem: {problem.problem_id}")

        episode = run_problem_episode(
            problem=problem,
            llm=llm,
            max_attempts=3,
            sandbox_config=config,
            feedback_ratio=0.4,
            split_seed=42,
            feedback_policy="structured",
        )

        episodes.append(to_jsonable(episode))

        print(
            f"Finished {problem.problem_id}: "
            f"feedback_loop_pass={episode.feedback_loop_pass}, "
            f"final_outcome={episode.final_outcome.value}"
        )

    out_path = Path("outputs/demo_openai_episodes.jsonl")
    write_jsonl(out_path, episodes)

    print(f"Wrote {len(episodes)} real-model episode logs to {out_path}")
    print(
        "Next: python -m agent_repair_eval.analysis "
        "--episodes outputs/demo_openai_episodes.jsonl --out reports_openai_demo"
    )


if __name__ == "__main__":
    main()