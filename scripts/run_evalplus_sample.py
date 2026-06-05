from __future__ import annotations

from pathlib import Path

from agent_repair_eval.jsonl import write_jsonl
from agent_repair_eval.llm import OpenAIChatClient
from agent_repair_eval.loaders import load_evalplus_benchmark
from agent_repair_eval.runner import run_problem_episode
from agent_repair_eval.schemas import SandboxConfig, to_jsonable


def main() -> None:
    # Start small. For a real paper, remove max_problems and choose a fixed model/version.
    problems = load_evalplus_benchmark(
        "humaneval",
        max_problems=5,
        max_tests_per_problem=80,
    )
    llm = OpenAIChatClient(model_id="gpt-4o-mini", temperature=0.0, max_tokens=1400)
    config = SandboxConfig(timeout_seconds=3.0, memory_limit_mb=512)

    episodes = []
    for problem in problems:
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
        print(f"Finished {problem.problem_id}: {episode.final_outcome.value}")

    out_path = Path("outputs/evalplus_sample_episodes.jsonl")
    write_jsonl(out_path, episodes)
    print(f"Wrote {len(episodes)} episode logs to {out_path}")


if __name__ == "__main__":
    main()
