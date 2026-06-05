from __future__ import annotations

from pathlib import Path

from agent_repair_eval.jsonl import write_jsonl
from agent_repair_eval.llm import ScriptedLLMClient
from agent_repair_eval.loaders import load_jsonl_benchmark
from agent_repair_eval.runner import run_problem_episode
from agent_repair_eval.schemas import SandboxConfig, to_jsonable


def main() -> None:
    problems = load_jsonl_benchmark("data/demo_benchmark.jsonl")
    llm = ScriptedLLMClient(
        scripts={
            "demo_add": [
                """```python\ndef add_numbers(a, b):\n    return a - b\n```""",
                """```python\ndef add_numbers(a, b):\n    return a + b\n```""",
            ],
            "demo_reverse": [
                """```python\ndef reverse_string(s):\n    return s\n```""",
                """```python\ndef reverse_string(s):\n    return s[::-1]\n```""",
            ],
        }
    )
    config = SandboxConfig(timeout_seconds=2.0, memory_limit_mb=512)
    episodes = []
    for problem in problems:
        episode = run_problem_episode(
            problem,
            llm,
            max_attempts=5,
            sandbox_config=config,
            feedback_ratio=0.4,
            split_seed=42,
            feedback_policy="structured",
        )
        episodes.append(to_jsonable(episode))

    out_path = Path("outputs/episodes.jsonl")
    write_jsonl(out_path, episodes)
    print(f"Wrote {len(episodes)} episode logs to {out_path}")
    print("Next: python -m agent_repair_eval.analysis --episodes outputs/episodes.jsonl --out reports")


if __name__ == "__main__":
    main()
