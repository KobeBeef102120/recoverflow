"""Tests that the repair loop feeds prior attempts and errors back to the model.

These guard the core mechanism that lets pass rate improve across attempts:
the model must see (1) its previous code, (2) the execution feedback/error, and
(3) on attempt 3+, the full history of all prior attempts.
"""
from dataclasses import dataclass, field

from agent_repair_eval.loaders import load_jsonl_benchmark
from agent_repair_eval.runner import run_problem_episode
from agent_repair_eval.schemas import SandboxConfig


@dataclass
class RecordingLLM:
    """Fake LLM that always returns the same wrong code and records every prompt."""

    model_id: str = "recorder"
    prompts: list = field(default_factory=list)

    def generate(self, prompt, *, problem_id, attempt):
        self.prompts.append((attempt, prompt))
        return "```python\ndef add_numbers(a, b):\n    return a - b\n```"


def _run():
    problem = load_jsonl_benchmark("data/demo_benchmark.jsonl")[0]
    llm = RecordingLLM()
    run_problem_episode(
        problem, llm, max_attempts=3,
        # Generous timeout: the sandbox spawns a fresh Python subprocess, whose
        # startup can exceed a tight limit on a loaded/Windows machine. The code
        # under test is trivial arithmetic, so a large limit avoids flaky TIMEOUTs.
        sandbox_config=SandboxConfig(timeout_seconds=30.0),
        feedback_ratio=0.4, split_seed=42,
    )
    return {a: p for a, p in llm.prompts}


def test_first_prompt_has_no_feedback():
    prompts = _run()
    assert "previous" not in prompts[1].lower()
    assert "add_numbers" in prompts[1]


def test_second_prompt_includes_previous_code_and_error():
    prompts = _run()
    p2 = prompts[2]
    assert "return a - b" in p2            # the previous attempt's code
    assert "Expected output" in p2          # the objective feedback
    assert "Actual output" in p2
    assert "WRONG_ALGORITHM" in p2          # the specific error category


def test_third_prompt_includes_full_history():
    prompts = _run()
    p3 = prompts[3]
    assert "Prior attempts" in p3           # history section present
    assert "Attempt 1:" in p3               # earlier attempt explicitly listed
    # The previous code appears for both the history entry and the most-recent block.
    assert p3.count("return a - b") >= 2
