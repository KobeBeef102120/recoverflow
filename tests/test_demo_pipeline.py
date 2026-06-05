from agent_repair_eval.llm import ScriptedLLMClient
from agent_repair_eval.loaders import load_jsonl_benchmark
from agent_repair_eval.runner import run_problem_episode
from agent_repair_eval.schemas import SandboxConfig
from agent_repair_eval.states import FinalOutcome, State


def test_demo_episode_recovers():
    problem = load_jsonl_benchmark("data/demo_benchmark.jsonl")[0]
    llm = ScriptedLLMClient(
        scripts={
            "demo_add": [
                "def add_numbers(a, b):\n    return a - b",
                "def add_numbers(a, b):\n    return a + b",
            ]
        }
    )
    episode = run_problem_episode(
        problem,
        llm,
        max_attempts=3,
        sandbox_config=SandboxConfig(timeout_seconds=2.0),
        feedback_ratio=0.4,
        split_seed=42,
    )
    states = [log.state for log in episode.trajectory]
    assert State.ASSERTION_FAILURE in states
    assert State.FEEDBACK_PASS in states
    assert episode.final_outcome == FinalOutcome.FINAL_PASS
