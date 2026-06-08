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
    # `return a - b` fails the value tests; depending on the split it lands in one
    # of the assertion-family states (WRONG_ALGORITHM / PARTIAL_PASS / NEAR_MISS /
    # ASSERTION_FAILURE) before the corrected attempt passes.
    assertion_family = {
        State.WRONG_ALGORITHM,
        State.PARTIAL_PASS,
        State.NEAR_MISS,
        State.ASSERTION_FAILURE,
    }
    assert any(s in assertion_family for s in states)
    assert State.FEEDBACK_PASS in states
    assert episode.final_outcome == FinalOutcome.FINAL_PASS
