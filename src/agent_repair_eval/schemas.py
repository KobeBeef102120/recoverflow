from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent_repair_eval.states import FeedbackType, FinalOutcome, State


@dataclass(slots=True)
class TestCase:
    """A single function-level test.

    input may be:
    - list/tuple: passed as positional args
    - dict: passed as keyword args
    - scalar: passed as one positional arg
    """

    input: Any
    expected: Any
    test_id: str | None = None


@dataclass(slots=True)
class Problem:
    problem_id: str
    prompt: str
    entry_point: str
    tests: list[TestCase]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SandboxConfig:
    timeout_seconds: float = 2.0
    memory_limit_mb: int = 512
    max_stdout_chars: int = 1200
    max_stderr_chars: int = 1200


@dataclass(slots=True)
class ExecutionResult:
    state: State
    passed: int
    total: int
    runtime_ms: int
    memory_mb: float | None = None
    stdout_excerpt: str | None = None
    stderr_excerpt: str | None = None
    exception_type: str | None = None
    line_number: int | None = None
    column_number: int | None = None
    failing_input: Any = None
    expected_output: Any = None
    actual_output: Any = None
    expected_format: str | None = None
    actual_format: str | None = None
    blocked_operation: str | None = None
    message: str | None = None

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass(slots=True)
class AttemptLog:
    episode_id: str
    problem_id: str
    model_id: str
    attempt: int
    max_attempts: int
    state: State
    exception_type: str | None
    stderr_excerpt: str | None
    stdout_excerpt: str | None
    feedback_tests_passed: int
    feedback_tests_total: int
    feedback_pass_rate: float
    delta_feedback_pass_rate: float
    runtime_ms: int
    memory_mb: float | None
    feedback_type: FeedbackType
    previous_state: State | None
    dwell_time_current_state: int
    ordered_history: list[str]
    cumulative_state_counts: dict[str, int]
    code_hash: str
    failing_input: Any = None
    expected_output: Any = None
    actual_output: Any = None


@dataclass(slots=True)
class EpisodeLog:
    episode_id: str
    problem_id: str
    model_id: str
    max_attempts: int
    trajectory: list[AttemptLog]
    final_hidden_tests_passed: int
    final_hidden_tests_total: int
    final_hidden_pass_rate: float
    final_outcome: FinalOutcome
    final_code_hash: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


# JSON helpers keep enum values serializable.
def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (State, FinalOutcome, FeedbackType)):
        return obj.value
    return obj
