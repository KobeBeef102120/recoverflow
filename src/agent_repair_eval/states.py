from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class State(StrEnum):
    FEEDBACK_PASS = "FEEDBACK_PASS"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    IMPORT_ERROR = "IMPORT_ERROR"
    NAME_ATTRIBUTE_ERROR = "NAME_ATTRIBUTE_ERROR"
    TYPE_VALUE_ERROR = "TYPE_VALUE_ERROR"
    INDEX_KEY_ERROR = "INDEX_KEY_ERROR"
    ZERO_DIVISION_ERROR = "ZERO_DIVISION_ERROR"
    WRONG_ALGORITHM = "WRONG_ALGORITHM"      # runs cleanly, passes 0/N tests
    PARTIAL_PASS = "PARTIAL_PASS"            # passes 1 to N-2 tests
    NEAR_MISS = "NEAR_MISS"                  # passes N-1/N tests (one test away)
    ASSERTION_FAILURE = "ASSERTION_FAILURE"  # kept for legacy compatibility
    TIMEOUT = "TIMEOUT"
    MEMORY_ERROR = "MEMORY_ERROR"
    OUTPUT_FORMAT_ERROR = "OUTPUT_FORMAT_ERROR"
    EMPTY_OR_INCOMPLETE_CODE = "EMPTY_OR_INCOMPLETE_CODE"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    TERMINAL_UNRESOLVED = "TERMINAL_UNRESOLVED"


class FinalOutcome(StrEnum):
    FINAL_PASS = "FINAL_PASS"
    FINAL_FAIL = "FINAL_FAIL"
    FINAL_ERROR = "FINAL_ERROR"


class FeedbackType(StrEnum):
    NONE = "NONE"
    BINARY_FEEDBACK = "BINARY_FEEDBACK"
    ERROR_CATEGORY_FEEDBACK = "ERROR_CATEGORY_FEEDBACK"
    RAW_TERMINAL_FEEDBACK = "RAW_TERMINAL_FEEDBACK"
    COUNTEREXAMPLE_FEEDBACK = "COUNTEREXAMPLE_FEEDBACK"
    STRUCTURED_FEEDBACK = "STRUCTURED_FEEDBACK"


# Fixed priority for ambiguous signals.
STATE_PRIORITY: list[State] = [
    State.SECURITY_VIOLATION,
    State.EMPTY_OR_INCOMPLETE_CODE,
    State.SYNTAX_ERROR,
    State.IMPORT_ERROR,
    State.TIMEOUT,
    State.MEMORY_ERROR,
    State.NAME_ATTRIBUTE_ERROR,
    State.TYPE_VALUE_ERROR,
    State.INDEX_KEY_ERROR,
    State.ZERO_DIVISION_ERROR,
    State.OUTPUT_FORMAT_ERROR,
    State.WRONG_ALGORITHM,
    State.PARTIAL_PASS,
    State.NEAR_MISS,
    State.ASSERTION_FAILURE,
    State.FEEDBACK_PASS,
    State.TERMINAL_UNRESOLVED,
]

NON_RUNNABLE_STATES: set[State] = {
    State.SYNTAX_ERROR,
    State.IMPORT_ERROR,
    State.NAME_ATTRIBUTE_ERROR,
    State.TYPE_VALUE_ERROR,
    State.INDEX_KEY_ERROR,
    State.ZERO_DIVISION_ERROR,
    State.TIMEOUT,
    State.MEMORY_ERROR,
    State.EMPTY_OR_INCOMPLETE_CODE,
    State.SECURITY_VIOLATION,
}

RUNNABLE_FAILURE_STATES: set[State] = {
    State.WRONG_ALGORITHM,
    State.PARTIAL_PASS,
    State.NEAR_MISS,
    State.ASSERTION_FAILURE,
    State.OUTPUT_FORMAT_ERROR,
}

# Thresholds for assertion-family subdivision
WRONG_ALGORITHM_MAX_PASS_RATE = 0.0       # 0/N tests pass
NEAR_MISS_MIN_PASS_RATE_FRACTION = 1.0    # passes all but at most 1 test


def classify_assertion_state(passed: int, total: int) -> State:
    """Subdivide assertion failures by how many tests passed."""
    if total == 0:
        return State.ASSERTION_FAILURE
    if passed == 0:
        return State.WRONG_ALGORITHM
    if passed >= total - 1:
        return State.NEAR_MISS
    return State.PARTIAL_PASS

TERMINAL_STATES: set[State] = {
    State.FEEDBACK_PASS,
    State.SECURITY_VIOLATION,
    State.TERMINAL_UNRESOLVED,
}


def coerce_state(value: str | State) -> State:
    if isinstance(value, State):
        return value
    return State(value)
