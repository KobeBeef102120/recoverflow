from __future__ import annotations

from agent_repair_eval.schemas import ExecutionResult
from agent_repair_eval.states import FeedbackType, State


def choose_feedback_type(result: ExecutionResult, policy: str = "structured") -> FeedbackType:
    policy = policy.lower().strip()
    if policy == "binary":
        return FeedbackType.BINARY_FEEDBACK
    if policy == "error_category":
        return FeedbackType.ERROR_CATEGORY_FEEDBACK
    if policy == "raw_terminal":
        return FeedbackType.RAW_TERMINAL_FEEDBACK
    if result.state == State.ASSERTION_FAILURE:
        return FeedbackType.COUNTEREXAMPLE_FEEDBACK
    return FeedbackType.STRUCTURED_FEEDBACK


def build_standardized_feedback(
    result: ExecutionResult,
    policy: str = "structured",
    timeout_seconds: float | None = None,
    memory_limit_mb: int | None = None,
) -> str:
    """Build objective feedback only from execution evidence."""
    feedback_type = choose_feedback_type(result, policy)
    header = (
        "Your previous solution did not pass the feedback tests.\n"
        "The following information is generated from objective execution results.\n"
        "Please revise the solution while preserving the required function signature."
    )

    if feedback_type == FeedbackType.BINARY_FEEDBACK:
        return header + "\n\nFeedback: The solution failed the feedback tests."

    if feedback_type == FeedbackType.ERROR_CATEGORY_FEEDBACK:
        return header + f"\n\nError category: {result.state.value}"

    if feedback_type == FeedbackType.RAW_TERMINAL_FEEDBACK:
        terminal = result.stderr_excerpt or result.stdout_excerpt or result.message or "No terminal text."
        return header + f"\n\nError category: {result.state.value}\nTerminal output:\n{terminal}"

    state = result.state
    if state == State.SYNTAX_ERROR:
        return (
            f"{header}\n\n"
            "Your previous solution could not be executed because Python raised a syntax-related error.\n"
            f"Error category: {state.value}\n"
            f"Exception type: {result.exception_type}\n"
            f"Location: line {result.line_number}, column {result.column_number}\n"
            f"Terminal message:\n{result.stderr_excerpt}\n"
            "Please revise the code so that it is syntactically valid."
        )

    if state == State.IMPORT_ERROR:
        return (
            f"{header}\n\n"
            "Your previous solution failed during import or module loading.\n"
            f"Error category: {state.value}\n"
            f"Exception type: {result.exception_type}\n"
            f"Terminal message:\n{result.stderr_excerpt}\n"
            "Please revise the solution using only libraries allowed by the benchmark environment."
        )

    if state == State.NAME_ATTRIBUTE_ERROR:
        return _exception_feedback(
            header,
            result,
            "Your previous solution raised a name or attribute error during execution.",
            "Please revise the solution so that all referenced variables, functions, methods, and attributes exist during execution.",
        )

    if state == State.TYPE_VALUE_ERROR:
        return _exception_feedback(
            header,
            result,
            "Your previous solution raised a type or value error during execution.",
            "Please revise the solution so that it handles the input types and values required by the problem.",
        )

    if state == State.INDEX_KEY_ERROR:
        return _exception_feedback(
            header,
            result,
            "Your previous solution raised an index or key error during execution.",
            "Please revise the solution so that it does not access invalid indexes or missing keys for the tested input.",
        )

    if state == State.ZERO_DIVISION_ERROR:
        return _exception_feedback(
            header,
            result,
            "Your previous solution raised a zero division error during execution.",
            "Please revise the solution so that division by zero does not occur for the tested input.",
        )

    if state == State.ASSERTION_FAILURE:
        return (
            f"{header}\n\n"
            "Your previous solution executed but failed one or more feedback tests.\n"
            f"Error category: {state.value}\n"
            f"Feedback tests passed: {result.passed}/{result.total}\n"
            "Example failing case:\n"
            f"Input:\n{result.failing_input}\n"
            f"Expected output:\n{result.expected_output}\n"
            f"Actual output:\n{result.actual_output}\n"
            "Please revise the solution using only the objective feedback above."
        )

    if state == State.TIMEOUT:
        time_info = f"{timeout_seconds} seconds" if timeout_seconds is not None else "the configured time limit"
        return (
            f"{header}\n\n"
            "Your previous solution exceeded the allowed execution time.\n"
            f"Error category: {state.value}\n"
            f"Time limit: {time_info}\n"
            f"Terminal message:\n{result.message}\n"
            "Please revise the solution so that it completes within the time limit."
        )

    if state == State.MEMORY_ERROR:
        mem_info = f"{memory_limit_mb} MB" if memory_limit_mb is not None else "the configured memory limit"
        return (
            f"{header}\n\n"
            "Your previous solution exceeded the allowed memory limit or raised a memory-related error.\n"
            f"Error category: {state.value}\n"
            f"Memory limit: {mem_info}\n"
            f"Terminal message:\n{result.stderr_excerpt or result.message}\n"
            "Please revise the solution so that it stays within the memory limit."
        )

    if state == State.OUTPUT_FORMAT_ERROR:
        return (
            f"{header}\n\n"
            "Your previous solution produced an output with an invalid format or type.\n"
            f"Error category: {state.value}\n"
            f"Expected output format or type: {result.expected_format}\n"
            f"Actual output format or type: {result.actual_format}\n"
            f"Failing input: {result.failing_input}\n"
            "Please revise the solution so that the returned output matches the required format."
        )

    if state == State.EMPTY_OR_INCOMPLETE_CODE:
        return (
            f"{header}\n\n"
            "Your previous response did not provide a complete executable solution.\n"
            f"Error category: {state.value}\n"
            f"Detected issue: {result.message}\n"
            "Please provide a complete implementation of the required function signature."
        )

    if state == State.SECURITY_VIOLATION:
        return (
            f"{header}\n\n"
            "Your previous solution attempted an operation that is not allowed in the evaluation sandbox.\n"
            f"Error category: {state.value}\n"
            f"Blocked operation: {result.blocked_operation}\n"
            "Please revise the solution using only permitted computation within the benchmark environment."
        )

    return header + f"\n\nError category: {state.value}\n{result.message or ''}"


def _exception_feedback(header: str, result: ExecutionResult, description: str, instruction: str) -> str:
    return (
        f"{header}\n\n"
        f"{description}\n"
        f"Error category: {result.state.value}\n"
        f"Exception type: {result.exception_type}\n"
        f"Failing input:\n{result.failing_input}\n"
        f"Terminal message:\n{result.stderr_excerpt}\n"
        f"{instruction}"
    )
