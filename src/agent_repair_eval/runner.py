from __future__ import annotations

from collections import Counter

from agent_repair_eval.code_extraction import extract_code
from agent_repair_eval.feedback import build_standardized_feedback, choose_feedback_type
from agent_repair_eval.llm import LLMClient
from agent_repair_eval.loaders import split_tests
from agent_repair_eval.prompts import build_initial_prompt, build_repair_prompt
from agent_repair_eval.sandbox import execute_on_tests
from agent_repair_eval.schemas import (
    AttemptLog,
    EpisodeLog,
    ExecutionResult,
    Problem,
    SandboxConfig,
)
from agent_repair_eval.states import FeedbackType, FinalOutcome, State  # noqa: F401 (State.WRONG_ALGORITHM etc used below)
from agent_repair_eval.utils import sha256_text


def run_problem_episode(
    problem: Problem,
    llm: LLMClient,
    *,
    max_attempts: int,
    sandbox_config: SandboxConfig,
    feedback_ratio: float = 0.2,
    split_seed: int = 42,
    feedback_policy: str = "structured",
    episode_seed: int = 0,
) -> EpisodeLog:
    feedback_tests, hidden_tests = split_tests(
        problem.tests,
        feedback_ratio=feedback_ratio,
        seed=stable_split_seed(problem.problem_id, split_seed),
        min_feedback_tests=3,
    )
    episode_id = f"{problem.problem_id}_{llm.model_id}_seed{episode_seed}"
    prompt = build_initial_prompt(problem.prompt)
    trajectory: list[AttemptLog] = []
    code_history: list[str] = []
    feedback_history: list[dict] = []
    final_code = ""
    counts: Counter[str] = Counter()
    previous_state: State | None = None

    for attempt in range(1, max_attempts + 1):
        response = llm.generate(prompt, problem_id=problem.problem_id, attempt=attempt)
        candidate_code = extract_code(response)
        code_history.append(candidate_code)
        final_code = candidate_code

        result = execute_on_tests(
            code=candidate_code,
            entry_point=problem.entry_point,
            tests=feedback_tests,
            config=sandbox_config,
        )

        state = result.state
        counts[state.value] += 1
        dwell_time = _compute_dwell_time(trajectory, state)
        ordered_history = [* [log.state.value for log in trajectory], state.value]
        feedback_type = (
            FeedbackType.NONE
            if state in (State.FEEDBACK_PASS, State.SECURITY_VIOLATION)
            else choose_feedback_type(result, feedback_policy)
        )
        delta_pass_rate = _compute_delta_pass_rate(trajectory, result.pass_rate)

        log = _build_attempt_log(
            episode_id=episode_id,
            problem_id=problem.problem_id,
            model_id=llm.model_id,
            attempt=attempt,
            max_attempts=max_attempts,
            result=result,
            feedback_type=feedback_type,
            previous_state=previous_state,
            dwell_time=dwell_time,
            ordered_history=ordered_history,
            counts=counts,
            code_hash=sha256_text(candidate_code),
            delta_pass_rate=delta_pass_rate,
        )
        trajectory.append(log)

        if state == State.FEEDBACK_PASS or state == State.SECURITY_VIOLATION:
            break

        previous_state = state

        if attempt < max_attempts:
            feedback = build_standardized_feedback(
                result,
                policy=feedback_policy,
                timeout_seconds=sandbox_config.timeout_seconds,
                memory_limit_mb=sandbox_config.memory_limit_mb,
            )
            prompt = build_repair_prompt(
                problem.prompt,
                candidate_code,
                feedback,
                stderr=result.stderr_excerpt,
                stdout=result.stdout_excerpt,
                attempt_history=feedback_history if feedback_history else None,
            )
            feedback_history.append({
                "attempt": attempt,
                "code": candidate_code,
                "feedback": feedback,
                "stderr": result.stderr_excerpt,
                "stdout": result.stdout_excerpt,
            })

    if trajectory[-1].state != State.FEEDBACK_PASS and trajectory[-1].state != State.SECURITY_VIOLATION:
        counts[State.TERMINAL_UNRESOLVED.value] += 1
        terminal_history = [log.state.value for log in trajectory] + [State.TERMINAL_UNRESOLVED.value]
        terminal_result = ExecutionResult(
            state=State.TERMINAL_UNRESOLVED,
            passed=trajectory[-1].feedback_tests_passed,
            total=trajectory[-1].feedback_tests_total,
            runtime_ms=0,
            message=f"Reached max_attempts={max_attempts} without FEEDBACK_PASS.",
        )
        trajectory.append(
            _build_attempt_log(
                episode_id=episode_id,
                problem_id=problem.problem_id,
                model_id=llm.model_id,
                attempt=max_attempts,
                max_attempts=max_attempts,
                result=terminal_result,
                feedback_type=FeedbackType.NONE,
                previous_state=trajectory[-1].state,
                dwell_time=1,
                ordered_history=terminal_history,
                counts=counts,
                code_hash=sha256_text(final_code),
                delta_pass_rate=0.0,
            )
        )

    hidden_result = execute_on_tests(
        code=final_code,
        entry_point=problem.entry_point,
        tests=hidden_tests,
        config=sandbox_config,
    )
    if hidden_result.state == State.FEEDBACK_PASS:
        final_outcome = FinalOutcome.FINAL_PASS
    elif hidden_result.state in (
        State.ASSERTION_FAILURE,
        State.WRONG_ALGORITHM,
        State.PARTIAL_PASS,
        State.NEAR_MISS,
        State.OUTPUT_FORMAT_ERROR,
    ):
        final_outcome = FinalOutcome.FINAL_FAIL
    else:
        final_outcome = FinalOutcome.FINAL_ERROR

    return EpisodeLog(
        episode_id=episode_id,
        problem_id=problem.problem_id,
        model_id=llm.model_id,
        max_attempts=max_attempts,
        trajectory=trajectory,
        final_hidden_tests_passed=hidden_result.passed,
        final_hidden_tests_total=hidden_result.total,
        final_hidden_pass_rate=hidden_result.pass_rate,
        final_outcome=final_outcome,
        final_code_hash=sha256_text(final_code) if final_code else None,
        metadata={
            "feedback_tests": len(feedback_tests),
            "hidden_tests": len(hidden_tests),
            "feedback_ratio": feedback_ratio,
            "split_seed": split_seed,
            "episode_seed": episode_seed,
        },
    )


def stable_split_seed(problem_id: str, seed: int) -> int:
    # Built-in hash is salted per Python process. This gives reproducible splits.
    import hashlib

    digest = hashlib.sha256(f"{problem_id}:{seed}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _compute_dwell_time(trajectory: list[AttemptLog], state: State) -> int:
    if not trajectory or trajectory[-1].state != state:
        return 1
    return trajectory[-1].dwell_time_current_state + 1


def _compute_delta_pass_rate(trajectory: list[AttemptLog], current_pass_rate: float) -> float:
    if not trajectory:
        return 0.0
    return current_pass_rate - trajectory[-1].feedback_pass_rate


def _build_attempt_log(
    *,
    episode_id: str,
    problem_id: str,
    model_id: str,
    attempt: int,
    max_attempts: int,
    result: ExecutionResult,
    feedback_type: FeedbackType,
    previous_state: State | None,
    dwell_time: int,
    ordered_history: list[str],
    counts: Counter[str],
    code_hash: str,
    delta_pass_rate: float,
) -> AttemptLog:
    return AttemptLog(
        episode_id=episode_id,
        problem_id=problem_id,
        model_id=model_id,
        attempt=attempt,
        max_attempts=max_attempts,
        state=result.state,
        exception_type=result.exception_type,
        stderr_excerpt=result.stderr_excerpt,
        stdout_excerpt=result.stdout_excerpt,
        feedback_tests_passed=result.passed,
        feedback_tests_total=result.total,
        feedback_pass_rate=result.pass_rate,
        delta_feedback_pass_rate=delta_pass_rate,
        runtime_ms=result.runtime_ms,
        memory_mb=result.memory_mb,
        feedback_type=feedback_type,
        previous_state=previous_state,
        dwell_time_current_state=dwell_time,
        ordered_history=ordered_history,
        cumulative_state_counts=dict(counts),
        code_hash=code_hash,
        failing_input=result.failing_input,
        expected_output=result.expected_output,
        actual_output=result.actual_output,
    )
