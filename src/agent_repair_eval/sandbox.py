from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agent_repair_eval.code_extraction import looks_empty_or_incomplete
from agent_repair_eval.schemas import ExecutionResult, SandboxConfig, TestCase
from agent_repair_eval.security import prescan_security
from agent_repair_eval.states import State


def _test_to_dict(test: TestCase) -> dict:
    return {"input": test.input, "expected": test.expected, "test_id": test.test_id}


def execute_on_tests(
    code: str,
    entry_point: str,
    tests: list[TestCase],
    config: SandboxConfig,
) -> ExecutionResult:
    """Run candidate code against tests in a child Python process.

    This is a research starter sandbox, not a production-grade isolation boundary. For a real
    paper experiment, run this process inside Docker/VM/firejail as an additional layer.
    """
    incomplete, reason = looks_empty_or_incomplete(code, entry_point)
    if incomplete:
        return ExecutionResult(
            state=State.EMPTY_OR_INCOMPLETE_CODE,
            passed=0,
            total=len(tests),
            runtime_ms=0,
            message=reason,
        )

    blocked, blocked_operation = prescan_security(code)
    if blocked:
        return ExecutionResult(
            state=State.SECURITY_VIOLATION,
            passed=0,
            total=len(tests),
            runtime_ms=0,
            blocked_operation=blocked_operation,
        )

    payload = {
        "code": code,
        "entry_point": entry_point,
        "tests": [_test_to_dict(t) for t in tests],
        "memory_limit_mb": config.memory_limit_mb,
        "max_stdout_chars": config.max_stdout_chars,
        "max_stderr_chars": config.max_stderr_chars,
    }

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="agent_repair_eval_") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / "payload.json"
        output_path = tmp_path / "result.json"
        input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, "-m", "agent_repair_eval._child_runner", str(input_path), str(output_path)],
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                state=State.TIMEOUT,
                passed=0,
                total=len(tests),
                runtime_ms=int((time.perf_counter() - started) * 1000),
                message=f"Execution exceeded {config.timeout_seconds} seconds.",
            )

        if not output_path.exists():
            stderr = (completed.stderr or completed.stdout or "Child process produced no result.")[
                : config.max_stderr_chars
            ]
            state = State.MEMORY_ERROR if completed.returncode != 0 else State.TYPE_VALUE_ERROR
            return ExecutionResult(
                state=state,
                passed=0,
                total=len(tests),
                runtime_ms=int((time.perf_counter() - started) * 1000),
                stderr_excerpt=stderr,
                message="Evaluator child process failed before writing result JSON.",
            )

        raw = json.loads(output_path.read_text(encoding="utf-8"))
        return ExecutionResult(
            state=State(raw["state"]),
            passed=int(raw.get("passed", 0)),
            total=int(raw.get("total", len(tests))),
            runtime_ms=int(raw.get("runtime_ms", 0)),
            memory_mb=raw.get("memory_mb"),
            stdout_excerpt=raw.get("stdout_excerpt"),
            stderr_excerpt=raw.get("stderr_excerpt"),
            exception_type=raw.get("exception_type"),
            line_number=raw.get("line_number"),
            column_number=raw.get("column_number"),
            failing_input=raw.get("failing_input"),
            expected_output=raw.get("expected_output"),
            actual_output=raw.get("actual_output"),
            expected_format=raw.get("expected_format"),
            actual_format=raw.get("actual_format"),
            blocked_operation=raw.get("blocked_operation"),
            message=raw.get("message"),
        )
