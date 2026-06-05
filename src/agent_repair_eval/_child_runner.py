from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ALLOWED_IMPORT_ROOTS = {
    "math",
    "itertools",
    "collections",
    "functools",
    "heapq",
    "bisect",
    "re",
    "string",
    "typing",
    "dataclasses",
    "statistics",
    "fractions",
    "decimal",
    "operator",
    "copy",
    "random",
    "array",
    "enum",
}


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _call_with_input(fn: Any, test_input: Any) -> Any:
    if isinstance(test_input, list):
        return fn(*copy.deepcopy(test_input))
    if isinstance(test_input, tuple):
        return fn(*copy.deepcopy(list(test_input)))
    if isinstance(test_input, dict):
        return fn(**copy.deepcopy(test_input))
    return fn(copy.deepcopy(test_input))


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"Import of module {name!r} is blocked by evaluator policy")
    return __import__(name, globals, locals, fromlist, level)


def _make_safe_builtins() -> dict[str, Any]:
    import builtins

    allowed_names = [
        "abs",
        "all",
        "any",
        "bool",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "Exception",
        "BaseException",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "NameError",
        "AttributeError",
        "ZeroDivisionError",
        "ImportError",
        "ModuleNotFoundError",
        "RuntimeError",
        "MemoryError",
        "SyntaxError",
    ]
    safe = {name: getattr(builtins, name) for name in allowed_names}
    safe["__import__"] = _safe_import
    return safe


def _state_from_exception(exc: BaseException) -> str:
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "IMPORT_ERROR"
    if isinstance(exc, (NameError, AttributeError)):
        return "NAME_ATTRIBUTE_ERROR"
    if isinstance(exc, (TypeError, ValueError)):
        return "TYPE_VALUE_ERROR"
    if isinstance(exc, (IndexError, KeyError)):
        return "INDEX_KEY_ERROR"
    if isinstance(exc, ZeroDivisionError):
        return "ZERO_DIVISION_ERROR"
    if isinstance(exc, MemoryError):
        return "MEMORY_ERROR"
    return "TYPE_VALUE_ERROR"


def _limit_memory(memory_limit_mb: int | None) -> None:
    if not memory_limit_mb:
        return
    try:
        import resource

        limit_bytes = int(memory_limit_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except Exception:
        # Windows does not have resource. Parent timeout still works.
        return


def run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    code = payload["code"]
    entry_point = payload["entry_point"]
    tests = payload["tests"]
    memory_limit_mb = payload.get("memory_limit_mb")
    max_stdout_chars = int(payload.get("max_stdout_chars", 1200))
    max_stderr_chars = int(payload.get("max_stderr_chars", 1200))

    _limit_memory(memory_limit_mb)

    started = time.perf_counter()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    result: dict[str, Any] = {
        "state": "ASSERTION_FAILURE",
        "passed": 0,
        "total": len(tests),
        "runtime_ms": 0,
        "stdout_excerpt": None,
        "stderr_excerpt": None,
        "exception_type": None,
        "line_number": None,
        "column_number": None,
        "failing_input": None,
        "expected_output": None,
        "actual_output": None,
        "expected_format": None,
        "actual_format": None,
        "blocked_operation": None,
        "message": None,
    }

    try:
        compiled = compile(code, "<candidate_solution>", "exec")
    except SyntaxError as exc:
        result.update(
            {
                "state": "SYNTAX_ERROR",
                "exception_type": type(exc).__name__,
                "line_number": exc.lineno,
                "column_number": exc.offset,
                "stderr_excerpt": str(exc)[:max_stderr_chars],
            }
        )
        result["runtime_ms"] = int((time.perf_counter() - started) * 1000)
        return result

    namespace: dict[str, Any] = {"__builtins__": _make_safe_builtins(), "__name__": "candidate"}

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(compiled, namespace, namespace)
    except BaseException as exc:
        result.update(
            {
                "state": _state_from_exception(exc),
                "exception_type": type(exc).__name__,
                "stderr_excerpt": "".join(traceback.format_exception_only(type(exc), exc))[
                    :max_stderr_chars
                ],
            }
        )
        result["runtime_ms"] = int((time.perf_counter() - started) * 1000)
        return result

    fn = namespace.get(entry_point)
    if not callable(fn):
        result.update(
            {
                "state": "EMPTY_OR_INCOMPLETE_CODE",
                "message": f"Required function {entry_point!r} was not defined as a callable.",
            }
        )
        result["runtime_ms"] = int((time.perf_counter() - started) * 1000)
        return result

    passed = 0
    for test in tests:
        test_input = test["input"]
        expected = test["expected"]
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                actual = _call_with_input(fn, test_input)
        except BaseException as exc:
            result.update(
                {
                    "state": _state_from_exception(exc),
                    "passed": passed,
                    "exception_type": type(exc).__name__,
                    "stderr_excerpt": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[
                        :max_stderr_chars
                    ],
                    "failing_input": _json_safe(test_input),
                }
            )
            result["runtime_ms"] = int((time.perf_counter() - started) * 1000)
            return result

        if type(actual) is not type(expected) and actual != expected:
            result.update(
                {
                    "state": "OUTPUT_FORMAT_ERROR",
                    "passed": passed,
                    "failing_input": _json_safe(test_input),
                    "expected_output": _json_safe(expected),
                    "actual_output": _json_safe(actual),
                    "expected_format": type(expected).__name__,
                    "actual_format": type(actual).__name__,
                }
            )
            result["runtime_ms"] = int((time.perf_counter() - started) * 1000)
            return result

        if actual == expected:
            passed += 1
        else:
            result.update(
                {
                    "state": "ASSERTION_FAILURE",
                    "passed": passed,
                    "failing_input": _json_safe(test_input),
                    "expected_output": _json_safe(expected),
                    "actual_output": _json_safe(actual),
                }
            )
            result["runtime_ms"] = int((time.perf_counter() - started) * 1000)
            return result

    result.update(
        {
            "state": "FEEDBACK_PASS",
            "passed": passed,
            "stdout_excerpt": stdout_buffer.getvalue()[:max_stdout_chars] or None,
            "stderr_excerpt": stderr_buffer.getvalue()[:max_stderr_chars] or None,
        }
    )
    result["runtime_ms"] = int((time.perf_counter() - started) * 1000)
    return result


def main() -> None:
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = run_payload(payload)
    output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
