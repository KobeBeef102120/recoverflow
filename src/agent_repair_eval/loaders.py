from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from agent_repair_eval.jsonl import read_jsonl
from agent_repair_eval.schemas import Problem, TestCase


def load_jsonl_benchmark(path: str | Path) -> list[Problem]:
    problems: list[Problem] = []
    for row in read_jsonl(path):
        tests = [TestCase(input=t["input"], expected=t["expected"], test_id=t.get("test_id")) for t in row["tests"]]
        problems.append(
            Problem(
                problem_id=row["problem_id"],
                prompt=row["prompt"],
                entry_point=row["entry_point"],
                tests=tests,
                metadata=row.get("metadata", {}),
            )
        )
    return problems


def split_tests(
    tests: list[TestCase],
    feedback_ratio: float,
    seed: int,
    min_feedback_tests: int = 3,
) -> tuple[list[TestCase], list[TestCase]]:
    if not 0 < feedback_ratio < 1:
        raise ValueError("feedback_ratio must be between 0 and 1")
    if len(tests) < 2:
        raise ValueError("Need at least two tests to create feedback and hidden splits.")

    rng = random.Random(seed)
    indices = list(range(len(tests)))
    rng.shuffle(indices)

    feedback_count = max(min_feedback_tests, round(len(tests) * feedback_ratio))
    feedback_count = min(feedback_count, len(tests) - 1)

    feedback_indices = set(indices[:feedback_count])
    feedback_tests = [tests[i] for i in range(len(tests)) if i in feedback_indices]
    hidden_tests = [tests[i] for i in range(len(tests)) if i not in feedback_indices]
    return feedback_tests, hidden_tests


def load_evalplus_benchmark(
    dataset: str,
    *,
    max_problems: int | None = None,
    max_tests_per_problem: int | None = None,
) -> list[Problem]:
    """Load HumanEval+ or MBPP+ through evalplus and convert inputs to explicit tests.

    EvalPlus stores test inputs. This loader computes expected outputs from the canonical
    solution, then your repair evaluator can split those explicit tests into feedback/hidden
    pools. This keeps hidden tests out of the LLM feedback loop.
    """
    try:
        if dataset.lower() in {"humaneval", "humaneval+", "human_eval"}:
            from evalplus.data import get_human_eval_plus as get_dataset
        elif dataset.lower() in {"mbpp", "mbpp+"}:
            from evalplus.data import get_mbpp_plus as get_dataset
        else:
            raise ValueError("dataset must be 'humaneval' or 'mbpp'")
    except ImportError as exc:
        raise RuntimeError("Install evalplus with: pip install evalplus") from exc

    raw = get_dataset()
    problems: list[Problem] = []
    for index, (task_id, problem) in enumerate(raw.items()):
        if max_problems is not None and index >= max_problems:
            break
        tests = _evalplus_tests(problem, max_tests_per_problem=max_tests_per_problem)
        if len(tests) < 2:
            continue
        problems.append(
            Problem(
                problem_id=task_id,
                prompt=problem["prompt"],
                entry_point=problem["entry_point"],
                tests=tests,
                metadata={"dataset": dataset, "source": "evalplus"},
            )
        )
    return problems


def _evalplus_tests(problem: dict[str, Any], max_tests_per_problem: int | None) -> list[TestCase]:
    inputs = list(problem.get("base_input", [])) + list(problem.get("plus_input", []))
    if max_tests_per_problem is not None:
        inputs = inputs[:max_tests_per_problem]

    entry_point = problem["entry_point"]
    canonical_code = problem["prompt"] + "\n" + problem.get("canonical_solution", "")
    namespace: dict[str, Any] = {}
    exec(canonical_code, namespace, namespace)
    fn = namespace[entry_point]

    tests: list[TestCase] = []
    for i, inp in enumerate(inputs):
        try:
            expected = _call_with_input(fn, inp)
        except Exception:
            # Canonical solution raised on this input — skip rather than crash the load.
            continue
        if not _is_json_serializable(inp) or not _is_json_serializable(expected):
            continue
        tests.append(TestCase(input=inp, expected=expected, test_id=f"evalplus_{i}"))
    return tests


def load_huggingface_dataset(
    dataset_id: str,
    *,
    split: str = "test",
    max_problems: int | None = None,
    max_tests_per_problem: int | None = 80,
    column_map: dict[str, str] | None = None,
) -> list[Problem]:
    """Load any HuggingFace coding benchmark and convert it to RecoverFlow Problems.

    Supports two test formats found in HF datasets:
    - Assert strings: ``"def check(candidate):\\n  assert candidate(2) == 4"``
    - Structured lists: ``[{"input": [2], "expected": 4}, ...]``

    Auto-detects columns for common datasets (openai/openai_humaneval, google-research-datasets/mbpp, etc.).
    Pass ``column_map`` to override, e.g.::

        column_map={"prompt": "question", "entry_point": "func_name", "tests": "test_cases"}
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets: pip install datasets") from exc

    print(f"  Loading dataset {dataset_id!r} (split={split!r}) from HuggingFace Hub...")
    ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
    print(f"  Loaded {len(ds)} rows.\n")

    cols = ds.column_names
    cmap = _detect_columns(cols, dataset_id, column_map or {})
    _validate_column_map(cmap, cols, dataset_id)

    problems: list[Problem] = []
    for i, row in enumerate(ds):
        if max_problems is not None and len(problems) >= max_problems:
            break

        problem_id = str(row[cmap["problem_id"]]) if cmap.get("problem_id") else f"{dataset_id}/{i}"
        prompt = str(row[cmap["prompt"]])
        entry_point = str(row[cmap["entry_point"]])
        canonical = str(row[cmap["canonical_solution"]]) if cmap.get("canonical_solution") else None
        raw_tests = row[cmap["tests"]] if cmap.get("tests") else None

        tests = _parse_tests(
            raw_tests=raw_tests,
            canonical=canonical,
            prompt=prompt,
            entry_point=entry_point,
            problem_id=problem_id,
            max_tests=max_tests_per_problem,
        )
        if len(tests) < 2:
            print(f"  Skipping {problem_id}: fewer than 2 parseable tests.")
            continue

        problems.append(Problem(
            problem_id=problem_id,
            prompt=prompt,
            entry_point=entry_point,
            tests=tests,
            metadata={"dataset": dataset_id, "source": "huggingface", "split": split},
        ))

    print(f"  Converted {len(problems)} problems with usable tests.\n")
    return problems


# Column name candidates for auto-detection
_COL_CANDIDATES: dict[str, list[str]] = {
    "problem_id":        ["task_id", "problem_id", "id", "idx", "index", "question_id"],
    "prompt":            ["prompt", "text", "question", "problem", "description", "declaration"],
    "entry_point":       ["entry_point", "function_name", "func_name", "function"],
    "canonical_solution":["canonical_solution", "solution", "code", "answer", "reference_code"],
    "tests":             ["test", "tests", "test_list", "test_cases", "assert_tests", "test_code"],
}

# Known dataset-specific overrides
_DATASET_OVERRIDES: dict[str, dict[str, str]] = {
    "openai/openai_humaneval":           {"problem_id": "task_id", "tests": "test"},
    "google-research-datasets/mbpp":     {"problem_id": "task_id", "prompt": "text",
                                          "canonical_solution": "code", "tests": "test_list",
                                          "entry_point": "function_name"},
    "mbpp":                              {"problem_id": "task_id", "prompt": "text",
                                          "canonical_solution": "code", "tests": "test_list",
                                          "entry_point": "function_name"},
}


def _detect_columns(cols: list[str], dataset_id: str, overrides: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    base = _DATASET_OVERRIDES.get(dataset_id, {})
    merged = {**base, **overrides}

    for field_name, candidates in _COL_CANDIDATES.items():
        if field_name in merged:
            result[field_name] = merged[field_name]
        else:
            for c in candidates:
                if c in cols:
                    result[field_name] = c
                    break
    return result


def _validate_column_map(cmap: dict[str, str], cols: list[str], dataset_id: str) -> None:
    required = ["prompt", "entry_point"]
    missing = [f for f in required if f not in cmap]
    if missing:
        raise ValueError(
            f"Could not auto-detect columns {missing} in dataset {dataset_id!r}.\n"
            f"Available columns: {cols}\n"
            "Pass column_map= to specify them manually, e.g.:\n"
            "  run_eval(..., column_map={'prompt': 'question', 'entry_point': 'func_name'})"
        )
    if "tests" not in cmap and "canonical_solution" not in cmap:
        raise ValueError(
            f"Dataset {dataset_id!r} needs at least a 'tests' or 'canonical_solution' column.\n"
            f"Available columns: {cols}\n"
            "Pass column_map= to specify them manually."
        )


def _parse_tests(
    *,
    raw_tests: Any,
    canonical: str | None,
    prompt: str,
    entry_point: str,
    problem_id: str,
    max_tests: int | None,
) -> list[TestCase]:
    """Parse tests from either assert strings or structured lists."""
    tests: list[TestCase] = []

    if isinstance(raw_tests, str) and raw_tests.strip():
        tests = _parse_assert_string(raw_tests, entry_point)
    elif isinstance(raw_tests, list):
        for item in raw_tests:
            if isinstance(item, str):
                tests.extend(_parse_assert_string(item, entry_point))
            elif isinstance(item, dict) and "input" in item and "expected" in item:
                if _is_json_serializable(item["input"]) and _is_json_serializable(item["expected"]):
                    tests.append(TestCase(input=item["input"], expected=item["expected"],
                                          test_id=item.get("test_id")))

    # If assert parsing yielded nothing but we have a canonical solution, run it against
    # any inputs we found to generate expected outputs
    if not tests and canonical:
        tests = _tests_from_canonical(prompt, canonical, entry_point, problem_id)

    if max_tests:
        tests = tests[:max_tests]
    return tests


def _parse_assert_string(code: str, entry_point: str) -> list[TestCase]:
    """Extract (input, expected) pairs from assert statements in a code string."""
    import ast

    tests: list[TestCase] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return tests

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test_expr = node.test
        # Pattern: assert fn(args) == expected
        if not (isinstance(test_expr, ast.Compare) and len(test_expr.ops) == 1
                and isinstance(test_expr.ops[0], ast.Eq)):
            continue
        left = test_expr.left
        right = test_expr.comparators[0]

        # left must be a call to entry_point (directly or via candidate/fn alias)
        call = None
        if isinstance(left, ast.Call):
            fn_name = _call_name(left)
            if fn_name in (entry_point, "candidate", "fn", "func"):
                call = left

        if call is None:
            continue

        try:
            args = [ast.literal_eval(a) for a in call.args]
            expected = ast.literal_eval(right)
        except (ValueError, TypeError):
            continue

        # Always store as a list of positional args so _call_with_input
        # splats them correctly. Never unwrap — a single list arg like
        # below_zero([1,2,3]) must stay as [[1,2,3]], not [1,2,3].
        inp = args
        if not _is_json_serializable(inp) or not _is_json_serializable(expected):
            continue
        tests.append(TestCase(input=inp, expected=expected))

    return tests


def _call_name(node: Any) -> str:
    import ast
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _tests_from_canonical(
    prompt: str, canonical: str, entry_point: str, problem_id: str
) -> list[TestCase]:
    """Execute canonical solution to generate expected outputs (fallback)."""
    try:
        namespace: dict[str, Any] = {}
        exec(prompt + "\n" + canonical, namespace, namespace)
        fn = namespace.get(entry_point)
        if not callable(fn):
            return []
    except Exception:
        return []
    return []  # No inputs to run without test data


def _call_with_input(fn: Any, test_input: Any) -> Any:
    if isinstance(test_input, tuple):
        return fn(*copy.deepcopy(list(test_input)))
    if isinstance(test_input, list):
        return fn(*copy.deepcopy(test_input))
    if isinstance(test_input, dict):
        return fn(**copy.deepcopy(test_input))
    return fn(copy.deepcopy(test_input))


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False
