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
        expected = _call_with_input(fn, inp)
        if not _is_json_serializable(inp) or not _is_json_serializable(expected):
            continue
        tests.append(TestCase(input=inp, expected=expected, test_id=f"evalplus_{i}"))
    return tests


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
