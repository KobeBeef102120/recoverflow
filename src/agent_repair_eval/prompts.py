from __future__ import annotations


def build_initial_prompt(problem_prompt: str) -> str:
    return (
        "You are solving a Python programming benchmark problem.\n"
        "Return only a complete Python implementation. Do not include explanations.\n\n"
        f"Problem:\n{problem_prompt}\n"
    )


def build_repair_prompt(original_problem: str, previous_code: str, feedback: str) -> str:
    return (
        "You are revising a previous Python solution using objective execution feedback.\n"
        "Return only a complete corrected Python implementation. Do not include explanations.\n\n"
        f"Original problem:\n{original_problem}\n\n"
        "Previous incorrect code:\n"
        f"```python\n{previous_code}\n```\n\n"
        f"Objective execution feedback:\n{feedback}\n"
    )