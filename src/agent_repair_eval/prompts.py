from __future__ import annotations


def build_initial_prompt(problem_prompt: str) -> str:
    """Return the user message for the first coding attempt."""
    return (
        "Implement the following Python function.\n"
        "Return your answer as a complete Python code block inside ```python ... ``` markers.\n"
        "Include only the function definition and any helper code — no prose or commentary.\n\n"
        f"{problem_prompt.strip()}"
    )


def build_repair_prompt(
    problem_prompt: str,
    previous_code: str,
    feedback: str,
    stderr: str | None = None,
    stdout: str | None = None,
) -> str:
    """Return the user message for a repair attempt given execution feedback."""
    parts = [
        "Your previous solution was incorrect. "
        "Study the execution feedback below and return a corrected implementation.\n"
        "Return your answer as a complete Python code block inside ```python ... ``` markers.\n"
        "Include only the function definition and any helper code — no prose, no apologies.\n\n"
        "Original problem:\n"
        f"{problem_prompt.strip()}\n\n"
        "Your previous attempt:\n"
        "```python\n"
        f"{previous_code.strip()}\n"
        "```\n\n"
        f"{feedback.strip()}",
    ]

    if stderr and stderr.strip():
        parts.append(f"\nError output:\n{stderr.strip()}")
    if stdout and stdout.strip():
        parts.append(f"\nProgram output:\n{stdout.strip()}")

    return "\n".join(parts)
