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
    attempt_history: list[dict] | None = None,
) -> str:
    """Return the user message for a repair attempt given execution feedback.

    attempt_history: list of dicts with keys 'attempt', 'code', 'feedback',
    'stderr', 'stdout' for all prior attempts (not including the current one).
    When provided, all previous attempts are shown before the most recent one.
    """
    parts = [
        "Your previous solution was incorrect. "
        "Study all prior attempts and their feedback below, then return a corrected implementation.\n"
        "Return your answer as a complete Python code block inside ```python ... ``` markers.\n"
        "Include only the function definition and any helper code — no prose, no apologies.\n\n"
        "Original problem:\n"
        f"{problem_prompt.strip()}\n"
    ]

    if attempt_history:
        parts.append("\n--- Prior attempts ---")
        for entry in attempt_history:
            parts.append(f"\nAttempt {entry['attempt']}:")
            parts.append("```python\n" + entry["code"].strip() + "\n```")
            parts.append(entry["feedback"].strip())
            if entry.get("stderr") and entry["stderr"].strip():
                parts.append(f"Error output:\n{entry['stderr'].strip()}")
            if entry.get("stdout") and entry["stdout"].strip():
                parts.append(f"Program output:\n{entry['stdout'].strip()}")
        parts.append("\n--- Most recent attempt ---")
    else:
        parts.append("\nYour previous attempt:")

    parts.append("```python\n" + previous_code.strip() + "\n```")
    parts.append(feedback.strip())

    if stderr and stderr.strip():
        parts.append(f"\nError output:\n{stderr.strip()}")
    if stdout and stdout.strip():
        parts.append(f"\nProgram output:\n{stdout.strip()}")

    return "\n".join(parts)
