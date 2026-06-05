from __future__ import annotations

import re

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)

_PLACEHOLDER_PATTERNS = [
    r"\bpass\b",
    r"TODO",
    r"NotImplementedError",
    r"your code here",
    r"placeholder",
]


def extract_code(response_text: str) -> str:
    """Extract Python code from an LLM response.

    Preference order:
    1. First fenced Python/code block.
    2. Whole response if no fence exists.
    """
    response_text = response_text or ""
    match = _CODE_BLOCK_RE.search(response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def looks_empty_or_incomplete(code: str, entry_point: str) -> tuple[bool, str | None]:
    stripped = (code or "").strip()
    if not stripped:
        return True, "No executable code was returned."
    if f"def {entry_point}" not in stripped:
        return True, f"Required function {entry_point!r} is missing."

    # Do not mark any use of pass as incomplete. Only flag tiny placeholder implementations.
    body_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    joined = "\n".join(body_lines).lower()
    if len(body_lines) <= 4:
        for pattern in _PLACEHOLDER_PATTERNS:
            if re.search(pattern, joined, re.IGNORECASE):
                return True, f"Detected placeholder or incomplete implementation: {pattern}"
    return False, None
