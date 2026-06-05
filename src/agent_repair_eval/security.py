from __future__ import annotations

import ast

BANNED_IMPORT_ROOTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "glob",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "pip",
    "importlib",
    "site",
    "builtins",
    "ctypes",
    "multiprocessing",
    "threading",
    "asyncio",
}

BANNED_CALL_NAMES = {
    "open",
    "input",
    "eval",
    "exec",
    "compile",
    "breakpoint",
    "__import__",
}

BANNED_ATTR_NAMES = {
    "system",
    "popen",
    "remove",
    "unlink",
    "rmdir",
    "mkdir",
    "makedirs",
    "listdir",
    "scandir",
    "walk",
    "chdir",
    "getcwd",
    "getenv",
    "environ",
}


def prescan_security(code: str) -> tuple[bool, str | None]:
    """Best-effort static safety screen.

    This is not a real security sandbox. It blocks obvious operations so the evaluator can label
    SECURITY_VIOLATION objectively. For serious untrusted-code research, run inside Docker or a VM.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BANNED_IMPORT_ROOTS:
                    return True, f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in BANNED_IMPORT_ROOTS:
                    return True, f"from {node.module} import ..."
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALL_NAMES:
                return True, f"call to {node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_ATTR_NAMES:
                return True, f"attribute call .{node.func.attr}"
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTR_NAMES:
                return True, f"attribute access .{node.attr}"

    return False, None
