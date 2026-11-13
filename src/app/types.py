"""Shared type definitions for coding-analysis."""

from __future__ import annotations

from typing import Literal

# Regex pattern for safe string fields (no control chars, no HTML injection)
SAFE_STRING_PATTERN = r"^[^<>&]*$"

# Regex for package version specifiers. PEP 440 dependency specifiers routinely
# contain ``<``, ``>``, ``=``, ``!``, ``~``, and commas — SAFE_STRING_PATTERN
# rejects them (e.g. ``>=3.8`` fails) and blew up every full_scan with
# ``string_pattern_mismatch`` on ScannedDependency.version. This allowlist
# keeps HTML tags out (no ``<foo>`` pairs, no ``&`` entities) while permitting
# every legal version specifier.
VERSION_PATTERN = r"^[A-Za-z0-9._\-,<>=!*~+ ]*$"

# Supported programming languages
SupportedLanguage = Literal[
    "python", "javascript", "typescript", "java", "go", "rust", "cpp", "csharp"
]

# Exit code type alias
ExitCode = int
