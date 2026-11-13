"""Architecture quality-analysis tool execution.

The single public ``ToolExecution`` class is composed from per-language
mixins so individual tool runners stay in focused, sub-300-line modules.
Public API and method names are unchanged from the previous single-file
implementation.
"""

# Re-expose stdlib modules at package level so existing test patch targets
# (e.g. ``patch("processing.architecture.tool_execution.subprocess.run")``)
# keep working unchanged after the split.
import subprocess  # noqa: F401
import tempfile  # noqa: F401
from pathlib import Path  # noqa: F401

from ._base import BaseToolExecution
from ._cpp import CppToolsMixin
from ._csharp import CSharpToolsMixin
from ._go import GoToolsMixin
from ._java import JavaToolsMixin
from ._javascript import JavaScriptToolsMixin
from ._python import PythonToolsMixin
from ._rust import RustToolsMixin
from ._security import SecurityToolsMixin


class ToolExecution(
    PythonToolsMixin,
    JavaScriptToolsMixin,
    JavaToolsMixin,
    GoToolsMixin,
    RustToolsMixin,
    CppToolsMixin,
    CSharpToolsMixin,
    SecurityToolsMixin,
    BaseToolExecution,
):
    """Handler for executing quality analysis tools."""


__all__ = ["ToolExecution"]
