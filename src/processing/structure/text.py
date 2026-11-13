"""Text and parsed-manifest helpers shared by the structure / contract extractors
(one copy, not one per module)."""

from __future__ import annotations

import ast
from typing import Any

#: Default length of an expression echoed back in a fact (``expression``, ``detail``).
CLIP_LIMIT = 160


def node_text(node: Any) -> str:
    """Source text of a tree-sitter node (``""`` for None / no text)."""
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", "replace")


def clip(text: str, limit: int = CLIP_LIMIT) -> str:
    """*text* with whitespace runs collapsed, cut to *limit* characters (ending ``…``)."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def unparse(node: ast.AST | None, limit: int = CLIP_LIMIT) -> str | None:
    """``ast.unparse`` clipped to *limit*; None for None or an unparsable node."""
    if node is None:
        return None
    try:
        return clip(ast.unparse(node), limit)
    except (ValueError, RecursionError):
        return None


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def get_path(data: Any, *keys: str) -> Any:
    """``data[k1][k2]...`` with a dict check at every level; ``None`` when absent."""
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur
