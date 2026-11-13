"""Base tree-sitter parser setup."""

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())


def get_parser() -> Parser:
    return Parser(PY_LANGUAGE)
