"""
Real per-entity complexity metrics computed from the entity's source span.

Each entity's span is located in a tree-sitter parse of the whole file (the
same grammar loader ``processing.symbols`` uses) and its decision points are
counted with an explicit per-grammar node-type table:

* ``cyclomatic`` = 1 + decision points (McCabe).
* ``cognitive``  = SonarSource-style: +1 per control structure plus the
  current nesting level, +1 (no nesting increment) per ``elif``/``else if``
  and per run of a boolean operator.
* ``nesting``    = maximum nesting of control structures inside the entity.
* ``loc``        = non-blank lines in the entity's span.

When the language has no grammar, the source is unavailable, the parse fails
or the entity cannot be located, the result is ``computed=False`` with a
reason - never a placeholder number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from processing import symbols


@dataclass(frozen=True)
class GrammarTable:
    """Node types that drive the metrics for one tree-sitter grammar."""

    #: Control structures: +1 cyclomatic, +1+nesting cognitive, raise nesting.
    structures: frozenset[str]
    #: Structures that branch but do not count toward cyclomatic (``switch``,
    #: ``try`` - their cases/catches are counted instead).
    containers: frozenset[str] = frozenset()
    #: +1 cyclomatic and +1 cognitive, no nesting increment (``elif``, ``catch``
    #: is a structure in Sonar, so it lives in ``structures``).
    flat: frozenset[str] = frozenset()
    #: Case labels: +1 cyclomatic unless they are the default/wildcard arm.
    cases: frozenset[str] = frozenset()
    #: Boolean operator node types, with the operator tokens that count.
    boolean_nodes: frozenset[str] = frozenset({"binary_expression"})
    boolean_ops: frozenset[str] = frozenset({"&&", "||"})
    #: Node types whose child ``if`` is an ``else if`` (Sonar: no nesting bump).
    else_nodes: frozenset[str] = frozenset({"else_clause"})


_C_LIKE_LOOPS = frozenset({"for_statement", "while_statement", "do_statement"})

TABLES: dict[str, GrammarTable] = {
    "python": GrammarTable(
        structures=frozenset(
            {
                "if_statement",
                "for_statement",
                "while_statement",
                "except_clause",
                "conditional_expression",
            }
        ),
        containers=frozenset({"match_statement"}),
        flat=frozenset({"elif_clause", "if_clause"}),
        cases=frozenset({"case_clause"}),
        boolean_nodes=frozenset({"boolean_operator"}),
        boolean_ops=frozenset({"and", "or"}),
    ),
    "javascript": GrammarTable(
        structures=_C_LIKE_LOOPS
        | frozenset({"if_statement", "for_in_statement", "catch_clause", "ternary_expression"}),
        containers=frozenset({"switch_statement"}),
        cases=frozenset({"switch_case"}),
        boolean_ops=frozenset({"&&", "||", "??"}),
    ),
    "java": GrammarTable(
        structures=_C_LIKE_LOOPS
        | frozenset(
            {"if_statement", "enhanced_for_statement", "catch_clause", "ternary_expression"}
        ),
        containers=frozenset({"switch_expression", "switch_statement"}),
        cases=frozenset({"switch_label", "switch_rule"}),
    ),
    "go": GrammarTable(
        structures=frozenset({"if_statement", "for_statement"}),
        containers=frozenset(
            {"expression_switch_statement", "type_switch_statement", "select_statement"}
        ),
        cases=frozenset({"expression_case", "type_case", "communication_case"}),
    ),
    "rust": GrammarTable(
        structures=frozenset(
            {"if_expression", "for_expression", "while_expression", "loop_expression"}
        ),
        containers=frozenset({"match_expression"}),
        cases=frozenset({"match_arm"}),
    ),
    "cpp": GrammarTable(
        structures=_C_LIKE_LOOPS
        | frozenset({"if_statement", "for_range_loop", "catch_clause", "conditional_expression"}),
        containers=frozenset({"switch_statement"}),
        cases=frozenset({"case_statement"}),
    ),
    "c_sharp": GrammarTable(
        structures=_C_LIKE_LOOPS
        | frozenset(
            {"if_statement", "foreach_statement", "catch_clause", "conditional_expression"}
        ),
        containers=frozenset({"switch_statement", "switch_expression"}),
        cases=frozenset({"switch_section", "switch_expression_arm"}),
        boolean_ops=frozenset({"&&", "||", "??"}),
    ),
}
TABLES["typescript"] = TABLES["javascript"]
TABLES[symbols.TSX] = TABLES["javascript"]

#: Rust ``loop`` nests but is not a decision (it has no condition).
_NON_DECISION_STRUCTURES = frozenset({"loop_expression"})


# Node types that ARE an entity definition, across the grammars in TABLES.
# Locating prefers the outermost of these that starts on the entity's row.
_DEFINITION_TYPES = frozenset(
    {
        # python
        "function_definition",
        "class_definition",
        "decorated_definition",
        # javascript / typescript
        "function_declaration",
        "generator_function_declaration",
        "method_definition",
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "arrow_function",
        "function_expression",
        # java / c#
        "method_declaration",
        "constructor_declaration",
        "enum_declaration",
        "record_declaration",
        "struct_declaration",
        "local_function_statement",
        "lambda_expression",
        # go
        "func_literal",
        # rust
        "function_item",
        "function_signature_item",
        "impl_item",
        "closure_expression",
    }
)

# Container bodies. A Python ``block`` starts on its first statement's row, so
# for the first method of a class the class body starts on the method's row;
# climbing into it would measure the whole class. Never return one of these.
_CONTAINER_BODY_TYPES = frozenset(
    {
        "block",
        "class_body",
        "declaration_list",
        "statement_block",
        "field_declaration_list",
        "compound_statement",
        "enum_body",
        "interface_body",
        "enum_variant_list",
    }
)


@dataclass
class BodyMetrics:
    """Metrics for one entity, or the reason they could not be computed."""

    computed: bool
    not_computed_reason: str | None = None
    cyclomatic: int = 0
    cognitive: int = 0
    nesting: int = 0
    loc: int = 0


class SourceContext:
    """One file's source + parse tree, shared by all entities of that file."""

    def __init__(self, language: str, file_path: str | None, content: str | None) -> None:
        self.grammar: str | None = None
        self.tree: Any | None = None
        self.lines: list[str] = []
        self.error: str | None = None
        self._error_rows: list[tuple[int, int]] | None = None
        if content is None:
            self.error = "source content not available"
            return
        self.lines = content.splitlines()
        try:
            self.grammar = symbols.resolve_language(language or None, file_path)
        except ValueError:
            self.error = f"no tree-sitter grammar for language '{language}'"
            return
        if self.grammar not in TABLES:
            self.error = f"no decision-point table for grammar '{self.grammar}'"
            return
        self.tree = symbols._parse(self.grammar, content.encode("utf-8"))
        if self.tree is None:
            self.error = f"tree-sitter grammar '{self.grammar}' unavailable or parse failed"

    def metrics_for(self, line_number: int | None) -> BodyMetrics:
        """Metrics for the entity whose definition starts at 1-based *line_number*."""
        if self.error is not None:
            return BodyMetrics(computed=False, not_computed_reason=self.error)
        node = self._locate(line_number)
        if node is None:
            return BodyMetrics(
                computed=False,
                not_computed_reason=f"no syntax node found at line {line_number}",
            )
        start = node.start_point[0]
        end = node.end_point[0]
        if self._span_has_parse_error(node, start, end):
            return BodyMetrics(computed=False, not_computed_reason="parse error in span")
        loc = sum(1 for line in self.lines[start : end + 1] if line.strip())
        counter = _Counter(TABLES[self.grammar or ""])
        counter.walk(node, 0)
        return BodyMetrics(
            computed=True,
            cyclomatic=1 + counter.decisions,
            cognitive=counter.cognitive,
            nesting=counter.max_nesting,
            loc=loc,
        )

    def _locate(self, line_number: int | None) -> Any | None:
        if self.tree is None or not line_number or line_number < 1:
            return None
        row = line_number - 1
        if row >= len(self.lines) or not self.lines[row].strip():
            return None
        text = self.lines[row]
        col = len(text.encode("utf-8")) - len(text.lstrip().encode("utf-8"))
        root = self.tree.root_node
        node = root.descendant_for_point_range((row, col), (row, col))
        if node is None:
            return None
        # Climb the nodes starting on this row, innermost first (name -> def ->
        # decorated def), never into a container body. The outermost DEFINITION
        # node wins; extractor end lines are not trusted (PythonEntityExtractor
        # reports the last statement's start).
        outermost = node
        definition = node if node.type in _DEFINITION_TYPES else None
        while node.parent is not None and node.parent != root:
            node = node.parent
            if node.start_point[0] != row or node.type in _CONTAINER_BODY_TYPES:
                break
            outermost = node
            if node.type in _DEFINITION_TYPES:
                definition = node
        if outermost.type in _CONTAINER_BODY_TYPES:
            return None
        return definition if definition is not None else outermost

    def _span_has_parse_error(self, node: Any, start: int, end: int) -> bool:
        """True when an ERROR/MISSING node lies in, or overlaps, rows start..end."""
        if self.tree is None or not self.tree.root_node.has_error:
            return False
        if node.has_error:
            return True
        if self._error_rows is None:
            self._error_rows = []
            stack = [self.tree.root_node]
            while stack:
                current = stack.pop()
                if current.is_error or current.is_missing:
                    self._error_rows.append((current.start_point[0], current.end_point[0]))
                    continue
                if current.has_error:
                    stack.extend(current.children)
        return any(e_start <= end and e_end >= start for e_start, e_end in self._error_rows)


class _Counter:
    def __init__(self, table: GrammarTable) -> None:
        self.table = table
        self.decisions = 0
        self.cognitive = 0
        self.max_nesting = 0

    def _is_default_case(self, node: Any) -> bool:
        if node.child_count and node.children[0].type == "default":
            return True
        if node.type == "switch_label" and node.child_count == 1:
            return node.children[0].type == "default"
        named = node.named_children
        if not named:
            return False
        first = named[0]
        return first.type in {"discard", "default_case"} or first.text.strip() == b"_"

    def _is_else_if(self, node: Any) -> bool:
        parent = node.parent
        if parent is None:
            return False
        if parent.type in self.table.else_nodes:
            return True
        if parent.type == node.type:
            prev = node.prev_sibling
            return prev is not None and prev.type == "else"
        return False

    def _boolean_operator(self, node: Any) -> str | None:
        if node.type not in self.table.boolean_nodes:
            return None
        op = node.child_by_field_name("operator")
        token = op.type if op is not None else None
        if token is None:
            for child in node.children:
                if not child.is_named and child.type in self.table.boolean_ops:
                    token = child.type
                    break
        return token if token in self.table.boolean_ops else None

    def walk(self, node: Any, nesting: int) -> None:
        table = self.table
        kind = node.type
        child_nesting = nesting
        if kind in table.structures:
            if kind not in _NON_DECISION_STRUCTURES:
                self.decisions += 1
            if self._is_else_if(node):
                self.cognitive += 1
                self.max_nesting = max(self.max_nesting, nesting)
            else:
                self.cognitive += 1 + nesting
                child_nesting = nesting + 1
                self.max_nesting = max(self.max_nesting, child_nesting)
        elif kind in table.containers:
            self.cognitive += 1 + nesting
            child_nesting = nesting + 1
            self.max_nesting = max(self.max_nesting, child_nesting)
        elif kind in table.flat:
            self.decisions += 1
            self.cognitive += 1
        elif kind in table.cases:
            if not self._is_default_case(node):
                self.decisions += 1
        else:
            op = self._boolean_operator(node)
            if op is not None:
                self.decisions += 1
                parent = node.parent
                if parent is None or self._boolean_operator(parent) != op:
                    self.cognitive += 1
        for child in node.children:
            self.walk(child, child_nesting)


__all__ = ["BodyMetrics", "GrammarTable", "SourceContext", "TABLES"]
