"""
Similarity Algorithms and Code Comparison

Core algorithms for detecting and measuring code similarities.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from processing.languages.parse_helpers import parse_tree

from .similarity_metrics import SimilarityMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Multi-language structural extraction via the tree-sitter framework.
#
# Python keeps its proven stdlib-``ast`` extractor (byte-identical behaviour);
# every other language is parsed with its tree-sitter grammar through
# ``processing.languages.parse_helpers.parse_tree`` and walked with a single,
# language-agnostic node-type classification — no per-language regex. A
# language with no registered grammar still falls back to the generic
# regex extractor, so nothing regresses.
# ---------------------------------------------------------------------------

# Tree-sitter node types, unified across every supported grammar.
_TS_FUNCTION_TYPES = frozenset(
    {
        "function_declaration",
        "function_definition",
        "function_item",
        "method_definition",
        "method_declaration",
        "generator_function_declaration",
        "constructor_declaration",
    }
)
_TS_CLASS_TYPES = frozenset(
    {
        "class_declaration",
        "class_definition",
        "class_specifier",
        "struct_item",
        "struct_specifier",
        "enum_item",
        "enum_declaration",
        "enum_specifier",
        "type_declaration",
        "impl_item",
    }
)
_TS_INTERFACE_TYPES = frozenset({"interface_declaration", "trait_item"})
_TS_IMPORT_TYPES = frozenset(
    {
        "import_statement",
        "import_from_statement",
        "import_declaration",
        "import_spec",
        "use_declaration",
        "using_directive",
        "preproc_include",
    }
)

# Node types whose text is a usable name / module path.
_TS_NAME_TYPES = frozenset(
    {"identifier", "type_identifier", "field_identifier", "scoped_type_identifier"}
)
_TS_IMPORT_NAME_TYPES = frozenset(
    {
        "string",
        "string_literal",
        "interpreted_string_literal",
        "raw_string_literal",
        "system_lib_string",
        "dotted_name",
        "scoped_identifier",
        "qualified_name",
        "namespace_identifier",
        "package_identifier",
        "identifier",
    }
)

# Language name (and common aliases) -> a representative file extension, used
# only to drive tree-sitter grammar detection in ``parse_tree``.
_LANGUAGE_ALIASES = {
    "c#": "csharp",
    "c++": "cpp",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "golang": "go",
}
_LANGUAGE_TO_EXTENSION = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "java": ".java",
    "go": ".go",
    "rust": ".rs",
    "cpp": ".cpp",
    "csharp": ".cs",
}


def _node_text(node: object) -> str:
    """Decode a tree-sitter node's source text, tolerant of bad bytes."""
    raw = getattr(node, "text", b"") or b""
    return raw.decode("utf-8", errors="replace")


def _ts_node_name(node: object) -> str:
    """Best-effort declaration name for a tree-sitter node.

    Prefers the grammar's ``name`` field; otherwise the first identifier-like
    descendant. Empty string when no name can be found — callers treat that as
    an anonymous declaration, never an error.
    """
    name_node = node.child_by_field_name("name")  # type: ignore[attr-defined]
    if name_node is not None:
        return _node_text(name_node)
    stack = list(getattr(node, "children", []))
    while stack:
        child = stack.pop()
        if child.type in _TS_NAME_TYPES:
            return _node_text(child)
        stack.extend(child.children)
    return ""


def _ts_import_name(node: object) -> str:
    """Best-effort module/path name for a tree-sitter import node."""
    stack = list(getattr(node, "children", []))
    while stack:
        child = stack.pop()
        if child.type in _TS_IMPORT_NAME_TYPES:
            return _node_text(child).strip("\"'<>")
        stack.extend(child.children)
    return _node_text(node).splitlines()[0].strip() if _node_text(node) else ""


class CodeFeatureExtractor:
    """Extract features from code for similarity analysis."""

    @staticmethod
    def extract_structural_features(code: str, language: str) -> dict[str, Any]:
        """Extract structural features from code.

        Python is parsed with the stdlib ``ast`` (unchanged); every other
        language is parsed with its tree-sitter grammar via the shared
        framework. Languages with no grammar fall back to the generic
        regex extractor.
        """
        features = {
            "language": language,
            "line_count": len(code.splitlines()),
            "char_count": len(code),
            "functions": [],
            "classes": [],
            "imports": [],
        }

        if language.lower() == "python":
            features.update(CodeFeatureExtractor._extract_python_features(code))
        else:
            features.update(CodeFeatureExtractor._extract_framework_features(code, language))

        return features

    @staticmethod
    def _extract_framework_features(code: str, language: str) -> dict[str, Any]:
        """Extract structural features for any non-Python language via tree-sitter.

        Walks the tree-sitter syntax tree once, classifying declaration nodes
        with the unified ``_TS_*`` node-type sets — the same logic for every
        grammar. When the language has no registered grammar (``parse_tree``
        returns ``None``), falls back to :meth:`_extract_generic_features`.
        """
        normalized = _LANGUAGE_ALIASES.get(language.lower(), language.lower())
        extension = _LANGUAGE_TO_EXTENSION.get(normalized)
        tree = (
            parse_tree(Path(f"_similarity_probe{extension}"), code)
            if extension is not None
            else None
        )
        if tree is None:
            # No grammar for this language — keep the generic regex signal.
            return CodeFeatureExtractor._extract_generic_features(code)

        functions: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        interfaces: list[dict[str, Any]] = []
        imports: list[str] = []

        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            node_type = node.type
            if node_type in _TS_FUNCTION_TYPES:
                functions.append({"name": _ts_node_name(node)})
            elif node_type in _TS_CLASS_TYPES:
                classes.append({"name": _ts_node_name(node)})
            elif node_type in _TS_INTERFACE_TYPES:
                interfaces.append({"name": _ts_node_name(node)})
            elif node_type in _TS_IMPORT_TYPES:
                # Skip container nodes (e.g. Go's ``import_declaration``
                # wrapping ``import_spec``s) — the inner specs carry the names.
                if not any(child.type in _TS_IMPORT_TYPES for child in node.children):
                    name = _ts_import_name(node)
                    if name:
                        imports.append(name)
            stack.extend(node.children)

        return {
            "functions": functions,
            "classes": classes,
            "interfaces": interfaces,
            "imports": imports,
        }

    @staticmethod
    def _extract_python_features(code: str) -> dict[str, Any]:
        """Extract Python-specific features."""
        features: dict[str, list[Any]] = {
            "functions": [],
            "classes": [],
            "decorators": [],
            "imports": [],
            "exceptions": [],
        }

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    features["functions"].append(
                        {
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "decorators": [
                                d.id if hasattr(d, "id") else str(d) for d in node.decorator_list
                            ],
                            "line_number": getattr(node, "lineno", 0),
                        }
                    )
                elif isinstance(node, ast.ClassDef):
                    features["classes"].append(
                        {
                            "name": node.name,
                            "bases": [
                                base.id if hasattr(base, "id") else str(base) for base in node.bases
                            ],
                            "decorators": [
                                d.id if hasattr(d, "id") else str(d) for d in node.decorator_list
                            ],
                            "line_number": getattr(node, "lineno", 0),
                        }
                    )
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            features["imports"].append(alias.name)
                    else:
                        features["imports"].append(node.module or "")
                elif isinstance(node, ast.ExceptHandler):
                    if node.type:
                        features["exceptions"].append(
                            node.type.id if hasattr(node.type, "id") else str(node.type)
                        )

        except SyntaxError:
            logger.warning("Failed to parse Python code for feature extraction")

        return features

    @staticmethod
    def _extract_generic_features(code: str) -> dict[str, Any]:
        """Extract generic features from any language."""
        features: dict[str, list[str]] = {
            "identifiers": [],
            "keywords": [],
            "operators": [],
            "literals": [],
        }

        # Common patterns
        identifier_pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
        features["identifiers"] = list(set(re.findall(identifier_pattern, code)))

        # Common operators
        operator_pattern = r"[+\-*/=<>!&|^%]+"
        features["operators"] = list(set(re.findall(operator_pattern, code)))

        # String literals
        string_pattern = r'["\'][^"\']*["\']'
        features["literals"] = list(set(re.findall(string_pattern, code)))

        return features

    @staticmethod
    def extract_semantic_features(code: str, language: str) -> dict[str, Any]:
        """Extract semantic features from code."""
        features: dict[str, list[str]] = {
            "api_calls": [],
            "data_structures": [],
            "algorithms": [],
            "design_patterns": [],
            "domain_concepts": [],
        }

        # Extract API calls (method/function calls)
        api_pattern = r"(\w+)\s*\("
        features["api_calls"] = list(set(re.findall(api_pattern, code)))

        # Data structure patterns
        data_structure_patterns = {
            "list": r"\b(?:list|array|ArrayList|List)\b",
            "map": r"\b(?:map|dict|HashMap|Map|dictionary)\b",
            "set": r"\b(?:set|Set|HashSet)\b",
            "queue": r"\b(?:queue|Queue|LinkedList)\b",
            "stack": r"\b(?:stack|Stack)\b",
        }

        for ds_type, pattern in data_structure_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                features["data_structures"].append(ds_type)

        # Algorithm patterns
        algorithm_patterns = {
            "sorting": r"\b(?:sort|Sort|sorted|quicksort|mergesort|heapsort)\b",
            "searching": r"\b(?:search|Search|find|binary_search|linear_search)\b",
            "hashing": r"\b(?:hash|Hash|md5|sha|digest)\b",
            "encryption": r"\b(?:encrypt|decrypt|cipher|crypto|ssl|tls)\b",
            "parsing": r"\b(?:parse|Parse|parser|json|xml|yaml)\b",
        }

        for algo_type, pattern in algorithm_patterns.items():
            if re.search(pattern, code, re.IGNORECASE):
                features["algorithms"].append(algo_type)

        # Design pattern indicators
        pattern_indicators = {
            "singleton": r"\b(?:getInstance|instance|Singleton)\b",
            "factory": r"\b(?:Factory|create|builder|Builder)\b",
            "observer": r"\b(?:Observer|notify|subscribe|listener)\b",
            "strategy": r"\b(?:Strategy|algorithm|execute)\b",
            "decorator": r"\b(?:Decorator|wrapper|wrap)\b",
        }

        for pattern_type, pattern in pattern_indicators.items():
            if re.search(pattern, code, re.IGNORECASE):
                features["design_patterns"].append(pattern_type)

        return features

    @staticmethod
    def extract_syntactic_features(code: str) -> dict[str, Any]:
        """Extract syntactic features from code."""
        features: dict[str, Any] = {
            "indentation_style": "spaces",  # or "tabs"
            "line_length_avg": 0,
            "bracket_style": "unknown",
            "naming_conventions": [],
            "comment_ratio": 0.0,
            "whitespace_patterns": [],
        }

        lines = code.splitlines()

        # Analyze indentation
        indent_chars = []
        for line in lines:
            if line.strip() and line.startswith((" ", "\t")):
                indent_chars.append(line[0])

        if indent_chars:
            tab_count = indent_chars.count("\t")
            space_count = indent_chars.count(" ")
            features["indentation_style"] = "tabs" if tab_count > space_count else "spaces"

        # Average line length
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            features["line_length_avg"] = sum(len(line) for line in non_empty_lines) / len(
                non_empty_lines
            )

        # Naming conventions
        camel_case_pattern = r"\b[a-z][a-zA-Z0-9]*\b"
        snake_case_pattern = r"\b[a-z][a-z0-9_]*\b"
        pascal_case_pattern = r"\b[A-Z][a-zA-Z0-9]*\b"

        if re.search(camel_case_pattern, code):
            features["naming_conventions"].append("camelCase")
        if re.search(snake_case_pattern, code):
            features["naming_conventions"].append("snake_case")
        if re.search(pascal_case_pattern, code):
            features["naming_conventions"].append("PascalCase")

        # Comment ratio
        comment_lines = len(
            [line for line in lines if line.strip().startswith(("#", "//", "/*", "*"))]
        )
        total_lines = len([line for line in lines if line.strip()])
        if total_lines > 0:
            features["comment_ratio"] = comment_lines / total_lines

        return features


class SimilarityAlgorithms:
    """Core similarity detection algorithms."""

    @staticmethod
    def compare_structural_similarity(
        features_a: dict[str, Any], features_b: dict[str, Any]
    ) -> float:
        """Compare structural similarity between two code segments."""
        scores = []

        # Compare function signatures
        funcs_a = set(f.get("name", "") for f in features_a.get("functions", []))
        funcs_b = set(f.get("name", "") for f in features_b.get("functions", []))
        if funcs_a or funcs_b:
            func_similarity = SimilarityMetrics.jaccard_similarity(funcs_a, funcs_b)
            scores.append(func_similarity * 0.3)

        # Compare class definitions
        classes_a = set(c.get("name", "") for c in features_a.get("classes", []))
        classes_b = set(c.get("name", "") for c in features_b.get("classes", []))
        if classes_a or classes_b:
            class_similarity = SimilarityMetrics.jaccard_similarity(classes_a, classes_b)
            scores.append(class_similarity * 0.2)

        # Compare imports
        imports_a = set(features_a.get("imports", []))
        imports_b = set(features_b.get("imports", []))
        if imports_a or imports_b:
            import_similarity = SimilarityMetrics.jaccard_similarity(imports_a, imports_b)
            scores.append(import_similarity * 0.2)

        # Compare complexity indicators
        complexity_a = features_a.get("line_count", 0)
        complexity_b = features_b.get("line_count", 0)
        if complexity_a > 0 and complexity_b > 0:
            complexity_ratio = min(complexity_a, complexity_b) / max(complexity_a, complexity_b)
            scores.append(complexity_ratio * 0.3)

        return sum(scores) if scores else 0.0

    @staticmethod
    def compare_semantic_similarity(
        features_a: dict[str, Any], features_b: dict[str, Any]
    ) -> float:
        """Compare semantic similarity between two code segments."""
        scores = []

        # Compare API calls
        api_a = set(features_a.get("api_calls", []))
        api_b = set(features_b.get("api_calls", []))
        if api_a or api_b:
            api_similarity = SimilarityMetrics.jaccard_similarity(api_a, api_b)
            scores.append(api_similarity * 0.4)

        # Compare data structures
        ds_a = set(features_a.get("data_structures", []))
        ds_b = set(features_b.get("data_structures", []))
        if ds_a or ds_b:
            ds_similarity = SimilarityMetrics.jaccard_similarity(ds_a, ds_b)
            scores.append(ds_similarity * 0.2)

        # Compare algorithms
        algo_a = set(features_a.get("algorithms", []))
        algo_b = set(features_b.get("algorithms", []))
        if algo_a or algo_b:
            algo_similarity = SimilarityMetrics.jaccard_similarity(algo_a, algo_b)
            scores.append(algo_similarity * 0.3)

        # Compare design patterns
        pattern_a = set(features_a.get("design_patterns", []))
        pattern_b = set(features_b.get("design_patterns", []))
        if pattern_a or pattern_b:
            pattern_similarity = SimilarityMetrics.jaccard_similarity(pattern_a, pattern_b)
            scores.append(pattern_similarity * 0.1)

        return sum(scores) if scores else 0.0

    @staticmethod
    def compare_syntactic_similarity(
        features_a: dict[str, Any], features_b: dict[str, Any]
    ) -> float:
        """
        Compare syntactic similarity between two code segments.

        Args:
            features_a: Syntactic features of first code segment
            features_b: Syntactic features of second code segment

        Returns:
            Syntactic similarity score (0.0 to 1.0)
        """
        scores = []

        # Compare naming conventions
        naming_a = set(features_a.get("naming_conventions", []))
        naming_b = set(features_b.get("naming_conventions", []))
        if naming_a or naming_b:
            naming_similarity = SimilarityMetrics.jaccard_similarity(naming_a, naming_b)
            scores.append(naming_similarity * 0.3)

        # Compare indentation style
        indent_a = features_a.get("indentation_style", "")
        indent_b = features_b.get("indentation_style", "")
        if indent_a and indent_b:
            indent_match = 1.0 if indent_a == indent_b else 0.0
            scores.append(indent_match * 0.2)

        # Compare line length similarity
        length_a = features_a.get("line_length_avg", 0)
        length_b = features_b.get("line_length_avg", 0)
        if length_a > 0 and length_b > 0:
            length_ratio = min(length_a, length_b) / max(length_a, length_b)
            scores.append(length_ratio * 0.2)

        # Compare comment ratio
        comment_a = features_a.get("comment_ratio", 0)
        comment_b = features_b.get("comment_ratio", 0)
        if comment_a > 0 or comment_b > 0:
            max_comment = max(comment_a, comment_b)
            if max_comment > 0:
                comment_similarity = 1.0 - abs(comment_a - comment_b) / max_comment
            else:
                comment_similarity = 1.0
            scores.append(comment_similarity * 0.3)

        return sum(scores) if scores else 0.0

    @staticmethod
    def detect_exact_duplicates(code_a: str, code_b: str) -> float:
        """
        Detect exact code duplicates using normalized comparison.

        Args:
            code_a: First code segment
            code_b: Second code segment

        Returns:
            Exact duplicate score (0.0 to 1.0)
        """
        # Normalize code by removing whitespace and comments
        normalized_a = SimilarityAlgorithms._normalize_code(code_a)
        normalized_b = SimilarityAlgorithms._normalize_code(code_b)

        if not normalized_a or not normalized_b:
            return 0.0

        # Compare normalized code
        if normalized_a == normalized_b:
            return 1.0

        # Use edit distance for near-exact matches
        return SimilarityMetrics.edit_distance_similarity(normalized_a, normalized_b)

    @staticmethod
    def _normalize_code(code: str) -> str:
        """Normalize code by removing whitespace and comments."""
        lines = []
        for line in code.splitlines():
            # Remove comments (basic patterns)
            line = re.sub(r"//.*$", "", line)  # C-style comments
            line = re.sub(r"#.*$", "", line)  # Python comments

            # Remove extra whitespace
            line = re.sub(r"\s+", " ", line.strip())

            if line:
                lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def calculate_code_hash(code: str) -> str:
        """Calculate hash of normalized code for fast comparison."""
        normalized = SimilarityAlgorithms._normalize_code(code)
        return hashlib.md5(normalized.encode()).hexdigest()  # nosec B324

    @staticmethod
    def find_similar_blocks(
        code_a: str, code_b: str, min_lines: int = 3
    ) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
        """
        Find similar code blocks between two code segments.

        Args:
            code_a: First code segment
            code_b: Second code segment
            min_lines: Minimum number of lines for a block

        Returns:
            List of tuples: ((start_a, end_a), (start_b, end_b), similarity_score)
        """
        lines_a = code_a.splitlines()
        lines_b = code_b.splitlines()

        similar_blocks = []

        # Compare all possible blocks
        for i in range(len(lines_a) - min_lines + 1):
            for j in range(len(lines_b) - min_lines + 1):
                for length in range(min_lines, min(len(lines_a) - i, len(lines_b) - j) + 1):
                    block_a = "\n".join(lines_a[i : i + length])
                    block_b = "\n".join(lines_b[j : j + length])

                    similarity = SimilarityAlgorithms.detect_exact_duplicates(block_a, block_b)

                    if similarity > 0.8:  # High similarity threshold
                        similar_blocks.append(
                            ((i, i + length - 1), (j, j + length - 1), similarity)
                        )

        # Remove overlapping blocks, keeping the highest scoring ones
        return SimilarityAlgorithms._remove_overlapping_blocks(similar_blocks)

    @staticmethod
    def _remove_overlapping_blocks(
        blocks: list[tuple[tuple[int, int], tuple[int, int], float]],
    ) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
        """Remove overlapping blocks, keeping the highest scoring ones."""
        # Sort by score (descending)
        blocks.sort(key=lambda x: x[2], reverse=True)

        non_overlapping = []
        used_ranges_a: set[int] = set()
        used_ranges_b: set[int] = set()

        for block in blocks:
            (start_a, end_a), (start_b, end_b), score = block

            # Check for overlap
            range_a = set(range(start_a, end_a + 1))
            range_b = set(range(start_b, end_b + 1))

            if not (range_a & used_ranges_a) and not (range_b & used_ranges_b):
                non_overlapping.append(block)
                used_ranges_a.update(range_a)
                used_ranges_b.update(range_b)

        return non_overlapping


def calculate_similarity(code_a: str, code_b: str, language: str = "python") -> dict[str, float]:
    """
    Calculate comprehensive similarity between two code segments.

    Args:
        code_a: First code segment
        code_b: Second code segment
        language: Programming language of the code

    Returns:
        Dictionary containing different similarity scores
    """
    try:
        # Extract features
        features_a_structural = CodeFeatureExtractor.extract_structural_features(code_a, language)
        features_b_structural = CodeFeatureExtractor.extract_structural_features(code_b, language)

        features_a_semantic = CodeFeatureExtractor.extract_semantic_features(code_a, language)
        features_b_semantic = CodeFeatureExtractor.extract_semantic_features(code_b, language)

        features_a_syntactic = CodeFeatureExtractor.extract_syntactic_features(code_a)
        features_b_syntactic = CodeFeatureExtractor.extract_syntactic_features(code_b)

        # Calculate different similarity scores
        structural_score = SimilarityAlgorithms.compare_structural_similarity(
            features_a_structural, features_b_structural
        )
        semantic_score = SimilarityAlgorithms.compare_semantic_similarity(
            features_a_semantic, features_b_semantic
        )
        syntactic_score = SimilarityAlgorithms.compare_syntactic_similarity(
            features_a_syntactic, features_b_syntactic
        )
        exact_score = SimilarityAlgorithms.detect_exact_duplicates(code_a, code_b)

        # Calculate overall score
        overall_score = (
            structural_score * 0.4
            + semantic_score * 0.3
            + syntactic_score * 0.2
            + exact_score * 0.1
        )

        return {
            "overall": overall_score,
            "structural": structural_score,
            "semantic": semantic_score,
            "syntactic": syntactic_score,
            "exact": exact_score,
        }

    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return {
            "overall": 0.0,
            "structural": 0.0,
            "semantic": 0.0,
            "syntactic": 0.0,
            "exact": 0.0,
        }


def find_similar_functions(
    functions_a: list[dict[str, Any]],
    functions_b: list[dict[str, Any]],
    language: str = "python",
) -> list[dict[str, Any]]:
    """
    Find similar functions between two sets of function definitions.

    Args:
        functions_a: List of function definitions from first code segment
        functions_b: List of function definitions from second code segment
        language: Programming language

    Returns:
        List of similar function pairs with similarity scores
    """
    similar_pairs = []

    try:
        for func_a in functions_a:
            for func_b in functions_b:
                # Create mock code segments for comparison
                code_a = f"def {func_a.get('name', 'unknown')}({', '.join(func_a.get('args', []))}):\n    pass"
                code_b = f"def {func_b.get('name', 'unknown')}({', '.join(func_b.get('args', []))}):\n    pass"

                similarity = calculate_similarity(code_a, code_b, language)

                if similarity["overall"] > 0.6:  # Threshold for similar functions
                    similar_pairs.append(
                        {
                            "function_a": func_a,
                            "function_b": func_b,
                            "similarity": similarity,
                        }
                    )

        # Sort by similarity score
        similar_pairs.sort(key=lambda x: x["similarity"]["overall"], reverse=True)

    except Exception as e:
        logger.error(f"Error finding similar functions: {e}")

    return similar_pairs


def detect_code_clones(
    code_segments: list[dict[str, str]],
    language: str = "python",
    threshold: float = 0.8,
) -> list[dict[str, Any]]:
    """
    Detect code clones within a collection of code segments.

    Args:
        code_segments: List of code segments with metadata
        language: Programming language
        threshold: Similarity threshold for clone detection

    Returns:
        List of clone groups with similarity information
    """
    clones = []

    try:
        # Compare all pairs of code segments
        for i, segment_a in enumerate(code_segments):
            for j, segment_b in enumerate(code_segments[i + 1 :], i + 1):
                code_a = segment_a.get("code", "")
                code_b = segment_b.get("code", "")

                if not code_a or not code_b:
                    continue

                similarity = calculate_similarity(code_a, code_b, language)

                if similarity["overall"] >= threshold:
                    clones.append(
                        {
                            "segment_a": {"index": i, "metadata": segment_a},
                            "segment_b": {"index": j, "metadata": segment_b},
                            "similarity": similarity,
                            "clone_type": _determine_clone_type(similarity),
                        }
                    )

        # Group similar clones
        clone_groups = _group_clones(clones)

    except Exception as e:
        logger.error(f"Error detecting code clones: {e}")
        clone_groups = []

    return clone_groups


def _determine_clone_type(similarity: dict[str, float]) -> str:
    """Determine the type of code clone based on similarity scores."""
    if similarity["exact"] > 0.95:
        return "Type I (Exact)"
    elif similarity["structural"] > 0.9:
        return "Type II (Structural)"
    elif similarity["semantic"] > 0.8:
        return "Type III (Semantic)"
    else:
        return "Type IV (Functional)"


def _group_clones(clones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group related clones into clone families."""
    clone_groups: list[dict[str, Any]] = []
    processed = set()

    for i, clone in enumerate(clones):
        if i in processed:
            continue

        group = {
            "group_id": len(clone_groups),
            "clones": [clone],
            "average_similarity": clone["similarity"]["overall"],
        }

        processed.add(i)

        # Find related clones
        for j, other_clone in enumerate(clones[i + 1 :], i + 1):
            if j in processed:
                continue

            # Check if clones share segments
            if _shares_segment(clone, other_clone):
                group["clones"].append(other_clone)
                processed.add(j)

        # Update average similarity
        if len(group["clones"]) > 1:
            total_sim = sum(c["similarity"]["overall"] for c in group["clones"])
            group["average_similarity"] = total_sim / len(group["clones"])

        clone_groups.append(group)

    return clone_groups


def _shares_segment(clone_a: dict[str, Any], clone_b: dict[str, Any]) -> bool:
    """Check if two clones share a common segment."""
    segments_a = {clone_a["segment_a"]["index"], clone_a["segment_b"]["index"]}
    segments_b = {clone_b["segment_a"]["index"], clone_b["segment_b"]["index"]}

    return bool(segments_a & segments_b)
