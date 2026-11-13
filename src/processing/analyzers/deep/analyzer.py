"""DeepCodeAnalyzer orchestrator.

Thin facade that wires together every sub-analyzer in the
``processing.analyzers.deep`` package. Each ``_analyze_*`` method delegates
to the corresponding module-level ``analyze`` function; legacy helper
methods on the class are kept as one-line forwarders so existing call sites
binding to ``DeepCodeAnalyzer`` continue to work without behaviour change.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from models.core_models import CoreCodeEntity, EntityRelation
from processing.utils.ast_utils import ASTUtils

from . import (
    architecture as _architecture,
)
from . import (
    bug_risks as _bug_risks,
)
from . import (
    complexity_checks as _complexity_checks,
)
from . import (
    documentation as _documentation,
)
from . import (
    entities as _entities_mod,
)
from . import (
    metrics as _metrics_mod,
)
from . import (
    performance as _performance,
)
from . import (
    security as _security,
)
from . import (
    smells as _smells,
)
from .base import (
    CodeIssue,
    CodeRelationship,
    DeepAnalysisResult,
    IssueType,
    Severity,
    calculate_cyclomatic_complexity,
    calculate_nesting_depth,
    extract_base_name,
    extract_decorator_name,
    extract_docstring,
    get_function_signature,
    get_line,
    safe_parse_ast,
)
from .multi_language_entities import collect_source_files, extract_graph_multilang

logger = logging.getLogger(__name__)

#: Languages ``analyze_code_snippet`` analyses; any other snippet comes back
#: empty (and ``/api/analysis/capabilities`` reports exactly these).
SNIPPET_LANGUAGES: tuple[str, ...] = ("python",)


class DeepCodeAnalyzer:
    """Comprehensive static code analysis."""

    def __init__(self, codebase_path: str, language: str = "python"):
        """Initialize the DeepCodeAnalyzer service.

        Args:
            codebase_path (str): Codebase path
            language (str): Programming language (default: "python")
        """
        self.codebase_path = Path(codebase_path)
        self.language = language
        self.ast_utils = ASTUtils()
        # Initialize the data structures for analysis results.
        # ``relationships`` accepts both the Python-path ``CodeRelationship``
        # and the multi-language ``EntityRelation`` (see DeepAnalysisResult).
        self.entities: list[CoreCodeEntity] = []
        self.relationships: list[CodeRelationship | EntityRelation] = []
        self.issues: list[CodeIssue] = []
        logger.info(
            f"Initializing DeepCodeAnalyzer with path: {self.codebase_path}, language: {self.language}"
        )
        # ``python_files`` drives the Python-only deep sub-analyzers;
        # ``source_files`` drives multi-language graph extraction.
        self.python_files: list[Path] = []
        self.source_files: list[Path] = []
        if not self.codebase_path.exists():
            logger.error(f"Codebase path does not exist: {self.codebase_path}")
        else:
            self.python_files = list(self.codebase_path.rglob("*.py"))
            self.source_files = collect_source_files(self.codebase_path)
            logger.info(
                f"Found {len(self.python_files)} Python files and "
                f"{len(self.source_files)} total source files in {self.codebase_path}"
            )

    # ------------------------------------------------------------------
    # AST helpers (kept as methods to preserve the original public API)
    # ------------------------------------------------------------------
    def _safe_parse_ast(self, content: str, filename: str = "<unknown>") -> ast.AST | None:
        """Parse Python code into AST while suppressing SyntaxWarnings about escape sequences."""
        return safe_parse_ast(content, filename)

    def _extract_docstring(self, node):
        """Extract docstring from AST node."""
        return extract_docstring(node)

    def _extract_decorator_name(self, decorator):
        """Extract decorator name from AST node."""
        return extract_decorator_name(decorator)

    def _extract_base_name(self, base):
        """Extract base class name from AST node."""
        return extract_base_name(base)

    def _get_line(self, content: str, line_no: int) -> str:
        """Get specific line from content."""
        return get_line(content, line_no)

    def _get_function_signature(self, content: str, node: ast.FunctionDef) -> str:
        """Get function signature."""
        return get_function_signature(content, node)

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        return calculate_cyclomatic_complexity(node)

    def _calculate_nesting_depth(self, node) -> int:
        """Calculate maximum nesting depth."""
        return calculate_nesting_depth(node)

    # ------------------------------------------------------------------
    # Top-level analysis pipeline
    # ------------------------------------------------------------------
    def analyze_codebase(self) -> DeepAnalysisResult:
        """Perform comprehensive analysis of the entire codebase."""
        logger.info(f"Starting deep analysis of {len(self.python_files)} Python files")
        self._extract_entities()
        self._analyze_security_issues()
        self._analyze_performance_issues()
        self._analyze_code_complexity()
        self._analyze_code_smells()
        self._analyze_architecture()
        self._analyze_documentation()
        self._analyze_bug_risks()
        metrics = self._calculate_metrics()
        overall_score = self._calculate_overall_score()
        recommendations = self._generate_recommendations()
        logger.info(
            f"✅ Deep analysis complete: {len(self.entities)} entities, {len(self.relationships)} relationships, {len(self.issues)} issues"
        )
        return DeepAnalysisResult(
            overall_score=overall_score,
            issues=self.issues,
            metrics=metrics,
            recommendations=recommendations,
            entities=self.entities,
            relationships=self.relationships,
        )

    async def analyze_graph_multilang(
        self,
    ) -> tuple[list[CoreCodeEntity], list[CodeRelationship | EntityRelation]]:
        """Extract a merged multi-language code graph (entities + relationships).

        Unlike :meth:`analyze_codebase` this performs *only* graph extraction —
        no security / performance / smell sub-analysis — but works for every
        supported language, not just Python. Used to populate the architecture
        graph for non-Python and polyglot repositories.

        Returns:
            A tuple of ``(entities, relationships)`` merged across all languages
            found in :attr:`source_files`. The relationship list is typed as the
            ``CodeRelationship | EntityRelation`` union so it can be assigned
            straight onto :class:`DeepAnalysisResult`.
        """
        entities, relationships = await extract_graph_multilang(
            self.codebase_path, self.source_files
        )
        # Widen to the DeepAnalysisResult union type (Iterable is covariant).
        widened: list[CodeRelationship | EntityRelation] = []
        widened.extend(relationships)
        logger.info(
            f"✅ Multi-language graph analysis complete: {len(entities)} entities, "
            f"{len(widened)} relationships"
        )
        return entities, widened

    # ------------------------------------------------------------------
    # Sub-analyzer dispatchers (delegate to per-domain modules)
    # ------------------------------------------------------------------
    def _extract_entities(self) -> None:
        """Extract code entities (functions, classes, etc.) from Python files."""
        _entities_mod.extract_entities(self)

    def _analyze_security_issues(self) -> None:
        """Analyze for security vulnerabilities."""
        _security.analyze(self)

    def _analyze_performance_issues(self) -> None:
        """Analyze for performance problems."""
        _performance.analyze(self)

    def _analyze_code_complexity(self) -> None:
        """Analyze code complexity."""
        _complexity_checks.analyze(self)

    def _analyze_code_smells(self) -> None:
        """Analyze for code smells."""
        _smells.analyze(self)

    def _analyze_architecture(self) -> None:
        """Analyze architectural issues."""
        _architecture.analyze(self)

    def _analyze_documentation(self) -> None:
        """Analyze documentation quality."""
        _documentation.analyze(self)

    def _analyze_bug_risks(self) -> None:
        """Analyze for potential bug risks."""
        _bug_risks.analyze(self)

    def _calculate_metrics(self):
        """Calculate overall codebase metrics."""
        return _metrics_mod.calculate_metrics(self)

    def _calculate_overall_score(self) -> float:
        """Calculate overall code quality score (0-10)."""
        return _metrics_mod.calculate_overall_score(self)

    def _generate_recommendations(self) -> list[str]:
        """Generate high-level recommendations."""
        return _metrics_mod.generate_recommendations(self)

    # ------------------------------------------------------------------
    # Snippet API
    # ------------------------------------------------------------------
    async def analyze_code_snippet(self, code: str, language: str = "python") -> DeepAnalysisResult:
        """Analyze a code snippet without needing a file path.

        Args:
            code: The code snippet to analyze
            language: The programming language (default: "python")

        Returns:
            DeepAnalysisResult with analysis findings
        """
        import tempfile
        from pathlib import Path

        # Create a temporary file to analyze the snippet
        with tempfile.NamedTemporaryFile(mode="w", suffix=f".{language}", delete=False) as tmp_file:
            tmp_file.write(code)
            tmp_file_path = Path(tmp_file.name)

        try:
            # Temporarily set up for single file analysis
            self.codebase_path = tmp_file_path.parent
            self.python_files = [tmp_file_path]
            self.language = language

            # Clear any existing data
            self.issues = []
            self.entities = []
            self.relationships = []

            # Only analyze if it's Python (for now)
            if language.lower() in SNIPPET_LANGUAGES:
                try:
                    tree = self._safe_parse_ast(code, "snippet.py")
                    if tree:
                        # Extract entities from the snippet
                        self._extract_entities_from_snippet(code, tree, "snippet.py")
                        # Run security analysis on the snippet
                        self._analyze_snippet_security(code, tree)
                        # Run complexity analysis
                        self._analyze_snippet_complexity(code, tree)
                        # Run code smell analysis
                        self._analyze_snippet_smells(code, tree)
                except SyntaxError as e:
                    self.issues.append(
                        CodeIssue(
                            type=IssueType.BUG_RISK,
                            severity=Severity.CRITICAL,
                            title="Syntax Error",
                            description=f"Code has syntax error: {str(e)}",
                            file_path="snippet",
                            line_number=e.lineno,
                            code_snippet=None,
                            suggestion="Fix the syntax error before analysis",
                            confidence=1.0,
                        )
                    )

            # Calculate metrics and score
            metrics = self._calculate_metrics()
            overall_score = self._calculate_overall_score()
            recommendations = self._generate_recommendations()

            return DeepAnalysisResult(
                overall_score=overall_score,
                issues=self.issues,
                metrics=metrics,
                recommendations=recommendations,
                entities=self.entities,
                relationships=self.relationships,
            )

        finally:
            # Clean up temporary file
            if tmp_file_path.exists():
                tmp_file_path.unlink()

    def _extract_entities_from_snippet(
        self, code: str, tree: ast.AST, filename: str = "snippet.py"
    ) -> None:
        """Extract entities from a code snippet."""
        _entities_mod.extract_entities_from_snippet(self, code, tree, filename)

    def _analyze_snippet_security(self, code: str, tree: ast.AST) -> None:
        """Analyze security issues in a code snippet."""
        # Check for common security issues
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    # Check for dangerous functions
                    if func_name in ["eval", "exec"]:
                        self.issues.append(
                            CodeIssue(
                                type=IssueType.SECURITY,
                                severity=Severity.HIGH,
                                title=f"Use of {func_name}()",
                                description=f"Using {func_name}() can execute arbitrary code",
                                file_path="snippet",
                                line_number=node.lineno,
                                code_snippet=self._get_line(code, node.lineno),
                                suggestion=f"Avoid using {func_name}(). Use safer alternatives like ast.literal_eval() for eval.",
                                confidence=0.9,
                            )
                        )

    def _analyze_snippet_complexity(self, code: str, tree: ast.AST) -> None:
        """Analyze complexity in a code snippet."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_cyclomatic_complexity(node)
                if complexity > 10:
                    self.issues.append(
                        CodeIssue(
                            type=IssueType.COMPLEXITY,
                            severity=Severity.MEDIUM,
                            title=f"High cyclomatic complexity in {node.name}",
                            description=f"Function has cyclomatic complexity of {complexity}",
                            file_path="snippet",
                            line_number=node.lineno,
                            code_snippet=self._get_function_signature(code, node),
                            suggestion="Consider breaking this function into smaller functions",
                            confidence=1.0,
                        )
                    )

    def _analyze_snippet_smells(self, code: str, tree: ast.AST) -> None:
        """Analyze code smells in a snippet."""
        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno if node.end_lineno else 0
                if func_lines > 50:
                    self.issues.append(
                        CodeIssue(
                            type=IssueType.CODE_SMELL,
                            severity=Severity.LOW,
                            title=f"Long function: {node.name}",
                            description=f"Function is {func_lines} lines long",
                            file_path="snippet",
                            line_number=node.lineno,
                            code_snippet=None,
                            suggestion="Consider splitting into smaller functions",
                            confidence=0.8,
                        )
                    )
