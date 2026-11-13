"""Security issue analysis for the deep code analyzer."""

from __future__ import annotations

import ast
import logging
import re

from .base import (
    CodeIssue,
    IssueCollector,
    IssueType,
    Severity,
    get_line,
    safe_parse_ast,
)

logger = logging.getLogger(__name__)


def analyze(analyzer: IssueCollector) -> None:
    """Analyze for security vulnerabilities."""
    logger.debug("Analyzing security issues")
    for file_path in analyzer.python_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = safe_parse_ast(content, str(file_path))
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ["pickle", "eval", "exec", "subprocess"]:
                            analyzer.issues.append(
                                CodeIssue(
                                    type=IssueType.SECURITY,
                                    severity=Severity.HIGH,
                                    title=f"Potentially dangerous import: {alias.name}",
                                    description=f"The '{alias.name}' module can be dangerous if used with untrusted input",
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    code_snippet=get_line(content, node.lineno),
                                    suggestion=f"Consider safer alternatives to {alias.name} or add input validation",
                                    confidence=0.8,
                                )
                            )
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in [
                        "eval",
                        "exec",
                    ]:
                        analyzer.issues.append(
                            CodeIssue(
                                type=IssueType.SECURITY,
                                severity=Severity.CRITICAL,
                                title=f"Use of {node.func.id}()",
                                description=f"Using {node.func.id}() with untrusted input is a security risk",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=get_line(content, node.lineno),
                                suggestion="Avoid eval/exec or validate input thoroughly",
                                confidence=0.9,
                            )
                        )
            secret_patterns = [
                ("password\\s*=\\s*[\"\\'][^\"\\']+[\"\\']", "Hardcoded password"),
                ("api_key\\s*=\\s*[\"\\'][^\"\\']+[\"\\']", "Hardcoded API key"),
                ("secret\\s*=\\s*[\"\\'][^\"\\']+[\"\\']", "Hardcoded secret"),
                ("token\\s*=\\s*[\"\\'][^\"\\']+[\"\\']", "Hardcoded token"),
            ]
            for pattern, description in secret_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_no = content[: match.start()].count("\n") + 1
                    analyzer.issues.append(
                        CodeIssue(
                            type=IssueType.SECURITY,
                            severity=Severity.HIGH,
                            title=description,
                            description="Hardcoded credentials should not be stored in source code",
                            file_path=str(file_path),
                            line_number=line_no,
                            code_snippet=get_line(content, line_no),
                            suggestion="Use environment variables or secure credential storage",
                            confidence=0.7,
                        )
                    )
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Error analyzing security in {file_path}: {e}")
