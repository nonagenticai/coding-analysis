"""
Documentation Extraction and Parsing Module
Handles extraction of documentation from various file types and formats.
"""

from __future__ import annotations

import ast
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from ..ast_analysis.function_extractor import FunctionExtractor

logger = logging.getLogger(__name__)


class DocumentationExtractor:
    """Handles extraction of documentation from various file types."""

    def __init__(self):
        """Initialize the documentation extractor."""
        self.function_extractor = FunctionExtractor()

    async def extract_file_documentation(self, file_path: str) -> dict[str, Any]:
        """Extract documentation from a single file."""
        try:
            file_ext = Path(file_path).suffix

            if file_ext == ".py":
                return await self._extract_python_documentation(file_path)
            if file_ext in [".js", ".ts"]:
                return await self._extract_javascript_documentation(file_path)
            if file_ext in [".md", ".rst"]:
                return await self._extract_markup_documentation(file_path)
            return {"content": "Unsupported file type", "coverage": 0.0}

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error extracting documentation from %s: %s", file_path, e)
            return {"error": str(e), "file_path": file_path}

    async def extract_snippet_documentation(
        self, code_snippet: str, file_extension: str = ".py"
    ) -> dict[str, Any]:
        """Extract documentation from a code snippet."""
        try:
            # Create temporary file for snippet analysis
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=file_extension, delete=False
            ) as tmp_file:
                tmp_file.write(code_snippet)
                tmp_file_path = tmp_file.name

            try:
                doc_data = await self.extract_file_documentation(tmp_file_path)
                doc_data["snippet"] = True
                return doc_data
            finally:
                import os

                os.unlink(tmp_file_path)

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error extracting documentation from snippet: %s", e)
            return {"error": str(e), "snippet": True}

    async def _extract_python_documentation(self, file_path: str) -> dict[str, Any]:
        """Extract documentation from Python file."""
        try:
            analysis = self.function_extractor.extract_comprehensive_signatures(file_path)

            # Extract module-level docstring
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            try:
                tree = ast.parse(content)
                module_docstring = ast.get_docstring(tree)
            except (SyntaxError, ValueError, OSError):  # fmt: skip
                module_docstring = None

            # Extract class information
            classes = []
            try:
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_info = {
                            "name": node.name,
                            "docstring": ast.get_docstring(node),
                            "methods": [],
                            "line_number": node.lineno,
                        }

                        # Extract method information
                        for class_node in node.body:
                            if isinstance(class_node, ast.FunctionDef):
                                method_info = {
                                    "name": class_node.name,
                                    "docstring": ast.get_docstring(class_node),
                                    "line_number": class_node.lineno,
                                    "is_private": class_node.name.startswith("_"),
                                }
                                class_info["methods"].append(method_info)

                        classes.append(class_info)
            except (SyntaxError, ValueError, OSError):  # fmt: skip
                classes = []

            # Extract import information
            imports = []
            try:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(
                                {
                                    "type": "import",
                                    "name": alias.name,
                                    "alias": alias.asname,
                                    "line_number": node.lineno,
                                }
                            )
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            imports.append(
                                {
                                    "type": "from_import",
                                    "module": module,
                                    "name": alias.name,
                                    "alias": alias.asname,
                                    "line_number": node.lineno,
                                }
                            )
            except (SyntaxError, ValueError, OSError):  # fmt: skip
                imports = []

            return {
                "file_type": "python",
                "module_docstring": module_docstring,
                "functions": analysis.get("functions", []),
                "classes": classes,
                "imports": imports,
                "type_coverage": analysis.get("type_coverage", 0.0),
                "external_dependencies": analysis.get("external_dependencies", []),
                "coverage": self._calculate_python_coverage(
                    module_docstring, analysis.get("functions", []), classes
                ),
            }

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error extracting Python documentation: %s", e)
            return {"error": str(e)}

    async def _extract_javascript_documentation(self, file_path: str) -> dict[str, Any]:
        """Extract documentation from JavaScript/TypeScript file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Extract JSDoc comments
            jsdoc_pattern = r"/\*\*\s*(.*?)\s*\*/"
            jsdoc_comments = re.findall(jsdoc_pattern, content, re.DOTALL)

            # Extract function declarations
            function_patterns = [
                r"function\s+(\w+)\s*\([^)]*\)",  # function declarations
                r"(\w+)\s*:\s*function\s*\([^)]*\)",  # object method declarations
                r"(\w+)\s*=\s*\([^)]*\)\s*=>",  # arrow function assignments
                r"(\w+)\s*\([^)]*\)\s*{",  # method shorthand
            ]

            functions = []
            for pattern in function_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    function_name = match.group(1)
                    line_number = content[: match.start()].count("\n") + 1
                    functions.append(
                        {
                            "name": function_name,
                            "line_number": line_number,
                            "signature": match.group(0),
                        }
                    )

            # Extract class declarations (ES6+)
            class_pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{"
            classes = []
            for match in re.finditer(class_pattern, content):
                class_name = match.group(1)
                parent_class = match.group(2)
                line_number = content[: match.start()].count("\n") + 1
                classes.append(
                    {
                        "name": class_name,
                        "parent": parent_class,
                        "line_number": line_number,
                    }
                )

            # Extract import/require statements
            import_patterns = [
                r"import\s+.*?from\s+['\"]([^'\"]+)['\"]",  # ES6 imports
                r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",  # CommonJS requires
            ]

            imports = []
            for pattern in import_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    module_name = match.group(1)
                    line_number = content[: match.start()].count("\n") + 1
                    imports.append(
                        {
                            "module": module_name,
                            "line_number": line_number,
                            "type": "es6" if "import" in match.group(0) else "commonjs",
                        }
                    )

            return {
                "file_type": "javascript",
                "jsdoc_comments": jsdoc_comments,
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "file_content_preview": content[:1000],
                "coverage": self._calculate_javascript_coverage(jsdoc_comments, functions),
            }

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error extracting JavaScript documentation: %s", e)
            return {"error": str(e)}

    async def _extract_markup_documentation(self, file_path: str) -> dict[str, Any]:
        """Extract content from markup files (Markdown, RST)."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            file_ext = Path(file_path).suffix

            if file_ext == ".md":
                return await self._parse_markdown_content(content)
            elif file_ext == ".rst":
                return await self._parse_rst_content(content)
            else:
                return {
                    "file_type": "markup",
                    "content": content,
                    "coverage": 1.0,  # Markup files are considered fully documented
                }

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error extracting markup documentation: %s", e)
            return {"error": str(e)}

    async def _parse_markdown_content(self, content: str) -> dict[str, Any]:
        """Parse markdown content and extract structure."""
        try:
            # Extract headers
            headers = []
            header_pattern = r"^(#{1,6})\s+(.+)$"
            for match in re.finditer(header_pattern, content, re.MULTILINE):
                level = len(match.group(1))
                title = match.group(2).strip()
                line_number = content[: match.start()].count("\n") + 1
                headers.append(
                    {
                        "level": level,
                        "title": title,
                        "line_number": line_number,
                    }
                )

            # Extract code blocks
            code_blocks = []
            code_pattern = r"```(\w+)?\n(.*?)```"
            for match in re.finditer(code_pattern, content, re.DOTALL):
                language = match.group(1) or "text"
                code = match.group(2).strip()
                line_number = content[: match.start()].count("\n") + 1
                code_blocks.append(
                    {
                        "language": language,
                        "code": code,
                        "line_number": line_number,
                    }
                )

            # Extract links
            links = []
            link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
            for match in re.finditer(link_pattern, content):
                text = match.group(1)
                url = match.group(2)
                line_number = content[: match.start()].count("\n") + 1
                links.append(
                    {
                        "text": text,
                        "url": url,
                        "line_number": line_number,
                    }
                )

            return {
                "file_type": "markdown",
                "content": content,
                "headers": headers,
                "code_blocks": code_blocks,
                "links": links,
                "word_count": len(content.split()),
                "coverage": 1.0,
            }

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error parsing markdown content: %s", e)
            return {"error": str(e)}

    async def _parse_rst_content(self, content: str) -> dict[str, Any]:
        """Parse reStructuredText content and extract structure."""
        try:
            # Extract headers (RST uses underlines)
            headers = []
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line and all(c in "=-~^'\"#*+<>" for c in next_line):
                        if len(next_line) >= len(line.strip()) and line.strip():
                            headers.append(
                                {
                                    "title": line.strip(),
                                    "underline_char": next_line[0],
                                    "line_number": i + 1,
                                }
                            )

            # Extract directives
            directives = []
            directive_pattern = r"^\.\. (\w+)::\s*(.*)"
            for match in re.finditer(directive_pattern, content, re.MULTILINE):
                directive_type = match.group(1)
                directive_content = match.group(2)
                line_number = content[: match.start()].count("\n") + 1
                directives.append(
                    {
                        "type": directive_type,
                        "content": directive_content,
                        "line_number": line_number,
                    }
                )

            return {
                "file_type": "rst",
                "content": content,
                "headers": headers,
                "directives": directives,
                "word_count": len(content.split()),
                "coverage": 1.0,
            }

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error parsing RST content: %s", e)
            return {"error": str(e)}

    def _calculate_python_coverage(
        self,
        module_docstring: str | None,
        functions: list[dict],
        classes: list[dict],
    ) -> float:
        """Calculate documentation coverage for Python files."""
        try:
            total_items = 1  # Module itself
            documented_items = 1 if module_docstring else 0

            # Count functions
            total_items += len(functions)
            documented_items += sum(1 for func in functions if func.get("docstring"))

            # Count classes and their methods
            for cls in classes:
                total_items += 1  # Class itself
                documented_items += 1 if cls.get("docstring") else 0

                # Count methods
                methods = cls.get("methods", [])
                total_items += len(methods)
                documented_items += sum(1 for method in methods if method.get("docstring"))

            return documented_items / total_items if total_items > 0 else 0.0

        except (ValueError, TypeError, ZeroDivisionError):  # fmt: skip
            return 0.0

    def _calculate_javascript_coverage(
        self, jsdoc_comments: list[str], functions: list[dict]
    ) -> float:
        """Calculate documentation coverage for JavaScript files."""
        try:
            if not functions:
                return 1.0 if jsdoc_comments else 0.0

            # Estimate coverage based on JSDoc comments vs functions
            # This is a rough estimate since we can't easily match comments to functions
            doc_ratio = min(len(jsdoc_comments) / len(functions), 1.0)
            return doc_ratio

        except (ValueError, TypeError, ZeroDivisionError):  # fmt: skip
            return 0.0
