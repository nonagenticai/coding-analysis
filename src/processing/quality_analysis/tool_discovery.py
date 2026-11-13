"""
Tool discovery and registration for quality analysis tools.
Manages the discovery and registration of tools for different programming languages.
"""

from __future__ import annotations

import logging

from .tool_data import (
    get_cpp_tools,
    get_csharp_tools,
    get_go_tools,
    get_java_tools,
    get_rust_tools,
)
from .tool_definitions import QualityTool, ToolCategory

logger = logging.getLogger(__name__)


class ToolDiscovery:
    """Handles discovery and registration of quality tools by language."""

    def __init__(self):
        self.tools = self._initialize_tools()

    def get_tools_for_language(self, language: str) -> dict[str, QualityTool]:
        """Get quality tools for specified language.

        Args:
            language (str): The language parameter.

        Returns:
            Dict[str, QualityTool]: The dict[str, qualitytool] result."""
        return self.tools.get(language.lower(), {})

    def get_tool(self, language: str, tool_name: str) -> QualityTool | None:
        """Get specific tool for language.

        Args:
            language (str): The language parameter.
            tool_name (str): The tool_name parameter.

        Returns:
            Optional[QualityTool]: The optional[qualitytool] result."""
        return self.tools.get(language.lower(), {}).get(tool_name)

    def get_tools_by_category(
        self, language: str, category: ToolCategory
    ) -> dict[str, QualityTool]:
        """Get tools by category for a language.

        Args:
            language (str): The language parameter.
            category (ToolCategory): The category parameter.

        Returns:
            Dict[str, QualityTool]: The dict[str, qualitytool] result."""
        language_tools = self.get_tools_for_language(language)
        return {name: tool for name, tool in language_tools.items() if tool.category == category}

    def list_supported_languages(self) -> list[str]:
        """Get list of supported languages.

        Returns:
            List[str]: The list[str] result."""
        return list(self.tools.keys())

    def list_available_tools(self, language: str) -> list[str]:
        """Get list of available tool names for a language.

        Args:
            language (str): The language parameter.

        Returns:
            List[str]: The list[str] result."""
        return list(self.get_tools_for_language(language).keys())

    def _initialize_tools(self) -> dict[str, dict[str, QualityTool]]:
        """Initialize the tool registry with all supported tools."""
        return {
            "python": self._init_python_tools(),
            "javascript": self._init_javascript_tools(),
            "typescript": self._init_typescript_tools(),
            "java": self._init_java_tools(),
            "rust": self._init_rust_tools(),
            "go": self._init_go_tools(),
            "csharp": self._init_csharp_tools(),
            "cpp": self._init_cpp_tools(),
        }

    def _init_python_tools(self) -> dict[str, QualityTool]:
        """Initialize Python quality tools."""
        return {
            "ruff": QualityTool(
                name="ruff",
                category=ToolCategory.LINTER,
                # flake8 (E/W/F) + isort (I) + pylint (PL*) rules in one pass.
                command=[
                    "ruff",
                    "check",
                    "--output-format",
                    "json",
                    "--extend-select",
                    "I,PL",
                ],
                output_format="json",
                description="Python linter (flake8, isort and pylint rules)",
                installation_cmd=["uv", "pip", "install", "--system", "ruff==0.17.3"],
                timeout=45,
                config_files=["ruff.toml", ".ruff.toml", "pyproject.toml"],
            ),
            "ruff-format": QualityTool(
                name="ruff-format",
                category=ToolCategory.FORMATTER,
                command=["ruff", "format", "--check", "--output-format", "json"],
                output_format="json",
                description="Python code formatter (check mode)",
                installation_cmd=["uv", "pip", "install", "--system", "ruff==0.17.3"],
                timeout=30,
                config_files=["ruff.toml", ".ruff.toml", "pyproject.toml"],
            ),
            "ty": QualityTool(
                name="ty",
                category=ToolCategory.TYPE_CHECKER,
                command=["ty", "check", "--output-format", "gitlab"],
                output_format="json",
                description="Static type checker for Python",
                installation_cmd=["uv", "pip", "install", "--system", "ty==0.0.92"],
                timeout=45,
                config_files=["ty.toml", "pyproject.toml"],
            ),
            "bandit": QualityTool(
                name="bandit",
                category=ToolCategory.SECURITY,
                command=["bandit", "-", "json", "-r"],
                output_format="json",
                description="Security scanner for Python",
                installation_cmd=["uv", "pip", "install", "--system", "bandit==1.7.6"],
                timeout=30,
                config_files=[".bandit", "bandit.yaml"],
            ),
            "vulture": QualityTool(
                name="vulture",
                category=ToolCategory.LINTER,
                command=["vulture", "--min-confidence", "80"],
                output_format="text",
                description="Dead code finder for Python",
                installation_cmd=["uv", "pip", "install", "--system", "vulture==2.11"],
                timeout=30,
            ),
            "radon": QualityTool(
                name="radon",
                category=ToolCategory.COMPLEXITY,
                command=["radon", "cc", "--json"],
                output_format="json",
                description="Code complexity analyzer",
                installation_cmd=["uv", "pip", "install", "--system", "radon==6.0.1"],
                timeout=30,
            ),
            "pydocstyle": QualityTool(
                name="pydocstyle",
                category=ToolCategory.DOCUMENTATION,
                command=["pydocstyle"],
                output_format="text",
                description="Docstring style checker",
                installation_cmd=["uv", "pip", "install", "--system", "pydocstyle==6.3.0"],
                timeout=30,
                config_files=[".pydocstyle", "setup.cfg", "tox.ini"],
            ),
            # New AI Agent Static Analysis Tools
            "pyan3": QualityTool(
                name="pyan3",
                category=ToolCategory.CALL_GRAPH,
                command=[
                    "pyan3",
                    "--uses",
                    "--defines",
                    "--colored",
                    "--grouped",
                    "--annotated",
                ],
                output_format="dot",
                description="Python call graph generator",
                installation_cmd=["uv", "pip", "install", "--system", "pyan3==1.2.0"],
                timeout=60,
            ),
            "code2flow": QualityTool(
                name="code2flow",
                category=ToolCategory.CALL_GRAPH,
                command=["code2flow"],
                output_format="dot",
                description="Multi-language call graph generator",
                installation_cmd=["uv", "pip", "install", "--system", "code2flow==2.5.1"],
                timeout=90,
            ),
            "pycg": QualityTool(
                name="pycg",
                category=ToolCategory.CALL_GRAPH,
                command=["pycg", "--format", "json"],
                output_format="json",
                description="Precise Python call graph generator",
                installation_cmd=["uv", "pip", "install", "--system", "pycg"],
                timeout=120,
            ),
            "crosshair": QualityTool(
                name="crosshair",
                category=ToolCategory.SYMBOLIC_ANALYSIS,
                command=["crosshair", "check", "--report_all"],
                output_format="text",
                description="Symbolic execution for Python",
                installation_cmd=["uv", "pip", "install", "--system", "crosshair-tool==0.0.65"],
                timeout=180,
            ),
            "pysa": QualityTool(
                name="pysa",
                category=ToolCategory.SYMBOLIC_ANALYSIS,
                command=["pyre", "analyze", "--output-format", "json"],
                output_format="json",
                description="Taint analysis and security scanning",
                installation_cmd=["uv", "pip", "install", "--system", "pyre-check"],
                timeout=300,
                config_files=[".pyre_configuration"],
            ),
            "sphinx": QualityTool(
                name="sphinx",
                category=ToolCategory.ARCHITECTURE,
                command=["sphinx-build", "-b", "html"],
                output_format="html",
                description="Documentation generator with architecture diagrams",
                installation_cmd=[
                    "uv",
                    "pip",
                    "install",
                    "--system",
                    "sphinx==7.2.6",
                    "sphinx-rtd-theme",
                ],
                timeout=120,
                config_files=["conf.py"],
            ),
        }

    def _init_javascript_tools(self) -> dict[str, QualityTool]:
        """Initialize JavaScript quality tools."""
        return {
            "eslint": QualityTool(
                name="eslint",
                category=ToolCategory.LINTER,
                command=["eslint", "--format=json"],
                output_format="json",
                description="JavaScript and JSX linter",
                installation_cmd=["npm", "install", "-g", "eslint@8.47.0"],
                timeout=45,
                config_files=[
                    ".eslintrc.js",
                    ".eslintrc.json",
                    ".eslintrc.yml",
                ],
            ),
            "prettier": QualityTool(
                name="prettier",
                category=ToolCategory.FORMATTER,
                command=["prettier", "--check"],
                output_format="text",
                description="Code formatter for JavaScript",
                installation_cmd=["npm", "install", "-g", "prettier@3.0.1"],
                timeout=30,
                config_files=[
                    ".prettierrc",
                    ".prettierrc.json",
                    "prettier.config.js",
                ],
            ),
            "jshint": QualityTool(
                name="jshint",
                category=ToolCategory.LINTER,
                command=["jshint", "--reporter=json"],
                output_format="json",
                description="JavaScript static analysis tool",
                installation_cmd=["npm", "install", "-g", "jshint"],
                timeout=30,
                config_files=[".jshintrc"],
            ),
        }

    def _init_typescript_tools(self) -> dict[str, QualityTool]:
        """Initialize TypeScript quality tools."""
        return {
            "eslint": QualityTool(
                name="eslint",
                category=ToolCategory.LINTER,
                command=["eslint", "--format=json", "--ext", ".ts,.tsx"],
                output_format="json",
                description="TypeScript linter with ESLint",
                installation_cmd=[
                    "npm",
                    "install",
                    "-g",
                    "eslint@8.47.0",
                    "@typescript-eslint/eslint-plugin@6.4.0",
                    "@typescript-eslint/parser@6.4.0",
                ],
                timeout=45,
                config_files=[".eslintrc.js", ".eslintrc.json"],
            ),
            "prettier": QualityTool(
                name="prettier",
                category=ToolCategory.FORMATTER,
                command=["prettier", "--check", "--parser=typescript"],
                output_format="text",
                description="Code formatter for TypeScript",
                installation_cmd=["npm", "install", "-g", "prettier@3.0.1"],
                timeout=30,
                config_files=[".prettierrc", "prettier.config.js"],
            ),
            "tsc": QualityTool(
                name="tsc",
                category=ToolCategory.TYPE_CHECKER,
                command=["tsc", "--noEmit", "--pretty"],
                output_format="text",
                description="TypeScript compiler type checker",
                installation_cmd=["npm", "install", "-g", "typescript@5.1.6"],
                timeout=60,
                config_files=["tsconfig.json"],
            ),
            "tslint": QualityTool(
                name="tslint",
                category=ToolCategory.LINTER,
                command=["tslint", "--format=json"],
                output_format="json",
                description="TypeScript linter (deprecated, use ESLint)",
                installation_cmd=["npm", "install", "-g", "tslint"],
                timeout=45,
                config_files=["tslint.json"],
            ),
        }

    def _init_java_tools(self) -> dict[str, QualityTool]:
        """Initialize Java quality tools."""
        return get_java_tools()

    def _init_rust_tools(self) -> dict[str, QualityTool]:
        """Initialize Rust quality tools."""
        return get_rust_tools()

    def _init_go_tools(self) -> dict[str, QualityTool]:
        """Initialize Go quality tools."""
        return get_go_tools()

    def _init_csharp_tools(self) -> dict[str, QualityTool]:
        """Initialize C# quality tools."""
        return get_csharp_tools()

    def _init_cpp_tools(self) -> dict[str, QualityTool]:
        """Initialize C++ quality tools."""
        return get_cpp_tools()
