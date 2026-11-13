"""Module for architecture pattern analysis and language-specific tool configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from processing.architecture.rust_gate import (
    CARGO_AUDIT_COMMAND,
    CLIPPY_COMMAND,
    RUSTFMT_COMMAND,
)
from processing.architecture.tool_execution._base import (
    FILTER_FINDINGS,
    SCOPE_MODES,
    TOOL_DISPATCH,
    UNSCOPABLE,
)

# Source extensions per language: what language detection and file counting
# recognise, and what the API advertises per language.
LANGUAGE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "python": (".py",),
    "javascript": (".js", ".jsx"),
    "typescript": (".ts", ".tsx"),
    "java": (".java",),
    "go": (".go",),
    "rust": (".rs",),
    "cpp": (".cpp", ".cc", ".cxx", ".hpp", ".h"),
    "csharp": (".cs",),
}


class ArchitecturePatternAnalysis:
    """Handler for architecture pattern detection and language-specific analysis."""

    def __init__(self) -> None:
        """Initialize the architecture pattern analyzer."""
        # Language-specific tool configurations
        self.tools_config = {
            "python": {
                "ruff": {
                    "command": [
                        "ruff",
                        "check",
                        "--output-format=concise",
                        "--extend-select",
                        "I,PL",
                    ],
                    "enabled": True,
                    "category": "linter",
                    "description": "Fast Python linter (flake8, isort and pylint rules)",
                },
                "ty": {
                    "command": ["ty", "check", "--output-format", "gitlab"],
                    "enabled": True,
                    "category": "type_checker",
                    "description": "Static type checker",
                },
                "bandit": {
                    "command": ["bandit", "-r", "-f", "json"],
                    "enabled": True,
                    "category": "security",
                    "description": "Security vulnerability scanner",
                },
                "radon": {
                    "command": ["radon", "cc", "-j"],
                    "enabled": True,
                    "category": "complexity",
                    "description": "Cyclomatic complexity analyzer",
                },
                "ruff-format": {
                    "command": ["ruff", "format", "--check", "--output-format=concise"],
                    "enabled": True,
                    "category": "formatter",
                    "description": "Code formatter compliance check",
                },
            },
            "javascript": {
                "eslint": {
                    "command": ["eslint", "--format=json"],
                    "enabled": True,
                    "category": "linter",
                    "description": "JavaScript/TypeScript linter",
                },
                "jshint": {
                    "command": ["jshint", "--reporter=json"],
                    "enabled": True,
                    "category": "linter",
                    "description": "JavaScript static analysis",
                },
                "prettier": {
                    "command": ["prettier", "--check"],
                    "enabled": True,
                    "category": "formatter",
                    "description": "Code formatter compliance check",
                },
                "semgrep": {
                    "command": ["semgrep", "--json", "--config=auto"],
                    "enabled": True,
                    "category": "security",
                    "description": "Static analysis for bugs and security",
                },
            },
            "typescript": {
                "tsc": {
                    "command": ["tsc", "--noEmit", "--pretty", "false"],
                    "enabled": True,
                    "category": "type_checker",
                    "description": "TypeScript compiler type checking",
                },
                "eslint": {
                    "command": [
                        "eslint",
                        "--format=json",
                        "--ext",
                        ".ts,.tsx",
                    ],
                    "enabled": True,
                    "category": "linter",
                    "description": "TypeScript linter",
                },
                "prettier": {
                    "command": ["prettier", "--check"],
                    "enabled": True,
                    "category": "formatter",
                    "description": "Code formatter compliance check",
                },
            },
            "java": {
                "checkstyle": {
                    "command": ["checkstyle", "-", "xml"],
                    "enabled": True,
                    "category": "linter",
                    "description": "Java coding standard checker",
                },
                "pmd": {
                    "command": ["pmd", "check", "-", "json"],
                    "enabled": True,
                    "category": "linter",
                    "description": "Java static analysis",
                },
                "spotbugs": {
                    "command": ["spotbugs", "-textui", "-xml"],
                    "enabled": True,
                    "category": "security",
                    "description": "Java bug pattern detector",
                },
            },
            "go": {
                "golint": {
                    "command": ["golint", "-set_exit_status"],
                    "enabled": True,
                    "category": "linter",
                    "description": "Go linter",
                },
                "go-vet": {
                    "command": ["go", "vet"],
                    "enabled": True,
                    "category": "linter",
                    "description": "Go static analysis",
                },
                "golangci-lint": {
                    "command": ["golangci-lint", "run", "--out-format=json"],
                    "enabled": True,
                    "category": "linter",
                    "description": "Comprehensive Go linter",
                },
                "gosec": {
                    "command": ["gosec", "-fmt=json"],
                    "enabled": True,
                    "category": "security",
                    "description": "Go security scanner",
                },
            },
            "rust": {
                "clippy": {
                    "command": list(CLIPPY_COMMAND),
                    "enabled": True,
                    "category": "linter",
                    "description": "Rust linter",
                },
                "rustfmt": {
                    "command": list(RUSTFMT_COMMAND),
                    "enabled": True,
                    "category": "formatter",
                    "description": "Rust formatter compliance check",
                },
                "cargo-audit": {
                    "command": list(CARGO_AUDIT_COMMAND),
                    "enabled": True,
                    "category": "security",
                    "description": "Rust security vulnerability scanner",
                },
            },
            "cpp": {
                "cppcheck": {
                    "command": ["cppcheck", "--enable=all", "--xml"],
                    "enabled": True,
                    "category": "linter",
                    "description": "C++ static analysis",
                },
                "clang-tidy": {
                    "command": ["clang-tidy", "--format-style=json"],
                    "enabled": True,
                    "category": "linter",
                    "description": "C++ linter and static analyzer",
                },
                "cpplint": {
                    "command": ["cpplint"],
                    "enabled": True,
                    "category": "linter",
                    "description": "C++ style checker",
                },
            },
            "csharp": {
                "dotnet-format": {
                    "command": [
                        "dotnet",
                        "format",
                        "--verify-no-changes",
                        "--verbosity",
                        "diagnostic",
                    ],
                    "enabled": True,
                    "category": "formatter",
                    "description": "C# formatter compliance check",
                },
                "roslynator": {
                    "command": ["roslynator", "analyze"],
                    "enabled": True,
                    "category": "linter",
                    "description": "C# static analyzer",
                },
            },
        }

        # Security-focused tool configurations
        self.security_tools = {
            "semgrep": {
                "languages": [
                    "python",
                    "javascript",
                    "typescript",
                    "java",
                    "go",
                    "rust",
                    "cpp",
                    "csharp",
                ],
                "command": ["semgrep", "--json", "--config=auto"],
                "description": "Multi-language security scanner",
            },
            "sonarqube": {
                "languages": [
                    "python",
                    "javascript",
                    "typescript",
                    "java",
                    "go",
                    "rust",
                    "cpp",
                    "csharp",
                ],
                "command": ["sonar-scanner", "-Dsonar.projectKey=temp"],
                "description": "Comprehensive code quality platform",
            },
        }

        # Language-specific severity mappings
        self.severity_mappings = {
            "python": {
                "bandit": {
                    "HIGH": "critical",
                    "MEDIUM": "high",
                    "LOW": "medium",
                },
                "ruff": {
                    "E": "high",
                    "W": "medium",
                    "F": "high",
                    "C": "low",
                    "N": "low",
                },
                "ty": {
                    "blocker": "critical",
                    "critical": "critical",
                    "major": "high",
                    "minor": "medium",
                    "info": "low",
                },
            },
            "javascript": {
                "eslint": {"2": "high", "1": "medium", "0": "low"},
                "semgrep": {
                    "ERROR": "critical",
                    "WARNING": "high",
                    "INFO": "medium",
                },
            },
            "go": {
                "gosec": {
                    "HIGH": "critical",
                    "MEDIUM": "high",
                    "LOW": "medium",
                },
                "golangci-lint": {
                    "error": "high",
                    "warning": "medium",
                    "info": "low",
                },
            },
            "rust": {
                "clippy": {"deny": "critical", "warn": "high", "allow": "low"},
                "cargo-audit": {
                    "critical": "critical",
                    "high": "high",
                    "medium": "medium",
                    "low": "low",
                },
            },
        }

    def detect_primary_language(self, repo_path: Path) -> str:
        """Detect the primary programming language in the repository."""
        file_counts: dict[str, int] = {}

        for lang, extensions in LANGUAGE_EXTENSIONS.items():
            count = 0
            for ext in extensions:
                count += len(list(repo_path.rglob(f"*{ext}")))
            if count > 0:
                file_counts[lang] = count

        if not file_counts:
            return "python"  # default

        # Return language with most files
        return max(file_counts.items(), key=lambda x: x[1])[0]

    def get_available_tools(self, language: str) -> list[str]:
        """Get list of available tools for a specific language."""
        if language in self.tools_config:
            return list(self.tools_config[language].keys())
        return []

    def get_tool_config(self, language: str, tool_name: str) -> dict[str, Any]:
        """Get configuration for a specific tool."""
        if language in self.tools_config and tool_name in self.tools_config[language]:
            return self.tools_config[language][tool_name]
        return {}

    def is_tool_enabled(self, language: str, tool_name: str) -> bool:
        """Check if a tool is enabled for a language."""
        config = self.get_tool_config(language, tool_name)
        return config.get("enabled", False)

    def get_security_tools(self, language: str) -> list[str]:
        """Get security tools available for a language."""
        security_tools = []
        for tool_name, tool_config in self.security_tools.items():
            if language in tool_config["languages"]:
                security_tools.append(tool_name)
        return security_tools

    def count_files_by_language(self, repo_path: Path, language: str) -> int:
        """Count files for the given language."""
        count = 0
        for ext in LANGUAGE_EXTENSIONS.get(language, ()):
            count += len(list(repo_path.rglob(f"*{ext}")))

        return count

    def detect_architecture_patterns(self, repo_path: Path) -> dict[str, Any]:
        """Detect common architecture patterns in the repository."""
        patterns = {
            "mvc": False,
            "microservices": False,
            "layered": False,
            "clean_architecture": False,
            "hexagonal": False,
            "event_driven": False,
            "domain_driven": False,
        }

        # Check for MVC pattern
        mvc_indicators = ["models", "views", "controllers", "templates"]
        mvc_score = sum(1 for indicator in mvc_indicators if any(repo_path.rglob(f"*{indicator}*")))
        patterns["mvc"] = mvc_score >= 2

        # Check for microservices
        microservice_indicators = [
            "services",
            "docker-compose",
            "kubernetes",
            "api-gateway",
        ]
        microservice_score = sum(
            1 for indicator in microservice_indicators if any(repo_path.rglob(f"*{indicator}*"))
        )
        patterns["microservices"] = microservice_score >= 2

        # Check for layered architecture
        layer_indicators = ["presentation", "business", "data", "infrastructure"]
        layer_score = sum(
            1 for indicator in layer_indicators if any(repo_path.rglob(f"*{indicator}*"))
        )
        patterns["layered"] = layer_score >= 2

        # Check for clean architecture
        clean_indicators = ["entities", "use_cases", "adapters", "frameworks"]
        clean_score = sum(
            1 for indicator in clean_indicators if any(repo_path.rglob(f"*{indicator}*"))
        )
        patterns["clean_architecture"] = clean_score >= 2

        # Check for event-driven
        event_indicators = ["events", "handlers", "subscribers", "publishers"]
        event_score = sum(
            1 for indicator in event_indicators if any(repo_path.rglob(f"*{indicator}*"))
        )
        patterns["event_driven"] = event_score >= 2

        return patterns

    def analyze_dependency_structure(self, repo_path: Path, language: str) -> dict[str, Any]:
        """Analyze dependency structure patterns."""
        analysis: dict[str, bool | int] = {
            "circular_dependencies": False,
            "deep_nesting": False,
            "god_objects": False,
            "tight_coupling": False,
        }

        # Language-specific dependency analysis
        if language == "python":
            # Check for common Python dependency issues
            init_files = list(repo_path.rglob("__init__.py"))
            analysis["module_count"] = len(init_files)

            # Simple heuristic for deep nesting
            max_depth = max((len(f.parts) for f in init_files), default=0)
            analysis["max_depth"] = max_depth
            analysis["deep_nesting"] = max_depth > 6

        elif language in ["javascript", "typescript"]:
            # Check for JS/TS specific patterns
            package_files = list(repo_path.rglob("package.json"))
            analysis["package_count"] = len(package_files)

            # Check for node_modules depth (indicates dependency complexity)
            node_modules = list(repo_path.rglob("node_modules"))
            analysis["has_node_modules"] = len(node_modules) > 0

        return analysis


#: Accepted spellings of a quality language in ``/api/quality/{language}/tools``.
LANGUAGE_ALIASES: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "rs": "rust",
    "c++": "cpp",
    "cs": "csharp",
    "c#": "csharp",
}

#: How a tool's scope mode behaves on a path-scoped (``paths``) request.
_PATH_SCOPING = {
    UNSCOPABLE: "skipped",
    FILTER_FINDINGS: "whole_project_findings_filtered",
}


def enabled_tool_details_by_language() -> dict[str, list[dict[str, str]]]:
    """``{name, category, description, path_scoping}`` of every runnable tool per language.

    A tool is listed iff it is enabled in the ``tools_config`` the analyzer
    iterates AND has a runner in ``TOOL_DISPATCH``, so what the API advertises
    (``/languages``, ``/info``, ``/{language}/tools``) and accepts is exactly
    what it runs. ``path_scoping``: ``paths`` (pointed at the scoped paths),
    ``whole_project_findings_filtered`` or ``skipped`` (not run when scoped).
    """
    config = ArchitecturePatternAnalysis().tools_config
    return {
        language: [
            {
                "name": name,
                "category": str(cfg.get("category", "")),
                "description": str(cfg.get("description", "")),
                "path_scoping": _PATH_SCOPING.get(SCOPE_MODES.get(name, UNSCOPABLE), "paths"),
            }
            for name, cfg in tools.items()
            if cfg.get("enabled", False) and name in TOOL_DISPATCH
        ]
        for language, tools in config.items()
    }


def enabled_tools_by_language() -> dict[str, list[str]]:
    """Every enabled tool key per language, in registry order.

    The single source of truth for which ``tools`` names a quality run accepts:
    it reads the same ``tools_config`` the analyzer iterates, so a tool removed
    from the registry stops being requestable in the same change.
    """
    return {
        language: [tool["name"] for tool in tools]
        for language, tools in enabled_tool_details_by_language().items()
    }
