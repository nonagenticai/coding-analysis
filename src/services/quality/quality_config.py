#!/usr/bin/env python3
"""Quality config module."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Core Configuration Constants
MAX_ANALYSIS_TIME = int(os.getenv("MAX_ANALYSIS_TIME", "3600"))  # 1 hour default
QUALITY_CACHE_TTL = int(os.getenv("QUALITY_CACHE_TTL", "7200"))
ENABLE_AUTOFIX = os.getenv("ENABLE_AUTOFIX", "false").lower() == "true"
TEST_WORKSPACE = "/app/test_workspace"
QUALITY_CACHE_DIR = "/app/quality_cache"

# Tool-specific timeout configurations - Optimized for faster processing
TOOL_TIMEOUTS = {
    "ruff": int(os.getenv("RUFF_TIMEOUT", "300")),  # 5 minutes
    "ruff-format": int(os.getenv("RUFF_FORMAT_TIMEOUT", "300")),  # 5 minutes
    "ty": int(os.getenv("TY_TIMEOUT", "900")),  # 15 minutes
    "bandit": int(os.getenv("BANDIT_TIMEOUT", "900")),  # 15 minutes
    "eslint": int(os.getenv("ESLINT_TIMEOUT", "900")),  # 15 minutes
    "prettier": int(os.getenv("PRETTIER_TIMEOUT", "300")),  # 5 minutes
    "tsc": int(os.getenv("TSC_TIMEOUT", "900")),  # 15 minutes
    "vulture": int(os.getenv("VULTURE_TIMEOUT", "600")),  # 10 minutes
    "radon": int(os.getenv("RADON_TIMEOUT", "300")),  # 5 minutes
    # 15 minutes default
    "default": int(os.getenv("DEFAULT_TOOL_TIMEOUT", "900")),
}

# Quality tool configurations - RESOLVED with proper command paths
QUALITY_TOOLS = {
    "python": {
        "ruff-format": {
            "command": ["ruff", "format", "--check", "--output-format=json"],
            "description": "Code formatting with ruff format",
        },
        "ruff": {
            # flake8 (E/W/F) + isort (I) + pylint (PL*) rules in one pass.
            "command": ["ruff", "check", "--output-format=json", "--extend-select", "I,PL"],
            "description": "Python linting and import sorting with ruff",
        },
        "ty": {
            "command": ["ty", "check", "--output-format", "gitlab"],
            "description": "Static type checking with ty",
        },
        "bandit": {
            "command": ["bandit", "-f", "json"],
            "description": "Security vulnerability scanning",
        },
        "vulture": {
            "command": ["vulture"],
            "description": "Dead code detection",
        },
        "radon": {
            "command": ["radon", "cc", "--json"],
            "description": "Cyclomatic complexity analysis",
        },
    },
    "javascript": {
        "eslint": {
            # Use full path to ensure tool is found
            "command": ["/usr/local/bin/eslint", "--format=json"],
            "description": "JavaScript/TypeScript linting",
        },
        "prettier": {
            # Use full path to ensure tool is found
            "command": ["/usr/local/bin/prettier", "--check"],
            "description": "Code formatting with Prettier",
        },
    },
    "typescript": {
        "eslint": {
            # Use full path and TypeScript-specific extensions
            "command": [
                "/usr/local/bin/eslint",
                "--format=json",
                "--ext=.ts,.tsx",
            ],
            "description": "TypeScript linting",
        },
        "prettier": {
            "command": ["/usr/local/bin/prettier", "--check"],
            "description": "Code formatting with Prettier",
        },
        "tsc": {
            "command": ["/usr/local/bin/tsc", "--noEmit"],
            "description": "TypeScript compiler checks",
        },
    },
}


class QualityConfig:
    """Configuration manager for quality analysis service."""

    def __init__(self):
        """Initialize configuration manager."""
        self.max_analysis_time = MAX_ANALYSIS_TIME
        self.quality_cache_ttl = QUALITY_CACHE_TTL
        self.enable_autofix = ENABLE_AUTOFIX
        self.test_workspace = TEST_WORKSPACE
        self.quality_cache_dir = QUALITY_CACHE_DIR
        self.tool_timeouts = TOOL_TIMEOUTS.copy()
        self.quality_tools = QUALITY_TOOLS.copy()

    def get_tool_timeout(self, tool_name: str) -> int:
        """Get timeout for specific tool."""
        return self.tool_timeouts.get(tool_name, self.tool_timeouts["default"])

    def get_tools_for_language(self, language: str) -> dict[str, dict[str, Any]]:
        """Get available tools for a specific language."""
        return self.quality_tools.get(language, {})

    def get_tool_config(self, language: str, tool_name: str) -> dict[str, Any] | None:
        """Get configuration for a specific tool."""
        tools = self.get_tools_for_language(language)
        return tools.get(tool_name)

    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return list(self.quality_tools.keys())

    def is_tool_available(self, language: str, tool_name: str) -> bool:
        """Check if a tool is available for a language."""
        tools = self.get_tools_for_language(language)
        return tool_name in tools

    def get_service_info(self) -> dict[str, Any]:
        """Get comprehensive service information."""
        return {
            "service": "quality-analyzer",
            "version": "1.0.0",
            "supported_languages": self.get_supported_languages(),
            "available_tools": self.quality_tools,
            "features": [
                "multi_language_analysis",
                "security_scanning",
                "code_formatting",
                "complexity_analysis",
                "automated_fixes",
            ],
            "configuration": {
                "max_analysis_time": self.max_analysis_time,
                "quality_cache_ttl": self.quality_cache_ttl,
                "enable_autofix": self.enable_autofix,
                "tool_timeouts": self.tool_timeouts,
            },
        }


async def create_temp_dir(analysis_id: str) -> Path:
    """Create temporary directory for analysis with basic configuration files."""
    temp_path = Path(QUALITY_CACHE_DIR) / analysis_id
    temp_path.mkdir(parents=True, exist_ok=True)

    # Create basic ESLint configuration to prevent configuration errors
    eslint_config = {
        "env": {"browser": True, "es2021": True, "node": True},
        "extends": ["eslint:recommended"],
        "parserOptions": {"ecmaVersion": "latest", "sourceType": "module"},
        "rules": {"no-unused-vars": "warn", "no-console": "off"},
    }

    # Write ESLint config file
    eslint_config_path = temp_path / ".eslintrc.json"
    with open(eslint_config_path, "w") as f:
        json.dump(eslint_config, f, indent=2)

    # Create basic TypeScript config for TypeScript analysis
    ts_config = {
        "compilerOptions": {
            "target": "es2020",
            "module": "esnext",
            "lib": ["es2020"],
            "allowJs": True,
            "checkJs": False,
            "strict": False,
            "esModuleInterop": True,
            "skipLibCheck": True,
            "forceConsistentCasingInFileNames": True,
            "noEmit": True,
        },
        "include": ["*.ts", "*.tsx", "*.js", "*.jsx"],
        "exclude": ["node_modules"],
    }

    ts_config_path = temp_path / "tsconfig.json"
    with open(ts_config_path, "w") as f:
        json.dump(ts_config, f, indent=2)

    return temp_path


def detect_language(file_path: str, content: str | None = None) -> str:
    """Detect programming language from file extension or content."""
    if file_path:
        ext = Path(file_path).suffix.lower()
        if ext in [".py"]:
            return "python"
        elif ext in [".js", ".jsx"]:
            return "javascript"
        elif ext in [".ts", ".tsx"]:
            return "typescript"

    if content:
        # Basic heuristics for language detection
        if "def " in content or "import " in content or "from " in content:
            return "python"
        elif "function " in content or "const " in content or "let " in content:
            if ": " in content and "interface " in content:
                return "typescript"
            return "javascript"

    return "python"  # Default


# Create global configuration instance
config = QualityConfig()
