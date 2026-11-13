"""
Tool configuration and installation management.
Handles installation commands, configuration files, and tool setup.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ToolConfiguration:
    """Manages installation commands and configuration for quality tools."""

    def __init__(self):
        self.installation_commands = self._initialize_installation_commands()

    def get_installation_commands(self, language: str) -> list[str]:
        """Get commands to install tools in container.

        Args:
            language (str): The language parameter.

        Returns:
            List[str]: The list[str] result."""
        return self.installation_commands.get(language.lower(), [])

    def get_individual_tool_installation(self, language: str, tool_name: str) -> list[str]:
        """Get installation commands for a specific tool.

        Args:
            language: Programming language
            tool_name: Name of the tool

        Returns:
            List of installation commands
        """
        # This would be enhanced to get individual tool installation commands
        # from the tool registry or tool definitions
        return []

    def validate_configuration_files(self, language: str, project_path: str) -> dict[str, bool]:
        """Validate presence of configuration files for tools.

        Args:
            language: Programming language
            project_path: Path to the project

        Returns:
            Dict mapping config file names to their existence status
        """
        # This would check for presence of config files like ruff.toml, .eslintrc, etc.
        # Implementation would scan the project directory
        return {}

    def generate_default_configs(self, language: str, output_path: str) -> bool:
        """Generate default configuration files for tools.

        Args:
            language: Programming language
            output_path: Where to write config files

        Returns:
            True if configs were generated successfully
        """
        # This would generate default configuration files for various tools
        # Based on best practices and common settings
        return True

    def _initialize_installation_commands(self) -> dict[str, list[str]]:
        """Initialize bulk installation commands for each language."""
        return {
            "python": [
                "uv pip install --system --no-cache ruff==0.17.3 ty==0.0.92 "
                "bandit==1.7.6 vulture==2.11 radon==6.0.1 pydocstyle==6.3.0",
                "uv pip install --system --no-cache pyan3==1.2.0 code2flow==2.5.1 "
                "pycg crosshair-tool==0.0.65 pyre-check sphinx==7.2.6 "
                "sphinx-rtd-theme",
                "apt-get update && apt-get install -y graphviz",
            ],
            "javascript": ["npm install -g eslint@8.47.0 prettier@3.0.1 jshint"],
            "typescript": [
                "npm install -g eslint@8.47.0 "
                "@typescript-eslint/eslint-plugin@6.4.0 "
                "@typescript-eslint/parser@6.4.0 prettier@3.0.1 typescript@5.1.6"
            ],
            "java": [
                "mkdir -p /tools",
                "apt-get update && apt-get install -y wget unzip openjdk-17-jdk maven gradle",
                "wget -O /tools/checkstyle-10.12.4-all.jar "
                "https://github.com/checkstyle/checkstyle/releases/download/"
                "checkstyle-10.12.4/checkstyle-10.12.4-all.jar",
                "wget -O /tools/checkstyle_checks.xml "
                "https://raw.githubusercontent.com/checkstyle/checkstyle/master/"
                "src/main/resources/google_checks.xml",
                "wget -O /tmp/pmd-bin-7.0.0.zip "
                "https://github.com/pmd/pmd/releases/download/"
                "pmd_releases%2F7.0.0/pmd-bin-7.0.0.zip && cd /tools && "
                "unzip /tmp/pmd-bin-7.0.0.zip && chmod +x /tools/pmd-bin-7.0.0/bin/pmd",
                "wget -O /tmp/spotbugs-4.8.2.tgz "
                "https://repo.maven.apache.org/maven2/com/github/spotbugs/"
                "spotbugs/4.8.2/spotbugs-4.8.2.tgz && cd /tools && "
                "tar -xzf /tmp/spotbugs-4.8.2.tgz && chmod +x /tools/spotbugs-4.8.2/bin/spotbugs",
                "wget -O /tools/google-java-format-1.18.1-all-deps.jar "
                "https://github.com/google/google-java-format/releases/download/"
                "v1.18.1/google-java-format-1.18.1-all-deps.jar",
                "wget -O /tools/jdepend-2.10.jar https://repo1.maven.org/maven2/jdepend/jdepend/2.10/jdepend-2.10.jar",
                "rm -rf /tmp/pmd-bin-7.0.0.zip /tmp/spotbugs-4.8.2.tgz",
            ],
            "rust": [
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
                "source ~/.cargo/env",
                "rustup component add clippy rustfmt rust-analyzer llvm-tools-preview",
                "rustup toolchain install nightly",
                "rustup +nightly component add miri",
                "cargo install cargo-audit cargo-deny cargo-outdated "
                "cargo-bloat cargo-machete cargo-udeps cargo-tarpaulin "
                "cargo-llvm-cov",
            ],
            "go": [
                "wget -O /tmp/go1.21.5.linux-amd64.tar.gz https://go.dev/dl/go1.21.5.linux-amd64.tar.gz",
                "rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go1.21.5.linux-amd64.tar.gz",
                "export PATH=$PATH:/usr/local/go/bin",
                "export GOPATH=/go",
                "export GOBIN=/go/bin",
                "mkdir -p /go/bin",
                "curl -sSfL https://raw.githubusercontent.com/golangci/"
                "golangci-lint/master/install.sh | sh -s -- -b /go/bin v1.54.2",
                "go install golang.org/x/tools/cmd/goimports@latest",
                "go install honnef.co/go/tools/cmd/staticcheck@latest",
                "go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest",
                "go install golang.org/x/vuln/cmd/govulncheck@latest",
                "go install github.com/fzipp/gocyclo/cmd/gocyclo@latest",
                "go install github.com/uudashr/gocognit/cmd/gocognit@latest",
                "go install github.com/gordonklaus/ineffassign@latest",
                "go install github.com/client9/misspell/cmd/misspell@latest",
                "go install github.com/go-critic/go-critic/cmd/gocritic@latest",
                "go install github.com/axw/gocov/gocov@latest",
                "go install github.com/AlekSi/gocov-xml@latest",
                "rm /tmp/go1.21.5.linux-amd64.tar.gz",
            ],
            "csharp": [
                "wget -O /tmp/dotnet-install.sh https://dot.net/v1/dotnet-install.sh",
                "chmod +x /tmp/dotnet-install.sh",
                "/tmp/dotnet-install.sh --channel 8.0 --install-dir /usr/share/dotnet",  # nosec B108
                "ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet",
                "export DOTNET_ROOT=/usr/share/dotnet",
                "export PATH=$PATH:/usr/share/dotnet",
                "dotnet tool install -g roslynator.dotnet.cli",
                "dotnet tool install -g dotnet-outdated-tool",
                "dotnet tool install -g dotnet-depends",
                "dotnet tool install -g dotnet-reportgenerator-globaltool",
                "dotnet tool install -g nbgv",
                "dotnet tool install -g dotnet-retire",
                "apt-get update && apt-get install -y opam",
                "opam init --disable-sandboxing -y",
                "eval $(opam env)",
                "opam install infer -y",
                "rm /tmp/dotnet-install.sh",
            ],
            "cpp": [
                "apt-get update",
                "apt-get install -y build-essential clang clang-format "
                "clang-tidy cppcheck valgrind lcov vera++ flawfinder "
                "iwyu gcc g++ cmake make",
                "uv pip install --system --no-cache cpplint",
                "wget -O /tmp/DrMemory-Linux-2.6.0.tar.gz "
                "https://github.com/DynamoRIO/drmemory/releases/download/"
                "release_2.6.0/DrMemory-Linux-2.6.0.tar.gz",
                "cd /opt && tar -xzf /tmp/DrMemory-Linux-2.6.0.tar.gz",
                "ln -s /opt/DrMemory-Linux-2.6.0/bin/drmemory /usr/local/bin/drmemory",
                "wget -O /tmp/oclint.tar.gz "
                "https://github.com/oclint/oclint/releases/download/v22.02/"
                "oclint-22.02-llvm-13.0.1-x86_64-linux-gnu-ubuntu-20.04.tar.gz",
                "cd /opt && tar -xzf /tmp/oclint.tar.gz",
                "ln -s /opt/oclint-22.02/bin/oclint /usr/local/bin/oclint",
                "apt-get install -y opam",
                "opam init --disable-sandboxing -y",
                "eval $(opam env)",
                "opam install infer -y",
                "rm -rf /tmp/DrMemory-Linux-2.6.0.tar.gz /tmp/oclint.tar.gz",
            ],
        }

    def get_language_specific_config_templates(self, language: str) -> dict[str, str]:
        """Get configuration file templates for a language.

        Args:
            language: Programming language

        Returns:
            Dict mapping config file names to their template content
        """
        templates = {
            "python": {
                "pyproject.toml": self._get_python_pyproject_config(),
            },
            "javascript": {
                ".eslintrc.json": self._get_eslint_config(),
                ".prettierrc": self._get_prettier_config(),
            },
            "typescript": {
                ".eslintrc.json": self._get_typescript_eslint_config(),
                "tsconfig.json": self._get_tsconfig(),
            },
            # Add more language configs as needed
        }

        return templates.get(language.lower(), {})

    def _get_python_pyproject_config(self) -> str:
        """Get default Python pyproject.toml configuration (ruff + ty)."""
        return """[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
# flake8 (E/W/F) + isort (I) + pylint (PL)
select = ["E", "W", "F", "I", "PL"]

[tool.ruff.format]
quote-style = "double"

[tool.ty.environment]
python-version = "3.12"

[tool.ty.rules]
possibly-unresolved-reference = "warn"
"""

    def _get_eslint_config(self) -> str:
        """Get default ESLint configuration."""
        return """{
  "env": {
    "browser": true,
    "es2021": true,
    "node": true
  },
  "extends": [
    "eslint:recommended"
  ],
  "parserOptions": {
    "ecmaVersion": 12,
    "sourceType": "module"
  },
  "rules": {
    "indent": ["error", 2],
    "linebreak-style": ["error", "unix"],
    "quotes": ["error", "single"],
    "semi": ["error", "always"]
  }
}"""

    def _get_prettier_config(self) -> str:
        """Get default Prettier configuration."""
        return """{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2
}"""

    def _get_typescript_eslint_config(self) -> str:
        """Get default TypeScript ESLint configuration."""
        return """{
  "parser": "@typescript-eslint/parser",
  "extends": [
    "eslint:recommended",
    "@typescript-eslint/recommended"
  ],
  "plugins": ["@typescript-eslint"],
  "rules": {
    "@typescript-eslint/no-unused-vars": "error"
  }
}"""

    def _get_tsconfig(self) -> str:
        """Get default TypeScript configuration."""
        return """{
  "compilerOptions": {
    "target": "es2018",
    "module": "commonjs",
    "lib": ["es2018"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}"""
