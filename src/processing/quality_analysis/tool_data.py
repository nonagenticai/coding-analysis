"""
Complete tool data definitions for all supported languages.
Contains the full tool definitions extracted from the original tool_registry.py.
"""

from __future__ import annotations

from .tool_definitions import QualityTool, ToolCategory


def get_java_tools() -> dict[str, QualityTool]:
    """Get complete Java tool definitions."""
    return {
        "checkstyle": QualityTool(
            name="checkstyle",
            category=ToolCategory.LINTER,
            command=[
                "java",
                "-jar",
                "/tools/checkstyle-10.12.4-all.jar",
                "-c",
                "/tools/checkstyle_checks.xml",
                "-",
                "xml",
            ],
            output_format="xml",
            description="Java coding standards checker",
            installation_cmd=[
                "wget -O /tools/checkstyle-10.12.4-all.jar "
                "https://github.com/checkstyle/checkstyle/releases/download/"
                "checkstyle-10.12.4/checkstyle-10.12.4-all.jar",
                "wget -O /tools/checkstyle_checks.xml "
                "https://raw.githubusercontent.com/checkstyle/checkstyle/master/"
                "src/main/resources/google_checks.xml",
            ],
            timeout=60,
            config_files=[
                "checkstyle.xml",
                "checkstyle_checks.xml",
                "google_checks.xml",
            ],
        ),
        "pmd": QualityTool(
            name="pmd",
            category=ToolCategory.LINTER,
            command=[
                "java",
                "-jar",
                "/tools/pmd-bin-7.0.0/lib/pmd-dist-7.0.0.jar",
                "pmd",
                "-d",
                ".",
                "-R",
                "category/java/bestpractices.xml",
                "-",
                "xml",
                "-r",
                "/tmp/pmd-report.xml",  # nosec B108
            ],
            output_format="xml",
            description="Java source code analyzer for bugs and code smells",
            installation_cmd=[
                "wget -O /tmp/pmd-bin-7.0.0.zip "
                "https://github.com/pmd/pmd/releases/download/"
                "pmd_releases%2F7.0.0/pmd-bin-7.0.0.zip",
                "cd /tools && unzip /tmp/pmd-bin-7.0.0.zip",
                "chmod +x /tools/pmd-bin-7.0.0/bin/pmd",
            ],
            timeout=120,
            config_files=["pmd.xml", "pmd-ruleset.xml", ".pmdrc"],
        ),
        "spotbugs": QualityTool(
            name="spotbugs",
            category=ToolCategory.SECURITY,
            command=[
                "java",
                "-jar",
                "/tools/spotbugs-4.8.2/lib/spotbugs.jar",
                "-textui",
                "-xml",
                "-output",
                "/tmp/spotbugs-report.xml",  # nosec B108
            ],
            output_format="xml",
            description="Static analysis tool for Java bytecode to find bugs",
            installation_cmd=[
                "wget -O /tmp/spotbugs-4.8.2.tgz "
                "https://repo.maven.apache.org/maven2/com/github/spotbugs/"
                "spotbugs/4.8.2/spotbugs-4.8.2.tgz",
                "cd /tools && tar -xzf /tmp/spotbugs-4.8.2.tgz",
                "chmod +x /tools/spotbugs-4.8.2/bin/spotbugs",
            ],
            timeout=180,
            config_files=[
                "spotbugs.xml",
                "findbugs.xml",
                "spotbugs-exclude.xml",
            ],
        ),
        "google-java-format": QualityTool(
            name="google-java-format",
            category=ToolCategory.FORMATTER,
            command=[
                "java",
                "-jar",
                "/tools/google-java-format-1.18.1-all-deps.jar",
                "--dry-run",
                "--set-exit-if-changed",
            ],
            output_format="text",
            description="Java code formatter following Google Java Style",
            installation_cmd=[
                "wget -O /tools/google-java-format-1.18.1-all-deps.jar "
                "https://github.com/google/google-java-format/releases/"
                "download/v1.18.1/google-java-format-1.18.1-all-deps.jar"
            ],
            timeout=45,
            config_files=[".java-format", ".editorconfig"],
        ),
        "maven-dependency-check": QualityTool(
            name="maven-dependency-check",
            category=ToolCategory.SECURITY,
            command=[
                "mvn",
                "org.owasp:dependency-check-maven:8.4.3:check",
                "-DfailBuildOnCVSS=0",
                "-DautoUpdate=false",
                "-Dformat=XML",
                "-DoutputDirectory=/tmp",
            ],
            output_format="xml",
            description="OWASP dependency vulnerability scanner for Maven projects",
            installation_cmd=[],
            timeout=300,
            config_files=["pom.xml", "dependency-check-suppressions.xml"],
        ),
        "gradle-dependency-check": QualityTool(
            name="gradle-dependency-check",
            category=ToolCategory.SECURITY,
            command=[
                "./gradlew",
                "dependencyCheckAnalyze",
                "--info",
                "--stacktrace",
            ],
            output_format="xml",
            description="OWASP dependency vulnerability scanner for Gradle projects",
            installation_cmd=[],
            timeout=300,
            config_files=[
                "build.gradle",
                "build.gradle.kts",
                "gradle.properties",
            ],
        ),
        "jacoco": QualityTool(
            name="jacoco",
            category=ToolCategory.COMPLEXITY,
            command=["mvn", "jacoco:report", "-DskipTests=false"],
            output_format="xml",
            description="Java code coverage analysis tool",
            installation_cmd=[],
            timeout=180,
            config_files=["pom.xml", "jacoco.xml"],
        ),
        "jdepend": QualityTool(
            name="jdepend",
            category=ToolCategory.ARCHITECTURE,
            command=[
                "java",
                "-jar",
                "/tools/jdepend-2.10.jar",
                "-format",
                "xml",
            ],
            output_format="xml",
            description="Java package dependency analyzer",
            installation_cmd=[
                "wget -O /tools/jdepend-2.10.jar https://repo1.maven.org/maven2/jdepend/jdepend/2.10/jdepend-2.10.jar"
            ],
            timeout=60,
        ),
        "archunit": QualityTool(
            name="archunit",
            category=ToolCategory.ARCHITECTURE,
            command=[
                "mvn",
                "test",
                "-Dtest=ArchitectureTest",
                "-DfailIfNoTests=false",
            ],
            output_format="text",
            description="Architecture testing framework for Java",
            installation_cmd=[],
            timeout=120,
            config_files=["pom.xml", "build.gradle"],
        ),
        "errorprone": QualityTool(
            name="errorprone",
            category=ToolCategory.LINTER,
            command=[
                "mvn",
                "compile",
                "-Dmaven.compiler.showWarnings=true",
                "-Dmaven.compiler.failOnWarning=false",
            ],
            output_format="text",
            description="Static analysis tool that catches common Java mistakes",
            installation_cmd=[],
            timeout=180,
            config_files=["pom.xml", "error-prone-excludes.txt"],
        ),
    }


def get_rust_tools() -> dict[str, QualityTool]:
    """Get complete Rust tool definitions."""
    return {
        "clippy": QualityTool(
            name="clippy",
            category=ToolCategory.LINTER,
            command=[
                "cargo",
                "clippy",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ],
            output_format="text",
            description="Rust linter for catching common mistakes and updating code",
            installation_cmd=["rustup", "component", "add", "clippy"],
            timeout=120,
            config_files=["clippy.toml", ".clippy.toml"],
        ),
        "rustfmt": QualityTool(
            name="rustfmt",
            category=ToolCategory.FORMATTER,
            command=["cargo", "fmt", "--check"],
            output_format="text",
            description="Rust code formatter",
            installation_cmd=["rustup", "component", "add", "rustfmt"],
            timeout=60,
            config_files=["rustfmt.toml", ".rustfmt.toml"],
        ),
        "cargo-audit": QualityTool(
            name="cargo-audit",
            category=ToolCategory.SECURITY,
            command=["cargo", "audit", "--format", "json"],
            output_format="json",
            description="Security audit for Rust dependencies",
            installation_cmd=["cargo", "install", "cargo-audit"],
            timeout=120,
            config_files=["Cargo.toml", "Cargo.lock"],
        ),
        "cargo-deny": QualityTool(
            name="cargo-deny",
            category=ToolCategory.SECURITY,
            command=["cargo", "deny", "check"],
            output_format="text",
            description="Cargo plugin for linting dependencies",
            installation_cmd=["cargo", "install", "cargo-deny"],
            timeout=90,
            config_files=["deny.toml"],
        ),
        "cargo-outdated": QualityTool(
            name="cargo-outdated",
            category=ToolCategory.LINTER,
            command=["cargo", "outdated", "--format", "json"],
            output_format="json",
            description="Check for outdated Rust dependencies",
            installation_cmd=["cargo", "install", "cargo-outdated"],
            timeout=60,
            config_files=["Cargo.toml"],
        ),
        "cargo-bloat": QualityTool(
            name="cargo-bloat",
            category=ToolCategory.COMPLEXITY,
            command=["cargo", "bloat", "--release", "--crates"],
            output_format="text",
            description="Find what takes most of the space in your executable",
            installation_cmd=["cargo", "install", "cargo-bloat"],
            timeout=180,
            config_files=["Cargo.toml"],
        ),
        "cargo-machete": QualityTool(
            name="cargo-machete",
            category=ToolCategory.LINTER,
            command=["cargo", "machete"],
            output_format="text",
            description="Remove unused Rust dependencies",
            installation_cmd=["cargo", "install", "cargo-machete"],
            timeout=90,
            config_files=["Cargo.toml"],
        ),
        "cargo-udeps": QualityTool(
            name="cargo-udeps",
            category=ToolCategory.LINTER,
            command=["cargo", "+nightly", "udeps", "--all-targets"],
            output_format="text",
            description="Find unused dependencies in Cargo.toml",
            installation_cmd=[
                "rustup",
                "toolchain",
                "install",
                "nightly",
                "cargo",
                "install",
                "cargo-udeps",
                "--locked",
            ],
            timeout=120,
            config_files=["Cargo.toml"],
        ),
        "cargo-tarpaulin": QualityTool(
            name="cargo-tarpaulin",
            category=ToolCategory.COMPLEXITY,
            command=[
                "cargo",
                "tarpaulin",
                "--out",
                "xml",
                "--output-dir",
                "/tmp",  # nosec B108
            ],
            output_format="xml",
            description="Code coverage tool for Rust projects",
            installation_cmd=["cargo", "install", "cargo-tarpaulin"],
            timeout=300,
            config_files=["tarpaulin.toml"],
        ),
        "cargo-llvm-cov": QualityTool(
            name="cargo-llvm-cov",
            category=ToolCategory.COMPLEXITY,
            command=[
                "cargo",
                "llvm-cov",
                "--lcov",
                "--output-path",
                "/tmp/lcov.info",  # nosec B108
            ],
            output_format="text",
            description="Source-based code coverage via LLVM",
            installation_cmd=[
                "rustup",
                "component",
                "add",
                "llvm-tools-preview",
                "cargo",
                "install",
                "cargo-llvm-cov",
            ],
            timeout=300,
            config_files=["Cargo.toml"],
        ),
        "rust-analyzer": QualityTool(
            name="rust-analyzer",
            category=ToolCategory.TYPE_CHECKER,
            command=["rust-analyzer", "diagnostics", "."],
            output_format="json",
            description="LSP server for Rust with rich diagnostics",
            installation_cmd=[
                "rustup",
                "component",
                "add",
                "rust-analyzer",
            ],
            timeout=120,
            config_files=["rust-project.json"],
        ),
        "miri": QualityTool(
            name="miri",
            category=ToolCategory.SECURITY,
            command=["cargo", "+nightly", "miri", "test"],
            output_format="text",
            description="Interpreter for Rust MIR for detecting undefined behavior",
            installation_cmd=[
                "rustup",
                "toolchain",
                "install",
                "nightly",
                "rustup",
                "+nightly",
                "component",
                "add",
                "miri",
            ],
            timeout=600,
            config_files=["Cargo.toml"],
        ),
    }


def get_go_tools() -> dict[str, QualityTool]:
    """Get complete Go tool definitions."""
    return {
        "golangci-lint": QualityTool(
            name="golangci-lint",
            category=ToolCategory.LINTER,
            command=["golangci-lint", "run", "--out-format", "json"],
            output_format="json",
            description="Fast Go linters runner with many built-in linters",
            installation_cmd=[
                "curl -sSfL https://raw.githubusercontent.com/golangci/"
                "golangci-lint/master/install.sh | sh -s -- -b "
                "$(go env GOPATH)/bin v1.54.2"
            ],
            timeout=180,
            config_files=[
                ".golangci.yml",
                ".golangci.yaml",
                "golangci.yml",
            ],
        ),
        "gofmt": QualityTool(
            name="gofmt",
            category=ToolCategory.FORMATTER,
            command=["gofmt", "-l", "-d", "."],
            output_format="text",
            description="Go source code formatter",
            installation_cmd=[],
            timeout=30,
        ),
        "goimports": QualityTool(
            name="goimports",
            category=ToolCategory.FORMATTER,
            command=["goimports", "-l", "-d", "."],
            output_format="text",
            description="Go import organizer and formatter",
            installation_cmd=[
                "go",
                "install",
                "golang.org/x/tools/cmd/goimports@latest",
            ],
            timeout=60,
        ),
        "go-vet": QualityTool(
            name="go-vet",
            category=ToolCategory.LINTER,
            command=["go", "vet", "./..."],
            output_format="text",
            description="Go's built-in static analysis tool",
            installation_cmd=[],
            timeout=60,
        ),
        "staticcheck": QualityTool(
            name="staticcheck",
            category=ToolCategory.LINTER,
            command=["staticcheck", "-", "json", "./..."],
            output_format="json",
            description="Advanced Go static analysis tool",
            installation_cmd=[
                "go",
                "install",
                "honnef.co/go/tools/cmd/staticcheck@latest",
            ],
            timeout=120,
            config_files=["staticcheck.con"],
        ),
        "gosec": QualityTool(
            name="gosec",
            category=ToolCategory.SECURITY,
            command=["gosec", "-fmt", "json", "./..."],
            output_format="json",
            description="Security analyzer for Go source code",
            installation_cmd=[
                "go",
                "install",
                "github.com/securecodewarrior/gosec/v2/cmd/gosec@latest",
            ],
            timeout=120,
            config_files=["gosec.json"],
        ),
        "govulncheck": QualityTool(
            name="govulncheck",
            category=ToolCategory.SECURITY,
            command=["govulncheck", "-json", "./..."],
            output_format="json",
            description="Go vulnerability checker",
            installation_cmd=[
                "go",
                "install",
                "golang.org/x/vuln/cmd/govulncheck@latest",
            ],
            timeout=180,
        ),
    }


def get_csharp_tools() -> dict[str, QualityTool]:
    """Get complete C# tool definitions."""
    return {
        "dotnet-format": QualityTool(
            name="dotnet-format",
            category=ToolCategory.FORMATTER,
            command=[
                "dotnet",
                "format",
                "--verify-no-changes",
                "--verbosity",
                "normal",
            ],
            output_format="text",
            description="Code formatter for .NET projects",
            installation_cmd=[],
            timeout=120,
            config_files=[".editorconfig", "Directory.Build.props"],
        ),
        "dotnet-analyzers": QualityTool(
            name="dotnet-analyzers",
            category=ToolCategory.LINTER,
            command=[
                "dotnet",
                "build",
                "--no-restore",
                "--verbosity",
                "normal",
                "/p:TreatWarningsAsErrors=false",
            ],
            output_format="text",
            description="Built-in .NET code analyzers and StyleCop",
            installation_cmd=[],
            timeout=300,
            config_files=[
                "Directory.Build.props",
                "stylecop.json",
                ".globalconfig",
            ],
        ),
        "security-code-scan": QualityTool(
            name="security-code-scan",
            category=ToolCategory.SECURITY,
            command=[
                "dotnet",
                "build",
                "--no-restore",
                "/p:RunCodeAnalysis=true",
            ],
            output_format="text",
            description="Security analyzer for .NET applications",
            installation_cmd=[],
            timeout=240,
            config_files=["SecurityCodeScan.config.yml"],
        ),
    }


def get_cpp_tools() -> dict[str, QualityTool]:
    """Get complete C++ tool definitions."""
    return {
        "clang-format": QualityTool(
            name="clang-format",
            category=ToolCategory.FORMATTER,
            command=[
                "clang-format",
                "--dry-run",
                "--Werror",
                "--style=file",
            ],
            output_format="text",
            description="C++ code formatter based on Clang",
            installation_cmd=["apt-get update && apt-get install -y clang-format"],
            timeout=60,
            config_files=[".clang-format"],
        ),
        "clang-tidy": QualityTool(
            name="clang-tidy",
            category=ToolCategory.LINTER,
            command=[
                "clang-tidy",
                "--format-style=file",
                "--export-resolves=/tmp/clang-tidy-resolves.yaml",
            ],
            output_format="yaml",
            description="C++ linter and static analyzer based on Clang",
            installation_cmd=["apt-get update && apt-get install -y clang-tidy"],
            timeout=300,
            config_files=[".clang-tidy", "compile_commands.json"],
        ),
        "cppcheck": QualityTool(
            name="cppcheck",
            category=ToolCategory.LINTER,
            command=[
                "cppcheck",
                "--enable=all",
                "--xml",
                "--xml-version=2",
                "--output-file=/tmp/cppcheck.xml",
            ],
            output_format="xml",
            description="Static analysis tool for C/C++ code",
            installation_cmd=["apt-get update && apt-get install -y cppcheck"],
            timeout=180,
            config_files=["cppcheck.cfg"],
        ),
        "cpplint": QualityTool(
            name="cpplint",
            category=ToolCategory.LINTER,
            command=["cpplint", "--output=junit", "--recursive"],
            output_format="xml",
            description="C++ style checker following Google's C++ style guide",
            installation_cmd=["uv", "pip", "install", "--system", "cpplint"],
            timeout=120,
            config_files=["CPPLINT.cfg"],
        ),
        "valgrind": QualityTool(
            name="valgrind",
            category=ToolCategory.SECURITY,
            command=[
                "valgrind",
                "--tool=memcheck",
                "--xml=yes",
                "--xml-file=/tmp/valgrind.xml",
            ],
            output_format="xml",
            description="Memory error detector and profiler for C/C++",
            installation_cmd=["apt-get update && apt-get install -y valgrind"],
            timeout=600,
            config_files=["valgrind.supp"],
        ),
    }
