"""
Container security manager that adapts Python test suite security patterns
for multi-container environments while maintaining security isolation.
"""

import logging
import re
from datetime import UTC, datetime
from typing import Any

from models.testing_models import SecurityLevel, TestRequest
from python_test_suite.src.logging_manager import LoggingManager
from python_test_suite.src.security_manager import (
    SecurityManager as TestPySecurityManager,
)
from python_test_suite.src.security_manager import (
    TestValidationSecurityConfig as SecurityConfig,
)

logger = logging.getLogger(__name__)


class ExecutionSecurityManager:
    """
    Security manager for container-based test execution.
    Combines TestPy security validation with container-level security policies.

    This class provides a public API for container security management.
    Methods may appear unused but are part of the security framework API.
    """

    def __init__(self, config: SecurityConfig):
        self.config = config

        # Initialize TestPy security manager for code validation
        self.testpy_logger = LoggingManager.get_instance()
        self.testpy_security = TestPySecurityManager(self.testpy_logger, config)

        # Container-specific security policies would be initialized here if needed

        # Dangerous code patterns for additional validation
        self.dangerous_patterns = self._initialize_dangerous_patterns()

    async def validate_and_sanitize(self, request: TestRequest) -> TestRequest:
        """
        Comprehensive security validation and sanitization of test request.
        Combines TestPy validation with container-specific checks.

        Public API method for security validation framework.
        """
        try:
            logger.debug("Starting security validation for %s test", request.language)

            # 1. Basic request validation
            self._validate_request_structure(request)

            # 2. Use TestPy security validation for code content
            if request.code:
                result = self.testpy_security.validate_code(request.code)
                if not result.passed:
                    raise ValueError(f"Security validation failed: {result.error}")
                await self._validate_container_specific_security(request.code, request.language)

            if request.test_code:
                result = self.testpy_security.validate_code(request.test_code)
                if not result.passed:
                    raise ValueError(f"Security validation failed: {result.error}")
                await self._validate_container_specific_security(
                    request.test_code, request.language
                )

            # 3. Validate resource requirements
            self._validate_resource_requirements(request)

            # 4. Apply sanitization if needed
            sanitized_request = await self._sanitize_request(request)

            logger.info("Security validation passed for %s test", request.language)
            return sanitized_request

        except Exception as e:
            logger.error("Security validation failed: %s", e)
            raise ValueError(f"Security validation failed: {str(e)}") from e

    def get_container_security_config(self, security_level: SecurityLevel) -> dict[str, Any]:
        """Get container security configuration based on security level."""
        base_config = {
            "read_only": True,
            "network_disabled": True,
            "user": "testuser",
            "security_opt": ["no-new-privileges:true"],
            "tmpfs": {"/tmp": "noexec,nosuid,size=100m"},  # nosec B108
            "cap_drop": ["ALL"],
            "cap_add": [],  # No additional capabilities
        }

        # Adjust based on security level
        level_configs = {
            SecurityLevel.HIGH: {
                **base_config,
                "mem_limit": "256m",
                "cpu_quota": 25000,  # 0.25 CPU
                "pids_limit": 50,
                "ulimits": [
                    {"name": "nofile", "soft": 64, "hard": 64},
                    {"name": "nproc", "soft": 16, "hard": 16},
                ],
            },
            SecurityLevel.MEDIUM: {
                **base_config,
                "mem_limit": "512m",
                "cpu_quota": 50000,  # 0.5 CPU
                "pids_limit": 100,
                "ulimits": [
                    {"name": "nofile", "soft": 128, "hard": 128},
                    {"name": "nproc", "soft": 32, "hard": 32},
                ],
            },
            SecurityLevel.LOW: {
                **base_config,
                "mem_limit": "1g",
                "cpu_quota": 100000,  # 1.0 CPU
                "pids_limit": 200,
                "ulimits": [
                    {"name": "nofile", "soft": 256, "hard": 256},
                    {"name": "nproc", "soft": 64, "hard": 64},
                ],
            },
        }

        return level_configs.get(security_level, level_configs[SecurityLevel.MEDIUM])

    def validate_container_runtime_security(self, container_config: dict[str, Any]) -> bool:
        """Validate that container configuration meets security requirements."""
        required_security_features = [
            "read_only",
            "network_disabled",
            "user",
            "security_opt",
            "mem_limit",
        ]

        for feature in required_security_features:
            if feature not in container_config:
                logger.error("Missing required security feature: %s", feature)
                return False

        # Validate specific security settings
        if container_config.get("user") == "root":
            logger.error("Container cannot run as root user")
            return False

        if not container_config.get("read_only"):
            logger.warning("Container should be read-only for security")

        if not container_config.get("network_disabled"):
            logger.warning("Container should have network disabled")

        return True

    async def scan_for_security_violations(self, code: str, language: str) -> list[dict[str, Any]]:
        """Scan code for potential security violations."""
        violations = []

        # Check for dangerous patterns
        for pattern_name, pattern_info in self.dangerous_patterns.get(language, {}).items():
            pattern = pattern_info["pattern"]
            severity = pattern_info["severity"]
            description = pattern_info["description"]

            matches = re.findall(pattern, code, re.IGNORECASE | re.MULTILINE)
            if matches:
                violations.append(
                    {
                        "type": "dangerous_pattern",
                        "pattern": pattern_name,
                        "severity": severity,
                        "description": description,
                        "matches": matches[:5],  # Limit to first 5 matches
                    }
                )

        # Check for suspicious imports
        suspicious_imports = await self._check_suspicious_imports(code, language)
        violations.extend(suspicious_imports)

        # Check for file system access attempts
        fs_violations = await self._check_filesystem_access(code, language)
        violations.extend(fs_violations)

        return violations

    def _validate_request_structure(self, request: TestRequest) -> None:
        """Validate basic request structure and parameters."""
        if not request.code and not request.test_code:
            raise ValueError("Request must contain either code or test_code")

        if not request.language:
            raise ValueError("Language must be specified")

        if request.language not in ["python", "javascript", "typescript"]:
            raise ValueError(f"Unsupported language: {request.language}")

        # Validate timeout
        if request.timeout and request.timeout > self.config.max_execution_time:
            raise ValueError(
                f"Timeout exceeds maximum: {request.timeout} > {self.config.max_execution_time}"
            )

    async def _validate_container_specific_security(self, code: str, language: str) -> None:
        """Additional container-specific security validation."""
        # Check for container escape attempts
        escape_patterns = [
            r"docker\s+run",
            r"docker\s+exec",
            r"/proc/self/",
            r"/sys/",
            r"mount\s+",
            r"chroot\s+",
            r"unshare\s+",
            r"nsenter\s+",
        ]

        for pattern in escape_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise ValueError(f"Potential container escape attempt detected: {pattern}")

        # Language-specific validations
        if language == "python":
            await self._validate_python_security(code)
        elif language in ["javascript", "typescript"]:
            await self._validate_javascript_security(code)

    async def _validate_python_security(self, code: str) -> None:
        """Python-specific security validation."""
        # Check for dangerous Python imports
        dangerous_imports = [
            "subprocess",
            "os",
            "sys",
            "socket",
            "urllib",
            "requests",
            "shutil",
            "tempfile",
            "pickle",
            "marshal",
            "imp",
            "importlib",
        ]

        for imp in dangerous_imports:
            patterns = [
                rf"import\s+{imp}",
                rf"from\s+{imp}\s+import",
                r"__import__\s*\(\s*[\'\"]{imp}[\'\"]",
            ]

            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    if imp not in self.config.allowed_modules:
                        raise ValueError(f"Dangerous import detected: {imp}")

        # Check for eval/exec usage
        dangerous_functions = [
            "eval",
            "exec",
            "compile",
            "__import__",
            "globals",
            "locals",
            "vars",
        ]
        for func in dangerous_functions:
            if re.search(rf"\b{func}\s*\(", code):
                raise ValueError(f"Dangerous function detected: {func}")

    async def _validate_javascript_security(self, code: str) -> None:
        """JavaScript-specific security validation."""
        # Check for dangerous JavaScript patterns
        dangerous_patterns = [
            r"require\s*\(\s*[\'\"](fs|child_process|cluster|os|path|url)[\'\"]\s*\)",
            r"process\.",
            r"global\.",
            r"__dirname",
            r"__filename",
            r"Buffer\.",
            r"eval\s*\(",
            r"Function\s*\(",
            r"setTimeout\s*\(",
            r"setInterval\s*\(",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise ValueError(f"Dangerous JavaScript pattern detected: {pattern}")

    def _validate_resource_requirements(self, request: TestRequest) -> None:
        """Validate resource requirements against security limits."""
        total_code = (request.code or "") + (request.test_code or "")

        if len(total_code) > self.config.max_code_length:
            raise ValueError(
                f"Code length exceeds limit: {len(total_code)} > {self.config.max_code_length}"
            )

        if request.timeout and request.timeout > self.config.max_execution_time:
            raise ValueError(
                f"Timeout exceeds limit: {request.timeout} > {self.config.max_execution_time}"
            )

    async def _sanitize_request(self, request: TestRequest) -> TestRequest:
        """Sanitize request by removing/modifying dangerous elements."""
        sanitized_request = request.model_copy()

        # Sanitize code content
        if sanitized_request.code:
            sanitized_request.code = await self._sanitize_code_content(
                sanitized_request.code, sanitized_request.language
            )

        if sanitized_request.test_code:
            sanitized_request.test_code = await self._sanitize_code_content(
                sanitized_request.test_code, sanitized_request.language
            )

        # Sanitize request metadata
        sanitized_request = await self._sanitize_request_metadata(sanitized_request)

        return sanitized_request

    async def _check_suspicious_imports(self, code: str, language: str) -> list[dict[str, Any]]:
        """Check for suspicious import patterns."""
        violations = []

        if language == "python":
            # Pattern for dynamic imports
            dynamic_import_pattern = r"__import__\s*\([^)]+\)"
            matches = re.findall(dynamic_import_pattern, code)
            if matches:
                violations.append(
                    {
                        "type": "suspicious_import",
                        "pattern": "dynamic_import",
                        "severity": "high",
                        "description": "Dynamic imports detected",
                        "matches": matches,
                    }
                )

        return violations

    async def _check_filesystem_access(self, code: str, language: str) -> list[dict[str, Any]]:
        """Check for file system access attempts."""
        violations = []

        fs_patterns = {
            "python": [
                r"open\s*\([^)]*[\'\"]/[^)]*\)",
                r"Path\s*\([^)]*[\'\"]/[^)]*\)",
                r"os\.path\.",
                r"pathlib\.",
            ],
            "javascript": [r"fs\.", r"path\.", r"readFile", r"writeFile"],
        }

        for pattern in fs_patterns.get(language, []):
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                violations.append(
                    {
                        "type": "filesystem_access",
                        "pattern": pattern,
                        "severity": "medium",
                        "description": "File system access detected",
                        "matches": matches[:3],
                    }
                )

        return violations

    def _initialize_dangerous_patterns(
        self,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Initialize dangerous code patterns by language."""
        return {
            "python": {
                "shell_injection": {
                    "pattern": r"(os\.system|subprocess\.call|subprocess\.run|"
                    r"subprocess\.Popen)\s*\([^)]*\)",
                    "severity": "critical",
                    "description": "Shell command execution detected",
                },
                "network_access": {
                    "pattern": r"(socket\.|urllib\.|requests\.|http\.)",
                    "severity": "high",
                    "description": "Network access attempt detected",
                },
                "file_manipulation": {
                    "pattern": r"(shutil\.|tempfile\.|pickle\.|marshal\.)",
                    "severity": "medium",
                    "description": "File manipulation functions detected",
                },
            },
            "javascript": {
                "process_access": {
                    "pattern": r"process\.(env|argv|cwd|exit)",
                    "severity": "high",
                    "description": "Process object access detected",
                },
                "require_dangerous": {
                    "pattern": r'require\s*\(\s*["\'](child_process|fs|os|cluster)["\']\s*\)',
                    "severity": "critical",
                    "description": "Dangerous module require detected",
                },
                "eval_usage": {
                    "pattern": r"(eval|Function)\s*\(",
                    "severity": "high",
                    "description": "Code evaluation detected",
                },
            },
            "typescript": {
                "process_access": {
                    "pattern": r"process\.(env|argv|cwd|exit)",
                    "severity": "high",
                    "description": "Process object access detected",
                },
                "import_dangerous": {
                    "pattern": r'import.*["\'](child_process|fs|os|cluster)["\']',
                    "severity": "critical",
                    "description": "Dangerous module import detected",
                },
            },
        }

    async def _sanitize_code_content(self, code: str, language: str) -> str:
        """Sanitize code content by removing or commenting out dangerous patterns."""
        sanitized_code = code

        # Remove null bytes and control characters
        sanitized_code = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", sanitized_code)

        # Limit code length
        max_length = getattr(self.config, "max_code_length", 100000)
        if len(sanitized_code) > max_length:
            logger.warning(
                "Code truncated from %d to %d characters",
                len(sanitized_code),
                max_length,
            )
            sanitized_code = sanitized_code[:max_length]
            # Add a comment indicating truncation
            sanitized_code += f"\n# Code truncated at {max_length} characters for security"

        # Language-specific sanitization
        if language == "python":
            sanitized_code = await self._sanitize_python_code(sanitized_code)
        elif language in ["javascript", "typescript"]:
            sanitized_code = await self._sanitize_javascript_code(sanitized_code)

        # Remove excessive whitespace and normalize line endings
        sanitized_code = re.sub(r"\r\n|\r", "\n", sanitized_code)  # Normalize line endings
        sanitized_code = re.sub(r"\n{3,}", "\n\n", sanitized_code)  # Limit consecutive newlines

        return sanitized_code.strip()

    async def _sanitize_python_code(self, code: str) -> str:
        """Python-specific code sanitization."""
        sanitized = code

        # Comment out dangerous imports (instead of removing to preserve line numbers)
        dangerous_imports = [
            "subprocess",
            "os",
            "sys",
            "socket",
            "urllib",
            "requests",
            "shutil",
            "tempfile",
            "pickle",
            "marshal",
            "imp",
            "importlib",
        ]

        for imp in dangerous_imports:
            if imp not in self.config.allowed_modules:
                # Comment out import statements
                patterns = [
                    (rf"^(\s*)(import\s+{imp}(?:\s|$))", r"\1# SANITIZED: \2"),
                    (rf"^(\s*)(from\s+{imp}\s+import)", r"\1# SANITIZED: \2"),
                    (
                        r"^(\s*)(__import__\s*\(\s*[\'\"]{imp}[\'\"])",
                        r"\1# SANITIZED: \2",
                    ),
                ]

                for pattern, replacement in patterns:
                    sanitized = re.sub(
                        pattern,
                        replacement,
                        sanitized,
                        flags=re.MULTILINE | re.IGNORECASE,
                    )

        # Comment out dangerous function calls
        dangerous_functions = [
            "eval",
            "exec",
            "compile",
            "__import__",
            "globals",
            "locals",
            "vars",
        ]
        for func in dangerous_functions:
            pattern = rf"^(\s*)({func}\s*\()"
            replacement = r"\1# SANITIZED: \2"
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.MULTILINE)

        # Remove or comment shell commands
        shell_patterns = [
            (r"^(\s*)(os\.system\s*\([^)]+\))", r"\1# SANITIZED: \2"),
            (r"^(\s*)(subprocess\.[^(]+\([^)]+\))", r"\1# SANITIZED: \2"),
        ]

        for pattern, replacement in shell_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.MULTILINE)

        return sanitized

    async def _sanitize_javascript_code(self, code: str) -> str:
        """JavaScript/TypeScript-specific code sanitization."""
        sanitized = code

        # Comment out dangerous requires/imports
        dangerous_modules = [
            "fs",
            "child_process",
            "cluster",
            "os",
            "path",
            "url",
        ]

        for dangerous_module in dangerous_modules:
            escaped_module = re.escape(dangerous_module)
            # Comment out require statements
            require_pattern = rf"^(\s*)(.*require\s*\(\s*[\'\"]{escaped_module}\s*[\'\"]\s*\))"
            sanitized = re.sub(
                require_pattern,
                r"\1// SANITIZED: \2",
                sanitized,
                flags=re.MULTILINE | re.IGNORECASE,
            )

            # Comment out ES6 imports
            import_pattern = rf"^(\s*)(import.*[\'\"]({escaped_module})[\'\"])"
            sanitized = re.sub(
                import_pattern,
                r"\1// SANITIZED: \2",
                sanitized,
                flags=re.MULTILINE | re.IGNORECASE,
            )

        # Comment out dangerous function calls
        dangerous_patterns = [
            (r"^(\s*)(eval\s*\()", r"\1// SANITIZED: \2"),
            (r"^(\s*)(Function\s*\()", r"\1// SANITIZED: \2"),
            (r"^(\s*)(process\.)", r"\1// SANITIZED: \2"),
        ]

        for pattern, replacement in dangerous_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.MULTILINE)

        return sanitized

    async def _sanitize_request_metadata(self, request: TestRequest) -> TestRequest:
        """Sanitize request metadata and parameters."""
        # Ensure timeout is within limits
        max_timeout = getattr(self.config, "max_execution_time", 30)
        if request.timeout and request.timeout > max_timeout:
            logger.warning("Timeout reduced from %s to %s", request.timeout, max_timeout)
            request.timeout = max_timeout

        # Sanitize description and other string fields
        if hasattr(request, "description") and request.description:
            description = str(request.description)
            request.description = self._sanitize_string_field(description, max_length=500)  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

        # Ensure security level is valid
        if hasattr(request, "security_level") and request.security_level:
            valid_levels = [level.value for level in SecurityLevel]
            if request.security_level not in valid_levels:
                logger.warning(
                    "Invalid security level %s, defaulting to MEDIUM",
                    request.security_level,
                )
                request.security_level = SecurityLevel.MEDIUM

        return request

    def _sanitize_string_field(self, value: str, max_length: int = 1000) -> str:
        """Sanitize a string field by removing dangerous content."""
        if not value:
            return value

        # Remove null bytes and control characters
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", value)

        # Remove potential script tags and dangerous HTML
        sanitized = re.sub(
            r"<\s*script[^>]*>.*?<\s*/\s*script\s*>",
            "",
            sanitized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        sanitized = re.sub(
            r"<\s*iframe[^>]*>.*?<\s*/\s*iframe\s*>",
            "",
            sanitized,
            flags=re.IGNORECASE | re.DOTALL,
        )
        sanitized = re.sub(r"javascript\s*:", "", sanitized, flags=re.IGNORECASE)

        # Limit length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            logger.warning("String field truncated to %d characters", max_length)

        return sanitized.strip()

    async def create_sanitization_report(
        self, original_request: TestRequest, sanitized_request: TestRequest
    ) -> dict[str, Any]:
        """Create a report of what was sanitized."""
        changes = []

        # Check code changes
        if original_request.code != sanitized_request.code:
            changes.append(
                {
                    "field": "code",
                    "type": "code_sanitization",
                    "original_length": len(original_request.code or ""),
                    "sanitized_length": len(sanitized_request.code or ""),
                    "modifications": "Dangerous patterns commented out or removed",
                }
            )

        if original_request.test_code != sanitized_request.test_code:
            changes.append(
                {
                    "field": "test_code",
                    "type": "code_sanitization",
                    "original_length": len(original_request.test_code or ""),
                    "sanitized_length": len(sanitized_request.test_code or ""),
                    "modifications": "Dangerous patterns commented out or removed",
                }
            )

        # Check timeout changes
        if original_request.timeout != sanitized_request.timeout:
            changes.append(
                {
                    "field": "timeout",
                    "type": "parameter_adjustment",
                    "original_value": original_request.timeout,
                    "sanitized_value": sanitized_request.timeout,
                    "reason": "Exceeded maximum allowed timeout",
                }
            )

        return {
            "sanitization_applied": len(changes) > 0,
            "changes_count": len(changes),
            "changes": changes,
            "timestamp": datetime.now(UTC).isoformat(),
        }


# Factory function for dependency injection
def create_test_security_manager(
    config: SecurityConfig,
) -> ExecutionSecurityManager:
    """Factory function to create ExecutionSecurityManager."""
    return ExecutionSecurityManager(config)
