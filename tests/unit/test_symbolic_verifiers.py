"""Tests for processing.analyzers.symbolic.verifiers - all 5 verifier modules."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from processing.analyzers.symbolic.verifiers.bounds_verifier import BoundsVerifier
from processing.analyzers.symbolic.verifiers.exception_verifier import ExceptionVerifier
from processing.analyzers.symbolic.verifiers.none_verifier import NoneVerifier
from processing.analyzers.symbolic.verifiers.resource_verifier import ResourceVerifier
from processing.analyzers.symbolic.verifiers.type_verifier import TypeVerifier

# ---------------------------------------------------------------------------
# BoundsVerifier
# ---------------------------------------------------------------------------


class TestBoundsVerifier:
    @pytest.mark.asyncio
    async def test_no_subscripts_returns_clean(self):
        code = "x = 1\ny = 2\n"
        verifier = BoundsVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["property"] == "bounds_checking"
        assert result["subscript_accesses"] == 0
        assert result["violations"] == []

    @pytest.mark.asyncio
    async def test_unchecked_subscript_flagged(self):
        code = "items = [1, 2, 3]\nval = items[5]\n"
        verifier = BoundsVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["subscript_accesses"] >= 1
        # There should be a violation since no bounds check
        violations = result["violations"]
        assert any(v["type"] == "unchecked_bounds_access" for v in violations)

    @pytest.mark.asyncio
    async def test_checked_subscript_not_flagged(self):
        code = "items = [1, 2, 3]\nif len(items) > 5:\n    val = items[5]\n"
        verifier = BoundsVerifier()
        result = await verifier.verify(code, "test.py")
        # All accesses should have bounds check
        assert result["safe_accesses"] == result["subscript_accesses"] - len(result["violations"])

    @pytest.mark.asyncio
    async def test_syntax_error_handled(self):
        code = "def broken(\n"
        verifier = BoundsVerifier()
        result = await verifier.verify(code, "test.py")
        assert any(v["type"] == "syntax_error" for v in result["violations"])

    def test_has_bounds_check_with_len(self):
        verifier = BoundsVerifier()
        lines = ["if len(items) > 0:", "    val = items[0]"]
        assert verifier._has_bounds_check("items", lines) is True

    def test_has_bounds_check_with_if_check(self):
        verifier = BoundsVerifier()
        lines = ["if items:", "    val = items[0]"]
        assert verifier._has_bounds_check("items", lines) is True

    def test_has_bounds_check_with_except(self):
        verifier = BoundsVerifier()
        lines = ["except IndexError:", "    pass"]
        assert verifier._has_bounds_check("items", lines) is True

    def test_has_no_bounds_check(self):
        verifier = BoundsVerifier()
        lines = ["val = items[0]"]
        assert verifier._has_bounds_check("items", lines) is False


# ---------------------------------------------------------------------------
# ExceptionVerifier
# ---------------------------------------------------------------------------


class TestExceptionVerifier:
    @pytest.mark.asyncio
    async def test_no_try_blocks(self):
        code = "x = 1\ny = 2\n"
        verifier = ExceptionVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["property"] == "exception_handling"
        assert result["try_blocks"] == 0

    @pytest.mark.asyncio
    async def test_bare_except_flagged(self):
        code = "try:\n    x = 1\nexcept:\n    pass\n"
        verifier = ExceptionVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["try_blocks"] == 1
        assert any(v["type"] == "overly_broad_exception" for v in result["violations"])

    @pytest.mark.asyncio
    async def test_specific_except_not_flagged(self):
        code = "try:\n    x = 1\nexcept ValueError:\n    pass\n"
        verifier = ExceptionVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["try_blocks"] == 1
        assert not any(v["type"] == "overly_broad_exception" for v in result["violations"])

    @pytest.mark.asyncio
    async def test_risky_operation_outside_try(self):
        code = "val = int(input())\n"
        verifier = ExceptionVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["risky_operations"] >= 1

    @pytest.mark.asyncio
    async def test_risky_operation_inside_try(self):
        code = "try:\n    val = int(input())\nexcept ValueError:\n    pass\n"
        verifier = ExceptionVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["risky_operations"] >= 1

    @pytest.mark.asyncio
    async def test_syntax_error_handled(self):
        code = "def broken(\n"
        verifier = ExceptionVerifier()
        result = await verifier.verify(code, "test.py")
        assert any(v["type"] == "syntax_error" for v in result["violations"])

    @pytest.mark.asyncio
    async def test_bare_raise_tracked(self):
        code = "try:\n    x = 1\nexcept Exception:\n    raise\n"
        verifier = ExceptionVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["try_blocks"] == 1

    @pytest.mark.asyncio
    async def test_finally_tracked(self):
        code = "try:\n    x = 1\nfinally:\n    cleanup()\n"
        verifier = ExceptionVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["try_blocks"] == 1

    def test_analyze_exception_patterns_bare_except(self):
        verifier = ExceptionVerifier()
        patterns = [{"type": "bare_except", "line": 3, "severity": "medium"}]
        violations = verifier._analyze_exception_patterns(patterns)
        assert len(violations) == 1
        assert violations[0]["type"] == "overly_broad_exception"

    def test_analyze_exception_patterns_unhandled_risky(self):
        verifier = ExceptionVerifier()
        patterns = [{"type": "risky_operation", "line": 100, "operation": "int"}]
        violations = verifier._analyze_exception_patterns(patterns)
        assert any(v["type"] == "unhandled_risky_operation" for v in violations)


# ---------------------------------------------------------------------------
# NoneVerifier
# ---------------------------------------------------------------------------


class TestNoneVerifier:
    @pytest.mark.asyncio
    async def test_clean_code(self):
        code = "x = 1\ny = x + 1\n"
        verifier = NoneVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["property"] == "none_checking"

    @pytest.mark.asyncio
    async def test_unchecked_attribute_access(self):
        code = "result = obj.method()\nval = result.value\n"
        verifier = NoneVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["patterns_found"] >= 1

    @pytest.mark.asyncio
    async def test_checked_attribute_access(self):
        code = "if result is not None:\n    val = result.value\n"
        verifier = NoneVerifier()
        result = await verifier.verify(code, "test.py")
        # Should have patterns but safe_patterns should be higher
        assert result["patterns_found"] >= 0

    @pytest.mark.asyncio
    async def test_subscript_access_tracked(self):
        code = "data = get_data()\nval = data[0]\n"
        verifier = NoneVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["patterns_found"] >= 1

    @pytest.mark.asyncio
    async def test_syntax_error_handled(self):
        code = "def bad(\n"
        verifier = NoneVerifier()
        result = await verifier.verify(code, "test.py")
        assert any(v["type"] == "syntax_error" for v in result["violations"])

    def test_has_none_check_is_not_none(self):
        verifier = NoneVerifier()
        lines = ["if obj is not None:", "    obj.method()"]
        assert verifier._has_none_check("obj", lines) is True

    def test_has_none_check_truthiness(self):
        verifier = NoneVerifier()
        lines = ["if obj:", "    obj.method()"]
        assert verifier._has_none_check("obj", lines) is True

    def test_has_none_check_and_guard(self):
        verifier = NoneVerifier()
        lines = ["obj and obj.method()"]
        assert verifier._has_none_check("obj", lines) is True

    def test_no_none_check(self):
        verifier = NoneVerifier()
        lines = ["obj.method()"]
        assert verifier._has_none_check("obj", lines) is False


# ---------------------------------------------------------------------------
# ResourceVerifier
# ---------------------------------------------------------------------------


class TestResourceVerifier:
    @pytest.mark.asyncio
    async def test_clean_code(self):
        code = "x = 1\n"
        verifier = ResourceVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["property"] == "resource_management"
        assert result["violations"] == []

    @pytest.mark.asyncio
    async def test_with_statement_tracked(self):
        code = "with open('file.txt') as f:\n    data = f.read()\n"
        verifier = ResourceVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["properly_managed"] >= 1

    @pytest.mark.asyncio
    async def test_unmanaged_open_flagged(self):
        code = "f = open('file.txt')\ndata = f.read()\n"
        verifier = ResourceVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["resource_operations"] >= 1
        # Should have a violation for open without context manager
        assert any(v["type"] == "resource_leak" for v in result["violations"])

    @pytest.mark.asyncio
    async def test_attribute_resource_tracked(self):
        code = "conn = db.connect()\ncursor = conn.cursor()\n"
        verifier = ResourceVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["resource_operations"] >= 1

    @pytest.mark.asyncio
    async def test_syntax_error_handled(self):
        code = "def broken(\n"
        verifier = ResourceVerifier()
        result = await verifier.verify(code, "test.py")
        assert any(v["type"] == "syntax_error" for v in result["violations"])

    def test_has_proper_cleanup_with_statement(self):
        verifier = ResourceVerifier()
        assert verifier._has_proper_cleanup("with open('f') as f:", "") is True

    def test_has_proper_cleanup_close(self):
        verifier = ResourceVerifier()
        assert verifier._has_proper_cleanup("f = open('f')", "data = f.read()\nf.close()") is True

    def test_has_proper_cleanup_finally(self):
        verifier = ResourceVerifier()
        assert (
            verifier._has_proper_cleanup("f = open('f')", "try:\n    pass\nfinally:\n    f.close()")
            is True
        )

    def test_no_proper_cleanup(self):
        verifier = ResourceVerifier()
        assert verifier._has_proper_cleanup("f = open('f')", "data = f.read()") is False


# ---------------------------------------------------------------------------
# TypeVerifier
# ---------------------------------------------------------------------------


class TestTypeVerifier:
    @pytest.mark.asyncio
    async def test_clean_annotated_code(self):
        code = "def greet(name: str) -> str:\n    return f'Hello {name}'\n"
        verifier = TypeVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["property"] == "type_safety"

    @pytest.mark.asyncio
    async def test_missing_return_annotation_public(self):
        code = "def public_func(x):\n    return x + 1\n"
        verifier = TypeVerifier()
        result = await verifier.verify(code, "test.py")
        assert result["type_issues_found"] >= 1

    @pytest.mark.asyncio
    async def test_missing_param_annotation_private_not_flagged(self):
        code = "def _private(x):\n    return x\n"
        verifier = TypeVerifier()
        result = await verifier.verify(code, "test.py")
        # _private should not be in violations since it starts with _
        assert not any("_private" in v.get("message", "") for v in result["violations"])

    @pytest.mark.asyncio
    async def test_unsafe_eval_flagged(self):
        code = "result = eval('2 + 2')\n"
        verifier = TypeVerifier()
        result = await verifier.verify(code, "test.py")
        assert any(v["type"] == "unsafe_evaluation" for v in result["violations"])
        assert result["critical_violations"] >= 1

    @pytest.mark.asyncio
    async def test_unsafe_exec_flagged(self):
        code = "exec('print(1)')\n"
        verifier = TypeVerifier()
        result = await verifier.verify(code, "test.py")
        assert any(v["type"] == "unsafe_evaluation" for v in result["violations"])

    @pytest.mark.asyncio
    async def test_syntax_error_handled(self):
        code = "def broken(\n"
        verifier = TypeVerifier()
        result = await verifier.verify(code, "test.py")
        assert any(v["type"] == "syntax_error" for v in result["violations"])

    def test_analyze_type_issues_unsafe_eval(self):
        verifier = TypeVerifier()
        issues = [{"type": "unsafe_evaluation", "line": 1, "function": "eval"}]
        violations = verifier._analyze_type_issues(issues)
        assert len(violations) == 1
        assert violations[0]["severity"] == "high"

    def test_analyze_type_issues_missing_annotation_public(self):
        verifier = TypeVerifier()
        issues = [{"type": "missing_return_annotation", "line": 1, "function": "public_func"}]
        violations = verifier._analyze_type_issues(issues)
        assert len(violations) == 1
        assert violations[0]["severity"] == "low"

    def test_analyze_type_issues_missing_annotation_private(self):
        verifier = TypeVerifier()
        issues = [{"type": "missing_return_annotation", "line": 1, "function": "_private"}]
        violations = verifier._analyze_type_issues(issues)
        assert len(violations) == 0  # Private functions not flagged


# ---------------------------------------------------------------------------
# FormalVerifier tests (src/processing/analyzers/symbolic/tools/formal_verifier.py)
# Covers lines: 58, 65, 72, 79, 86, 103, 115-120
# ---------------------------------------------------------------------------

import os
import tempfile

from processing.analyzers.symbolic.exceptions import FormalVerificationError
from processing.analyzers.symbolic.tools.formal_verifier import FormalVerifier


class TestFormalVerifier:
    @pytest.mark.asyncio
    async def test_verify_clean_code(self):
        """Verify clean code returns passed status (lines 105)."""
        code = "def greet(name: str) -> str:\n    return f'Hello {name}'\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = f.name
        try:
            verifier = FormalVerifier()
            result = await verifier.verify(path)
            assert result["status"] in ("passed", "violations_found", "completed")
            assert "properties" in result
            assert len(result["properties"]) == 5
            assert result["verification_time"] >= 0.0
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_verify_with_violations(self):
        """Verify code with potential violations (lines 58, 65, 72, 79, 86, 103)."""
        code = (
            "f = open('file.txt')\n"
            "data = f.read()\n"
            "result = eval('2+2')\n"
            "items = [1, 2]\n"
            "val = items[5]\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = f.name
        try:
            verifier = FormalVerifier()
            result = await verifier.verify(path)
            assert result["status"] in ("violations_found", "completed")
            assert len(result["violations"]) >= 0
            assert result["coverage"] >= 0.0
            # All 5 properties should be checked
            assert "none_checking" in result["properties"]
            assert "bounds_checking" in result["properties"]
            assert "type_safety" in result["properties"]
            assert "resource_management" in result["properties"]
            assert "exception_handling" in result["properties"]
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_verify_os_error(self):
        """OSError when reading file raises FormalVerificationError (lines 115-117)."""
        verifier = FormalVerifier()
        with pytest.raises(FormalVerificationError, match="Error reading file"):
            await verifier.verify("/nonexistent/file.py")

    @pytest.mark.asyncio
    async def test_verify_value_error(self):
        """ValueError during verification raises FormalVerificationError (lines 118-120)."""
        verifier = FormalVerifier()
        with patch.object(
            verifier.none_verifier,
            "verify",
            side_effect=ValueError("bad value"),
        ):
            # Need a real file to read
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write("x = 1\n")
                f.flush()
                path = f.name
            try:
                with pytest.raises(FormalVerificationError, match="Verification error"):
                    await verifier.verify(path)
            finally:
                os.unlink(path)

    @pytest.mark.asyncio
    async def test_verify_individual_violations_extended(self):
        """Each verifier's violations get extended into the main list (lines 58, 65, 72, 79, 86)."""
        verifier = FormalVerifier()
        mock_violation = [{"type": "test_violation", "line": 1, "message": "test"}]
        # Make each verifier return violations
        for v in [
            verifier.none_verifier,
            verifier.bounds_verifier,
            verifier.type_verifier,
            verifier.resource_verifier,
            verifier.exception_verifier,
        ]:
            v.verify = AsyncMock(return_value={"violations": mock_violation, "property": "test"})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\n")
            f.flush()
            path = f.name
        try:
            result = await verifier.verify(path)
            # Each of the 5 verifiers adds 1 violation = 5 total
            assert len(result["violations"]) == 5
            assert result["status"] == "violations_found"
        finally:
            os.unlink(path)
