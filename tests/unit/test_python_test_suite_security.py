"""Unit tests for python_test_suite SecurityManager."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from python_test_suite.src.security_manager import (
    SecurityManager,
    TestValidationSecurityConfig,
)

# ---------------------------------------------------------------------------
# Constructor patterns
# ---------------------------------------------------------------------------


def test_constructor_no_args():
    sm = SecurityManager()
    assert isinstance(sm.config, TestValidationSecurityConfig)


def test_constructor_config_only():
    cfg = TestValidationSecurityConfig(max_execution_time=60)
    sm = SecurityManager(cfg)
    assert sm.config is cfg
    assert sm.config.max_execution_time == 60


def test_constructor_logger_and_config():
    fake_logger = logging.getLogger("test")
    cfg = TestValidationSecurityConfig(max_execution_time=120)
    sm = SecurityManager(fake_logger, cfg)
    assert sm.config is cfg
    assert sm.logger is fake_logger


def test_constructor_logger_no_config():
    fake_logger = logging.getLogger("test2")
    sm = SecurityManager(fake_logger)
    assert isinstance(sm.config, TestValidationSecurityConfig)


# ---------------------------------------------------------------------------
# validate_code — dangerous patterns
# ---------------------------------------------------------------------------


def test_validate_code_safe():
    sm = SecurityManager()
    result = sm.validate_code("def add(a, b):\n    return a + b\n")
    assert result["safe"] is True
    assert result["issues"] == []


def test_validate_code_detects_os_system():
    sm = SecurityManager()
    result = sm.validate_code("import os\nos.system('rm -rf /')\n")
    assert result["safe"] is False
    assert any("os.system(" in issue for issue in result["issues"])


def test_validate_code_detects_eval():
    sm = SecurityManager()
    result = sm.validate_code("eval('print(1)')\n")
    assert result["safe"] is False
    assert any("eval(" in issue for issue in result["issues"])


def test_validate_code_detects_exec():
    sm = SecurityManager()
    result = sm.validate_code("exec('x = 1')\n")
    assert result["safe"] is False
    assert any("exec(" in issue for issue in result["issues"])


def test_validate_code_detects_subprocess_run():
    sm = SecurityManager()
    result = sm.validate_code("import subprocess\nsubprocess.run(['ls'])\n")
    assert result["safe"] is False


def test_validate_code_detects_subprocess_popen():
    sm = SecurityManager()
    result = sm.validate_code("subprocess.Popen(['bash', '-c', 'id'])\n")
    assert result["safe"] is False


def test_validate_code_detects_dunder_import():
    sm = SecurityManager()
    result = sm.validate_code("m = __import__('os')\n")
    assert result["safe"] is False


def test_validate_code_detects_blocked_module():
    cfg = TestValidationSecurityConfig(blocked_modules=["socket"])
    sm = SecurityManager(cfg)
    result = sm.validate_code("import socket\nsocket.connect(('evil.com', 80))\n")
    assert result["safe"] is False
    assert any("socket" in issue for issue in result["issues"])


def test_validate_code_blocked_module_not_used():
    cfg = TestValidationSecurityConfig(blocked_modules=["socket"])
    sm = SecurityManager(cfg)
    result = sm.validate_code("x = 1\n")
    assert result["safe"] is True


def test_validate_code_multiple_issues():
    sm = SecurityManager()
    code = "eval('x')\nexec('y')\nos.system('z')\n"
    result = sm.validate_code(code)
    assert result["safe"] is False
    assert len(result["issues"]) >= 3


# ---------------------------------------------------------------------------
# validate_test_request
# ---------------------------------------------------------------------------


def test_validate_test_request_ok():
    cfg = TestValidationSecurityConfig(max_execution_time=60, max_code_length=1000)
    sm = SecurityManager(cfg)
    req = MagicMock()
    req.timeout = 30
    req.source_code = "x = 1"
    assert sm.validate_test_request(req) is True


def test_validate_test_request_timeout_too_large():
    cfg = TestValidationSecurityConfig(max_execution_time=30)
    sm = SecurityManager(cfg)
    req = MagicMock()
    req.timeout = 999
    req.source_code = ""
    req.timeout_seconds = None
    assert sm.validate_test_request(req) is False


def test_validate_test_request_code_too_long():
    cfg = TestValidationSecurityConfig(max_code_length=10)
    sm = SecurityManager(cfg)
    req = MagicMock()
    req.timeout = 5
    req.timeout_seconds = None
    req.source_code = "x" * 100
    req.code = ""
    assert sm.validate_test_request(req) is False


def test_validate_test_request_no_limits():
    sm = SecurityManager()
    sm.config.max_execution_time = 0
    sm.config.max_code_length = 0
    req = MagicMock()
    req.timeout = 9999
    req.timeout_seconds = None
    req.source_code = "x" * 99999
    req.code = ""
    # With limits at 0 (falsy), validation passes
    assert sm.validate_test_request(req) is True


# ---------------------------------------------------------------------------
# get_sandbox_config
# ---------------------------------------------------------------------------


def test_get_sandbox_config_defaults():
    sm = SecurityManager()
    cfg = sm.get_sandbox_config()
    assert cfg["allow_network"] is False
    assert cfg["allow_filesystem"] is True
    assert cfg["allow_subprocess"] is False
