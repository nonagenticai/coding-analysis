"""Unit tests for SecureRunner / security_config."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from app.core.security_config import CommandConfig, SecurityError, secure_runner

# ---------------------------------------------------------------------------
# secure_runner() call pattern
# ---------------------------------------------------------------------------


def test_secure_runner_call_with_config(mock_completed_process):
    config = CommandConfig(
        command="echo", args=["hello"], timeout=5
    )  # ty resolves root CommandConfig (NamedTuple) instead of coding-analysis @dataclass
    with patch("app.core.security_config.subprocess.run", return_value=mock_completed_process):
        result = secure_runner(config=config)
    assert result.returncode == 0


def test_secure_runner_call_with_command_arg(mock_completed_process):
    with patch("app.core.security_config.subprocess.run", return_value=mock_completed_process):
        result = secure_runner(command="echo", args=["hi"], timeout=5)
    assert result.returncode == 0


def test_secure_runner_call_no_args_raises():
    with pytest.raises(SecurityError, match="No command"):
        secure_runner()


# ---------------------------------------------------------------------------
# run_command pattern
# ---------------------------------------------------------------------------


def test_run_command_basic(mock_completed_process):
    with patch("app.core.security_config.subprocess.run", return_value=mock_completed_process):
        result = secure_runner.run_command(["echo", "hi"])
    assert result.returncode == 0


def test_run_command_with_config(mock_completed_process):
    config = CommandConfig(timeout=10)
    with patch("app.core.security_config.subprocess.run", return_value=mock_completed_process):
        result = secure_runner.run_command(["echo", "hi"], config=config)
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# allowed_commands enforcement
# ---------------------------------------------------------------------------


def test_allowed_commands_passes(mock_completed_process):
    config = CommandConfig(allowed_commands=["echo"])
    with patch("app.core.security_config.subprocess.run", return_value=mock_completed_process):
        result = secure_runner.run_command(["echo", "ok"], config=config)
    assert result.returncode == 0


def test_allowed_commands_blocks_unknown():
    config = CommandConfig(allowed_commands=["echo"])
    with pytest.raises(SecurityError, match="not in allowed commands"):
        secure_runner.run_command(["rm", "-rf", "/"], config=config)


def test_allowed_commands_empty_list_allows_all(mock_completed_process):
    config = CommandConfig(allowed_commands=[])
    with patch("app.core.security_config.subprocess.run", return_value=mock_completed_process):
        result = secure_runner.run_command(["anything"], config=config)
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Timeout raises SecurityError
# ---------------------------------------------------------------------------


def test_timeout_raises_security_error():
    config = CommandConfig(command="sleep", args=["60"], timeout=1)
    with patch(
        "app.core.security_config.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["sleep", "60"], timeout=1),
    ):
        with pytest.raises(SecurityError, match="timed out"):
            secure_runner(config=config)


def test_run_command_timeout_raises_security_error():
    config = CommandConfig(timeout=1)
    with patch(
        "app.core.security_config.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["sleep"], timeout=1),
    ):
        with pytest.raises(SecurityError, match="timed out"):
            secure_runner.run_command(["sleep", "60"], config=config)


# ---------------------------------------------------------------------------
# Command not found raises SecurityError
# ---------------------------------------------------------------------------


def test_command_not_found_raises_security_error():
    with patch(
        "app.core.security_config.subprocess.run",
        side_effect=FileNotFoundError("No such file"),
    ):
        with pytest.raises(SecurityError, match="Command not found"):
            secure_runner.run_command(["nonexistent_cmd"])


# ---------------------------------------------------------------------------
# Working directory and environment forwarding
# ---------------------------------------------------------------------------


def test_working_directory_passed_to_subprocess(mock_completed_process):
    config = CommandConfig(working_directory="/tmp")
    with patch("app.core.security_config.subprocess.run", return_value=mock_completed_process) as m:
        secure_runner.run_command(["echo"], config=config)
    call_kwargs = m.call_args[1]
    assert call_kwargs["cwd"] == "/tmp"


def test_environment_merged(mock_completed_process):
    config = CommandConfig(environment={"MY_VAR": "hello"})
    with patch("app.core.security_config.subprocess.run", return_value=mock_completed_process) as m:
        secure_runner.run_command(["printenv"], config=config)
    call_kwargs = m.call_args[1]
    assert call_kwargs["env"]["MY_VAR"] == "hello"


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


def test_stdout_truncated_to_max_output_size():
    big_output = "x" * (2 * 1024 * 1024)  # 2 MB
    fake_result = subprocess.CompletedProcess(
        args=["echo"], returncode=0, stdout=big_output, stderr=""
    )
    config = CommandConfig(max_output_size=1024)
    with patch("app.core.security_config.subprocess.run", return_value=fake_result):
        result = secure_runner.run_command(["echo"], config=config)
    assert len(result.stdout) == 1024


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_completed_process():
    return subprocess.CompletedProcess(args=["echo"], returncode=0, stdout="ok\n", stderr="")
