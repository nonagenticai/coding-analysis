from processing.contract_analysis.checks.syntax_check import check_syntax


def test_valid_python3_no_violations():
    sources = {
        "good.py": "x = 1\ndef foo():\n    return x\n",
    }
    assert check_syntax(sources) == []


def test_pep758_except_syntax_tolerated():
    """PEP 758 unparenthesized except is valid Python 3.14 — not flagged."""
    sources = {
        "pep758.py": "try:\n    pass\nexcept ValueError, TypeError:\n    pass\n",
    }
    violations = check_syntax(sources)
    assert len(violations) == 0


def test_real_syntax_error_still_fails():
    """Actual syntax errors (not PEP 758) are still caught."""
    sources = {
        "bad.py": "def foo(\n",
    }
    violations = check_syntax(sources)
    assert len(violations) == 1
    assert violations[0].check == "python3_syntax"
    assert violations[0].severity == "FAIL"
    assert "bad.py" in violations[0].message
    assert "bad.py:" in violations[0].service_location


def test_other_syntax_error_fails():
    sources = {
        "missing_colon.py": "def foo()\n    pass\n",
    }
    violations = check_syntax(sources)
    assert len(violations) == 1
    assert "missing_colon.py" in violations[0].message


def test_multiple_files_one_invalid():
    sources = {
        "ok.py": "x = 1\n",
        "bad.py": "def foo(\n",
        "also_ok.py": "y = 2\n",
    }
    violations = check_syntax(sources)
    assert len(violations) == 1
    assert "bad.py" in violations[0].message


def test_empty_sources_no_violations():
    assert check_syntax({}) == []
