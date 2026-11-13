"""Tests for SemanticAnalysisService.analyze()."""

from __future__ import annotations

import pytest

from services.semantic_analysis_service import (
    SemanticAnalysisService,
    get_semantic_analysis_service,
)


@pytest.fixture
def service() -> SemanticAnalysisService:
    return SemanticAnalysisService()


# ---------------------------------------------------------------------------
# Python code analysis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_python_functions(service: SemanticAnalysisService):
    code = """\
def greet(name):
    return f"Hello, {name}"

async def fetch_data(url):
    pass
"""
    result = await service.analyze(code, "python")

    assert result["language"] == "python"
    assert result["errors"] == []

    names = {e["name"] for e in result["entities"]}
    assert "greet" in names
    assert "fetch_data" in names

    # Verify async flag
    async_funcs = [e for e in result["entities"] if e.get("is_async")]
    assert len(async_funcs) == 1
    assert async_funcs[0]["name"] == "fetch_data"


@pytest.mark.asyncio
async def test_analyze_python_classes(service: SemanticAnalysisService):
    code = """\
class Base:
    pass

class Child(Base):
    def method(self):
        pass
"""
    result = await service.analyze(code, "python")

    class_entities = [e for e in result["entities"] if e["type"] == "class"]
    assert len(class_entities) == 2

    # Inheritance relationship
    inherits = [r for r in result["relationships"] if r["type"] == "inherits"]
    assert len(inherits) == 1
    assert inherits[0]["source"] == "Child"
    assert inherits[0]["target"] == "Base"


@pytest.mark.asyncio
async def test_analyze_python_imports(service: SemanticAnalysisService):
    code = """\
import os
from pathlib import Path
"""
    result = await service.analyze(code, "python")

    import_entities = [e for e in result["entities"] if e["type"] == "import"]
    assert len(import_entities) == 2
    import_names = {e["name"] for e in import_entities}
    assert "os" in import_names
    assert "pathlib.Path" in import_names


@pytest.mark.asyncio
async def test_analyze_python_calls(service: SemanticAnalysisService):
    code = """\
def helper():
    pass

def main():
    helper()
"""
    result = await service.analyze(code, "python")

    calls = [r for r in result["relationships"] if r["type"] == "calls"]
    assert any(r["source"] == "main" and r["target"] == "helper" for r in calls)


@pytest.mark.asyncio
async def test_analyze_metrics(service: SemanticAnalysisService):
    code = """\
def a():
    pass
def b():
    a()
"""
    result = await service.analyze(code, "python")

    assert result["metrics"]["total_entities"] == len(result["entities"])
    assert result["metrics"]["total_relationships"] == len(result["relationships"])
    assert result["metrics"]["parse_time_ms"] >= 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_empty_code(service: SemanticAnalysisService):
    result = await service.analyze("", "python")

    assert result["entities"] == []
    assert result["relationships"] == []
    assert result["errors"] == []
    assert result["metrics"]["total_entities"] == 0


@pytest.mark.asyncio
async def test_analyze_syntax_error(service: SemanticAnalysisService):
    result = await service.analyze("def broken(:", "python")

    assert len(result["errors"]) > 0
    assert "SyntaxError" in result["errors"][0]
    assert result["entities"] == []
    assert result["relationships"] == []


# ---------------------------------------------------------------------------
# Non-Python languages (generic regex fallback)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_javascript(service: SemanticAnalysisService):
    code = """\
import { foo } from 'bar';

function greet(name) {
    return `Hello, ${name}`;
}

class Animal extends Base {
    constructor() {}
}
"""
    result = await service.analyze(code, "javascript")

    assert result["language"] == "javascript"
    assert result["errors"] == []

    names = {e["name"] for e in result["entities"]}
    assert "greet" in names
    assert "Animal" in names

    inherits = [r for r in result["relationships"] if r["type"] == "inherits"]
    assert any(r["source"] == "Animal" and r["target"] == "Base" for r in inherits)


@pytest.mark.asyncio
async def test_analyze_unsupported_returns_empty(service: SemanticAnalysisService):
    """A language with no recognisable patterns should still return cleanly."""
    result = await service.analyze("-- this is some SQL\nSELECT 1;", "sql")

    assert result["language"] == "sql"
    assert result["errors"] == []
    assert isinstance(result["entities"], list)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


def test_get_semantic_analysis_service_singleton():
    s1 = get_semantic_analysis_service()
    s2 = get_semantic_analysis_service()
    assert s1 is s2
