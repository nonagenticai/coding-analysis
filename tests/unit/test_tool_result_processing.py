"""Unit tests for processing.quality_analysis.tool_result_processing module."""

from __future__ import annotations

import json

from processing.quality_analysis.tool_definitions import QualityTool, ToolCategory
from processing.quality_analysis.tool_result_processing import ToolResultProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(
    name: str = "ruff",
    category: ToolCategory = ToolCategory.LINTER,
    output_format: str = "json",
) -> QualityTool:
    return QualityTool(
        name=name,
        category=category,
        command=[name],
        output_format=output_format,
        description=f"{name} tool",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalizationOfRawOutputs:
    """test_normalization_of_raw_outputs"""

    def test_process_json_output(self):
        processor = ToolResultProcessor()
        data = [{"type": "error", "message": "bad", "line": 1, "column": 0, "message-id": "E001"}]
        result = processor._process_json_output(json.dumps(data))
        assert isinstance(result, list)
        assert len(result) == 1

    def test_process_text_output(self):
        processor = ToolResultProcessor()
        result = processor._process_text_output("line1\nline2\nline3")
        assert result == ["line1", "line2", "line3"]

    def test_process_text_output_empty(self):
        processor = ToolResultProcessor()
        result = processor._process_text_output("")
        assert result == []

    def test_process_tool_result_json(self):
        processor = ToolResultProcessor()
        tool = _make_tool("ruff", output_format="json")
        data = [
            {
                "type": "error",
                "message": "bad",
                "line": 5,
                "column": 0,
                "message-id": "E001",
                "path": "x.py",
            }
        ]
        result = processor.process_tool_result(tool, json.dumps(data), exit_code=0)
        assert result["tool_name"] == "ruff"
        assert result["success"] is True
        assert result["processed_output"] is not None
        assert "issues" in result

    def test_process_tool_result_text(self):
        processor = ToolResultProcessor()
        tool = _make_tool("ty", output_format="text")
        result = processor.process_tool_result(tool, "all good", exit_code=0)
        assert result["success"] is True
        assert isinstance(result["processed_output"], list)

    def test_normalize_results(self):
        processor = ToolResultProcessor()
        results = [
            {
                "tool_name": "ruff",
                "tool_category": "linter",
                "success": True,
                "issues": [{"severity": "error", "message": "bad"}],
                "summary": {
                    "tool_name": "ruff",
                    "success": True,
                    "issue_count": 1,
                    "has_metrics": False,
                },
            },
            {
                "tool_name": "ty",
                "tool_category": "type_checker",
                "success": True,
                "issues": [],
                "summary": {
                    "tool_name": "ty",
                    "success": True,
                    "issue_count": 0,
                    "has_metrics": False,
                },
            },
        ]
        normalized = processor.normalize_results(results)
        assert normalized["total_tools"] == 2
        assert normalized["successful_tools"] == 2
        assert normalized["total_issues"] == 1
        assert "issues_by_category" in normalized
        assert "overall_score" in normalized


class TestMalformedOutput:
    """test_malformed_output"""

    def test_invalid_json_returns_parse_error(self):
        processor = ToolResultProcessor()
        result = processor._process_json_output("{not valid json")
        assert "parse_error" in result

    def test_invalid_xml_returns_parse_error(self):
        processor = ToolResultProcessor()
        result = processor._process_xml_output("<invalid>xml<no-close>")
        assert isinstance(result, dict)
        assert "parse_error" in result

    def test_process_result_with_bad_output_adds_error(self):
        processor = ToolResultProcessor()
        tool = _make_tool("ruff", output_format="json")
        result = processor.process_tool_result(tool, "this is not json", exit_code=1)
        assert result["exit_code"] == 1
        # processed_output should have parse_error
        assert "parse_error" in result["processed_output"]


class TestSeverityMapping:
    """test_severity_mapping"""

    def test_extract_ruff_issues(self):
        processor = ToolResultProcessor()
        tool = _make_tool("ruff")
        data = [
            {
                "filename": "a.py",
                "location": {"row": 10, "column": 5},
                "severity": "error",
                "message": "bad code",
                "code": "E001",
            },
            {
                "filename": "b.py",
                "location": {"row": 20, "column": 1},
                "severity": "warning",
                "message": "not great",
                "code": "W002",
            },
        ]
        issues = processor._extract_issues(tool, data)
        assert len(issues) == 2
        assert issues[0]["severity"] == "error"
        assert issues[1]["severity"] == "warning"
        assert [i["rule"] for i in issues] == ["E001", "W002"]

    def test_extract_ty_issues(self):
        processor = ToolResultProcessor()
        tool = _make_tool("ty")
        data = [
            {
                "check_name": "invalid-return-type",
                "description": "invalid-return-type: nope",
                "severity": "major",
                "location": {"path": "a.py", "positions": {"begin": {"line": 4, "column": 12}}},
            }
        ]
        issues = processor._extract_issues(tool, data)
        assert issues == [
            {
                "file": "a.py",
                "line": 4,
                "column": 12,
                "severity": "major",
                "message": "nope",
                "rule": "invalid-return-type",
            }
        ]

    def test_extract_eslint_issues(self):
        processor = ToolResultProcessor()
        tool = _make_tool("eslint")
        data = [
            {
                "filePath": "app.js",
                "messages": [
                    {"line": 1, "column": 1, "severity": 2, "message": "err", "ruleId": "no-var"},
                ],
            }
        ]
        issues = processor._extract_issues(tool, data)
        assert len(issues) == 1
        assert issues[0]["rule"] == "no-var"

    def test_generate_summary(self):
        processor = ToolResultProcessor()
        result = {
            "tool_name": "test",
            "success": True,
            "issues": [
                {"severity": "error", "message": "a"},
                {"severity": "error", "message": "b"},
                {"severity": "warning", "message": "c"},
            ],
            "metrics": {},
        }
        summary = processor._generate_summary(result)
        assert summary["issue_count"] == 3
        assert summary["severity_breakdown"]["error"] == 2
        assert summary["severity_breakdown"]["warning"] == 1

    def test_format_for_output_json(self):
        processor = ToolResultProcessor()
        result = {"tool_name": "test", "issues": []}
        formatted = processor.format_for_output(result, "json")
        parsed = json.loads(formatted)
        assert parsed["tool_name"] == "test"

    def test_format_for_output_text(self):
        processor = ToolResultProcessor()
        result = {
            "tool_name": "test",
            "tool_category": "linter",
            "success": True,
            "issues": [],
            "metrics": {},
        }
        text = processor.format_for_output(result, "text")
        assert "Tool: test" in text
