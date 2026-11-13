"""Deep tests for processing.quality_analysis.tool_result_processing module."""

from __future__ import annotations

import json

from processing.quality_analysis.tool_definitions import QualityTool, ToolCategory
from processing.quality_analysis.tool_result_processing import ToolResultProcessor


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


class TestToolResultProcessorInit:
    def test_supported_formats(self):
        processor = ToolResultProcessor()
        assert "json" in processor.supported_formats
        assert "xml" in processor.supported_formats
        assert "text" in processor.supported_formats


class TestProcessToolResult:
    def test_json_output(self):
        processor = ToolResultProcessor()
        tool = _make_tool(output_format="json")
        raw = json.dumps(
            [
                {
                    "path": "a.py",
                    "line": 1,
                    "column": 0,
                    "type": "error",
                    "message": "bad",
                    "message-id": "E001",
                }
            ]
        )
        result = processor.process_tool_result(tool, raw, exit_code=0)
        assert result["tool_name"] == "ruff"
        assert result["success"] is True
        assert isinstance(result["processed_output"], list)

    def test_xml_output(self):
        processor = ToolResultProcessor()
        tool = _make_tool(output_format="xml")
        raw = "<root><item key='val'>text</item></root>"
        result = processor.process_tool_result(tool, raw, exit_code=0)
        assert result["processed_output"] is not None

    def test_text_output(self):
        processor = ToolResultProcessor()
        tool = _make_tool(output_format="text")
        raw = "line1\nline2\nline3"
        result = processor.process_tool_result(tool, raw, exit_code=0)
        assert result["processed_output"] == ["line1", "line2", "line3"]

    def test_txt_output(self):
        processor = ToolResultProcessor()
        tool = _make_tool(output_format="txt")
        raw = "line1\nline2"
        result = processor.process_tool_result(tool, raw, exit_code=0)
        assert result["processed_output"] == ["line1", "line2"]

    def test_unknown_format_returns_raw(self):
        processor = ToolResultProcessor()
        tool = _make_tool(output_format="custom")
        raw = "some output"
        result = processor.process_tool_result(tool, raw, exit_code=0)
        assert result["processed_output"] == "some output"

    def test_nonzero_exit_code(self):
        processor = ToolResultProcessor()
        tool = _make_tool(output_format="json")
        raw = "[]"
        result = processor.process_tool_result(tool, raw, exit_code=1)
        assert result["success"] is False
        assert result["exit_code"] == 1

    def test_processing_error_captures_error(self):
        processor = ToolResultProcessor()
        tool = _make_tool(output_format="json")
        # Invalid JSON
        raw = "not json"
        result = processor.process_tool_result(tool, raw, exit_code=0)
        # JSON parse will fail gracefully (returns parse_error dict), not raise
        assert result["processed_output"] is not None

    def test_summary_generated(self):
        processor = ToolResultProcessor()
        tool = _make_tool(output_format="json")
        raw = json.dumps(
            [
                {
                    "path": "a.py",
                    "line": 1,
                    "column": 0,
                    "type": "error",
                    "message": "bad",
                    "message-id": "E001",
                }
            ]
        )
        result = processor.process_tool_result(tool, raw, exit_code=0)
        assert "summary" in result
        assert result["summary"]["tool_name"] == "ruff"


class TestProcessJsonOutput:
    def test_valid_json(self):
        processor = ToolResultProcessor()
        result = processor._process_json_output('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json(self):
        processor = ToolResultProcessor()
        result = processor._process_json_output("invalid")
        assert "parse_error" in result


class TestProcessXmlOutput:
    def test_valid_xml(self):
        processor = ToolResultProcessor()
        result = processor._process_xml_output("<root><child>text</child></root>")
        assert isinstance(result, dict)
        assert result.get("child") == "text"

    def test_invalid_xml(self):
        processor = ToolResultProcessor()
        result = processor._process_xml_output("not xml")
        assert "parse_error" in result

    def test_xml_with_attributes(self):
        processor = ToolResultProcessor()
        result = processor._process_xml_output('<root attr="val"><child>text</child></root>')
        assert isinstance(result, dict)
        assert result.get("attr") == "val"

    def test_xml_duplicate_children(self):
        processor = ToolResultProcessor()
        result = processor._process_xml_output("<root><item>a</item><item>b</item></root>")
        assert isinstance(result, dict)
        assert isinstance(result.get("item"), list)
        assert len(result["item"]) == 2


class TestProcessTextOutput:
    def test_multiline(self):
        processor = ToolResultProcessor()
        result = processor._process_text_output("a\nb\nc")
        assert result == ["a", "b", "c"]

    def test_empty(self):
        processor = ToolResultProcessor()
        result = processor._process_text_output("")
        assert result == []

    def test_whitespace_only(self):
        processor = ToolResultProcessor()
        result = processor._process_text_output("   \n  ")
        assert isinstance(result, list)


class TestXmlToDict:
    def test_text_only_element(self):
        import xml.etree.ElementTree as ET

        processor = ToolResultProcessor()
        elem = ET.fromstring("<root>hello</root>")
        result = processor._xml_to_dict(elem)
        assert result == "hello"

    def test_mixed_content(self):
        import xml.etree.ElementTree as ET

        processor = ToolResultProcessor()
        elem = ET.fromstring("<root>text<child>inner</child></root>")
        result = processor._xml_to_dict(elem)
        assert isinstance(result, dict)
        assert result.get("text") == "text"
        assert result.get("child") == "inner"


class TestExtractIssues:
    def test_ruff_issues(self):
        processor = ToolResultProcessor()
        tool = _make_tool(name="ruff")
        output = [
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
        issues = processor._extract_issues(tool, output)
        assert len(issues) == 2
        assert issues[0]["file"] == "a.py"
        assert issues[0]["line"] == 10
        assert issues[0]["severity"] == "error"
        assert issues[0]["rule"] == "E001"

    def test_eslint_issues(self):
        processor = ToolResultProcessor()
        tool = _make_tool(name="eslint")
        output = [
            {
                "filePath": "/src/app.js",
                "messages": [
                    {
                        "line": 5,
                        "column": 3,
                        "severity": 2,
                        "message": "unused var",
                        "ruleId": "no-unused-vars",
                    },
                ],
            }
        ]
        issues = processor._extract_issues(tool, output)
        assert len(issues) == 1
        assert issues[0]["file"] == "/src/app.js"
        assert issues[0]["rule"] == "no-unused-vars"

    def test_flake8_issues_empty(self):
        processor = ToolResultProcessor()
        tool = _make_tool(name="ruff")
        output = {}
        issues = processor._extract_issues(tool, output)
        assert issues == []

    def test_unknown_tool(self):
        processor = ToolResultProcessor()
        tool = _make_tool(name="custom_tool")
        issues = processor._extract_issues(tool, "some output")
        assert issues == []


class TestExtractMetrics:
    def test_radon_complexity(self):
        processor = ToolResultProcessor()
        tool = _make_tool(name="radon", category=ToolCategory.COMPLEXITY)
        output = {
            "file1.py": [{"complexity": 5}, {"complexity": 3}],
            "file2.py": [{"complexity": 2}],
        }
        metrics = processor._extract_metrics(tool, output)
        assert metrics["total_complexity"] == 10
        assert metrics["file_count"] == 2
        assert metrics["average_complexity"] == 5.0

    def test_security_metrics(self):
        processor = ToolResultProcessor()
        tool = _make_tool(name="bandit", category=ToolCategory.SECURITY)
        output = {
            "results": [
                {"issue_severity": "HIGH"},
                {"issue_severity": "MEDIUM"},
                {"issue_severity": "LOW"},
                {"issue_severity": "HIGH"},
            ]
        }
        metrics = processor._extract_metrics(tool, output)
        assert metrics["total_issues"] == 4
        assert metrics["high_severity"] == 2
        assert metrics["medium_severity"] == 1
        assert metrics["low_severity"] == 1

    def test_no_metrics_for_linter(self):
        processor = ToolResultProcessor()
        tool = _make_tool(name="ruff", category=ToolCategory.LINTER)
        metrics = processor._extract_metrics(tool, [])
        assert metrics == {}


class TestGenerateSummary:
    def test_with_issues(self):
        processor = ToolResultProcessor()
        result = {
            "tool_name": "ruff",
            "success": True,
            "issues": [
                {"severity": "error"},
                {"severity": "error"},
                {"severity": "warning"},
            ],
            "metrics": {"total": 3},
        }
        summary = processor._generate_summary(result)
        assert summary["tool_name"] == "ruff"
        assert summary["issue_count"] == 3
        assert summary["has_metrics"] is True
        assert summary["severity_breakdown"]["error"] == 2
        assert summary["severity_breakdown"]["warning"] == 1

    def test_without_issues(self):
        processor = ToolResultProcessor()
        result = {
            "tool_name": "ruff",
            "success": True,
            "issues": [],
            "metrics": {},
        }
        summary = processor._generate_summary(result)
        assert summary["issue_count"] == 0
        assert summary["has_metrics"] is False
        assert "severity_breakdown" not in summary


class TestNormalizeResults:
    def test_normalize_multiple_results(self):
        processor = ToolResultProcessor()
        results = [
            {
                "tool_name": "ruff",
                "tool_category": "linter",
                "success": True,
                "issues": [{"severity": "error"}, {"severity": "warning"}],
                "metrics": {},
                "summary": {"tool_name": "ruff"},
            },
            {
                "tool_name": "bandit",
                "tool_category": "security",
                "success": False,
                "issues": [{"severity": "high"}],
                "metrics": {},
                "summary": {"tool_name": "bandit"},
            },
        ]
        normalized = processor.normalize_results(results)
        assert normalized["total_tools"] == 2
        assert normalized["successful_tools"] == 1
        assert normalized["failed_tools"] == 1
        assert normalized["total_issues"] == 3
        assert normalized["issues_by_category"]["linter"] == 2
        assert normalized["issues_by_category"]["security"] == 1
        assert len(normalized["tool_summaries"]) == 2
        assert normalized["overall_score"] > 0

    def test_normalize_empty(self):
        processor = ToolResultProcessor()
        normalized = processor.normalize_results([])
        assert normalized["total_tools"] == 0
        assert normalized["overall_score"] == 0.0


class TestFormatForOutput:
    def test_json_format(self):
        processor = ToolResultProcessor()
        result = {"key": "value"}
        output = processor.format_for_output(result, "json")
        assert json.loads(output) == {"key": "value"}

    def test_text_format(self):
        processor = ToolResultProcessor()
        result = {
            "tool_name": "ruff",
            "tool_category": "linter",
            "success": True,
            "issues": [{"file": "a.py", "line": 1, "message": "bad"}],
            "metrics": {"total": 1},
        }
        output = processor.format_for_output(result, "text")
        assert "ruff" in output
        assert "linter" in output
        assert "bad" in output

    def test_unknown_format(self):
        processor = ToolResultProcessor()
        result = {"data": 42}
        output = processor.format_for_output(result, "custom")
        assert "42" in output

    def test_text_report_more_than_10_issues(self):
        processor = ToolResultProcessor()
        result = {
            "tool_name": "ruff",
            "tool_category": "linter",
            "success": True,
            "issues": [{"file": f"f{i}.py", "line": i, "message": f"issue {i}"} for i in range(15)],
            "metrics": {},
        }
        output = processor._format_text_report(result)
        assert "... and 5 more issues" in output
