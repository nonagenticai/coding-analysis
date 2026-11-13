"""Tests for JavaScript-related modules:
- processing.javascript.javascript_parser
- processing.javascript_processor (LegacyJavaScriptProcessor)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from processing.javascript.javascript_parser import JavaScriptParser

# ---------------------------------------------------------------------------
# JavaScriptParser
# ---------------------------------------------------------------------------


class TestJavaScriptParserInit:
    def test_init_node_available(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(
                returncode=0, stdout="v20.0.0", stderr=""
            )
            JavaScriptParser()
            assert mock_runner.run_command.called

    def test_init_node_not_available(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(
                returncode=1, stdout="", stderr="not found"
            )
            JavaScriptParser()  # Should not raise

    def test_init_subprocess_error(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.side_effect = subprocess.SubprocessError("subprocess failed")
            JavaScriptParser()  # Should not raise

    def test_init_os_error(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.side_effect = OSError("no node")
            JavaScriptParser()  # Should not raise


class TestJavaScriptParserParseCode:
    @pytest.mark.asyncio
    async def test_parse_code_success(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            # Init call
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            # Parse call
            ast_result = {"type": "Program", "body": []}
            mock_runner.run_command.return_value = MagicMock(
                returncode=0, stdout=json.dumps(ast_result), stderr=""
            )

            result = await parser.parse_code("const x = 1;", "test.js")
            assert result is not None
            assert result["type"] == "Program"

    @pytest.mark.asyncio
    async def test_parse_code_failure(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            mock_runner.run_command.return_value = MagicMock(
                returncode=1, stdout="", stderr='{"error": "parse error"}'
            )

            result = await parser.parse_code("invalid {{{", "test.js")
            assert result is None

    @pytest.mark.asyncio
    async def test_parse_code_json_decode_error(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            mock_runner.run_command.return_value = MagicMock(
                returncode=0, stdout="not json", stderr=""
            )

            result = await parser.parse_code("const x = 1;", "test.js")
            assert result is None

    @pytest.mark.asyncio
    async def test_parse_code_os_error(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            mock_runner.run_command.side_effect = OSError("file system error")

            result = await parser.parse_code("const x = 1;", "test.js")
            assert result is None

    @pytest.mark.asyncio
    async def test_parse_code_typescript(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            ast_result = {"type": "Program", "body": []}
            mock_runner.run_command.return_value = MagicMock(
                returncode=0, stdout=json.dumps(ast_result), stderr=""
            )

            result = await parser.parse_code(
                "const x: number = 1;", "test.ts", {"is_typescript": True}
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_parse_code_returns_none_for_non_dict(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            # Return a list instead of dict
            mock_runner.run_command.return_value = MagicMock(
                returncode=0, stdout=json.dumps([1, 2, 3]), stderr=""
            )

            result = await parser.parse_code("const x = 1;", "test.js")
            assert result is None


class TestJavaScriptParserHelpers:
    def test_generate_parser_script_js(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            script = parser._generate_parser_script("/tmp/code.js", "test.js", is_typescript=False)
            assert "parse" in script
            assert "ecmaVersion" in script
            assert "project:" not in script  # No TS project config

    def test_generate_parser_script_tsx(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            script = parser._generate_parser_script("/tmp/code.tsx", "test.tsx", is_typescript=True)
            assert "jsx: true" in script
            assert "project:" in script

    def test_handle_parse_error_json(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            result = MagicMock()
            result.stderr = json.dumps({"error": "unexpected token"})
            # Should not raise
            parser._handle_parse_error(result, "test.js")

    def test_handle_parse_error_plain_text(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            result = MagicMock()
            result.stderr = "plain error text"
            # Should not raise
            parser._handle_parse_error(result, "test.js")

    def test_cleanup_temp_files(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            with patch("processing.javascript.javascript_parser.os.unlink") as mock_unlink:
                parser._cleanup_temp_files("/tmp/script.js", "/tmp/code.js")
                assert mock_unlink.call_count == 2

    def test_cleanup_temp_files_none(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            with patch("processing.javascript.javascript_parser.os.unlink") as mock_unlink:
                parser._cleanup_temp_files(None, None)
                assert mock_unlink.call_count == 0

    def test_cleanup_temp_files_oserror(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            with patch("processing.javascript.javascript_parser.os.unlink", side_effect=OSError):
                # Should not raise
                parser._cleanup_temp_files("/tmp/a.js", "/tmp/b.js")


class TestJavaScriptParserValidateSyntax:
    @pytest.mark.asyncio
    async def test_valid_syntax(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            ast_result = {"type": "Program", "body": []}
            mock_runner.run_command.return_value = MagicMock(
                returncode=0, stdout=json.dumps(ast_result), stderr=""
            )

            result = await parser.validate_syntax("const x = 1;", "test.js")
            assert result is True

    @pytest.mark.asyncio
    async def test_invalid_syntax(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            parser = JavaScriptParser()

            mock_runner.run_command.return_value = MagicMock(
                returncode=1, stdout="", stderr='{"error": "unexpected"}'
            )

            result = await parser.validate_syntax("invalid {{{", "test.js")
            assert result is False


# ---------------------------------------------------------------------------
# LegacyJavaScriptProcessor
# ---------------------------------------------------------------------------
from processing.javascript_processor import LegacyJavaScriptProcessor


class TestLegacyJavaScriptProcessor:
    def test_init(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            fs = MagicMock()
            processor = LegacyJavaScriptProcessor(file_service=fs)
            assert processor.language == "javascript"

    def test_init_no_file_service_raises(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with pytest.raises(ValueError, match="FileService is required"):
                LegacyJavaScriptProcessor(file_service=None)

    def test_determine_language_js(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            fs = MagicMock()
            processor = LegacyJavaScriptProcessor(file_service=fs)
            assert processor._determine_language("test.js") == "javascript"
            assert processor._determine_language("test.jsx") == "javascript"

    def test_determine_language_ts(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            fs = MagicMock()
            processor = LegacyJavaScriptProcessor(file_service=fs)
            assert processor._determine_language("test.ts") == "typescript"
            assert processor._determine_language("test.tsx") == "typescript"

    def test_create_error_result(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            fs = MagicMock()
            processor = LegacyJavaScriptProcessor(file_service=fs)
            result = processor._create_error_result("test.js", "something failed")
            assert result["error"] == "something failed"
            assert result["entities"] == []
            assert result["relationships"] == []

    def test_get_supported_extensions(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            fs = MagicMock()
            processor = LegacyJavaScriptProcessor(file_service=fs)
            exts = processor.get_supported_extensions()
            assert ".js" in exts
            assert ".tsx" in exts

    @pytest.mark.asyncio
    async def test_process_document_reads_file(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            fs = MagicMock()
            fs.read_file.return_value = "const x = 1;"
            processor = LegacyJavaScriptProcessor(file_service=fs)

            # Mock the parser to return None (parse failure)
            processor.parser.parse_code = AsyncMock(return_value=None)

            result = await processor.process_document("test.js")
            assert result["error"] == "Failed to parse file"

    @pytest.mark.asyncio
    async def test_process_document_with_code(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            fs = MagicMock()
            processor = LegacyJavaScriptProcessor(file_service=fs)

            ast = {"type": "Program", "body": []}
            processor.parser.parse_code = AsyncMock(return_value=ast)
            processor.entity_extractor.extract_entities = AsyncMock(return_value=[])
            processor.relation_extractor.extract_all_relations = AsyncMock(return_value=[])

            result = await processor.process_document("test.js", code="const x = 1;")
            assert result["file_path"] == "test.js"
            assert result["entities"] == []

    @pytest.mark.asyncio
    async def test_process_document_exception(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            fs = MagicMock()
            fs.read_file.side_effect = OSError("file not found")
            processor = LegacyJavaScriptProcessor(file_service=fs)

            result = await processor.process_document("missing.js")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_analyze_directory_invalid(self):
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            fs = MagicMock()
            processor = LegacyJavaScriptProcessor(file_service=fs)

            result = await processor.analyze_directory("/nonexistent/path")
            assert result["entities"] == []

    @pytest.mark.asyncio
    async def test_process_document_no_file_service(self):
        """Process document when file_service read is needed (line 62)."""
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            fs = MagicMock()
            fs.read_file.return_value = "function test() { return 1; }"
            processor = LegacyJavaScriptProcessor(file_service=fs)
            processor.parser.parse_code = AsyncMock(return_value={"type": "Program", "body": []})
            processor.entity_extractor.extract_entities = AsyncMock(return_value=[])
            processor.relation_extractor.extract_all_relations = AsyncMock(return_value=[])

            result = await processor.process_document("test.js")
            assert result["file_path"] == "test.js"
            assert result["language"] == "javascript"
            fs.read_file.assert_called_once_with("test.js")

    @pytest.mark.asyncio
    async def test_analyze_directory_valid(self, tmp_path):
        """Analyze directory with JS files (lines 151-187)."""
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            fs = MagicMock()
            processor = LegacyJavaScriptProcessor(file_service=fs)

            # Create test JS file
            js_file = tmp_path / "app.js"
            js_file.write_text("const x = 1;")

            processor.parser.parse_code = AsyncMock(return_value={"type": "Program", "body": []})
            processor.entity_extractor.extract_entities = AsyncMock(return_value=[])
            processor.relation_extractor.extract_all_relations = AsyncMock(return_value=[])

            result = await processor.analyze_directory(str(tmp_path))
            assert "entities" in result
            assert "relationships" in result
            assert result["directory"] == str(tmp_path)

    @pytest.mark.asyncio
    async def test_analyze_directory_non_recursive(self, tmp_path):
        """Non-recursive directory scan (lines 158-159)."""
        with patch("processing.javascript.javascript_parser.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            fs = MagicMock()
            processor = LegacyJavaScriptProcessor(file_service=fs)

            # Create JS files at different depths
            (tmp_path / "root.js").write_text("const x = 1;")
            sub = tmp_path / "sub"
            sub.mkdir()
            (sub / "deep.js").write_text("const y = 2;")

            processor.parser.parse_code = AsyncMock(return_value={"type": "Program", "body": []})
            processor.entity_extractor.extract_entities = AsyncMock(return_value=[])
            processor.relation_extractor.extract_all_relations = AsyncMock(return_value=[])

            result = await processor.analyze_directory(str(tmp_path), recursive=False)
            assert "entities" in result


# ---------------------------------------------------------------------------
# JSAnalyzerProcessor tests (processing.analyzers.javascript_analyzer)
# Covers lines: 119-121, 174, 223, 247-249, 280-281, 300-303, 325-327,
# 365-367, 373-374
# ---------------------------------------------------------------------------

from processing.analyzers.javascript_analyzer import JSAnalyzerProcessor


class TestJSAnalyzerProcessor:
    def test_init(self):
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="v20", stderr="")
            processor = JSAnalyzerProcessor()
            assert processor.language == "javascript"

    def test_init_node_not_found(self):
        """Node.js not found warning (lines 118-119)."""
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(
                returncode=1, stdout="", stderr="not found"
            )
            processor = JSAnalyzerProcessor()
            assert processor.language == "javascript"

    def test_init_node_check_error(self):
        """Error checking Node.js (lines 120-121)."""
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.side_effect = OSError("no node")
            processor = JSAnalyzerProcessor()
            assert processor.language == "javascript"

    def test_get_supported_extensions(self):
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            exts = processor.get_supported_extensions()
            assert ".js" in exts
            assert ".ts" in exts
            assert ".mjs" in exts

    def test_get_language_name(self):
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            assert processor.get_language_name() == "javascript"

    @pytest.mark.asyncio
    async def test_analyze_file_success(self, tmp_path):
        """Analyze file reads and delegates (line 174)."""
        js_file = tmp_path / "test.js"
        js_file.write_text("function hello() { return 1; }")

        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            result = await processor.analyze_file(str(js_file))
            assert result.language in ("javascript", "typescript")
            assert result.file_path == str(js_file)

    @pytest.mark.asyncio
    async def test_analyze_file_error(self):
        """File read error (lines 180-182)."""
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            result = await processor.analyze_file("/nonexistent/file.js")
            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_analyze_snippet_javascript(self):
        """Analyze JavaScript snippet (line 223)."""
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            result = await processor.analyze_snippet("const x = 1;")
            assert result.language == "javascript"

    @pytest.mark.asyncio
    async def test_analyze_snippet_typescript(self):
        """Analyze TypeScript snippet (lines 217-218)."""
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            result = await processor.analyze_snippet("const x: number = 1;", "test.ts")
            assert result.language == "typescript"

    @pytest.mark.asyncio
    async def test_analyze_snippet_error(self):
        """Snippet analysis error (lines 247-249)."""
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            # Patch _parse_code to raise
            with patch.object(processor, "_parse_code", side_effect=OSError("disk error")):
                result = await processor.analyze_snippet("const x = 1;")
                assert result.success is False

    @pytest.mark.asyncio
    async def test_analyze_snippet_parse_failure(self):
        """Parse returns None (line 223)."""
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            with patch.object(processor, "_parse_code", return_value=None):
                result = await processor.analyze_snippet("invalid code")
                assert result.success is False

    @pytest.mark.asyncio
    async def test_analyze_directory_with_files(self, tmp_path):
        """Analyze directory with JS files (lines 276-323)."""
        js_file = tmp_path / "app.js"
        js_file.write_text("function foo() {}")

        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            result = await processor.analyze_directory(str(tmp_path))
            assert result.success is True
            assert result.metadata["directory"] == str(tmp_path)

    @pytest.mark.asyncio
    async def test_analyze_directory_with_file_errors(self, tmp_path):
        """Directory with files that fail analysis (lines 300-303)."""
        js_file = tmp_path / "bad.js"
        js_file.write_text("///")

        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            # Make analyze_file return failed result
            with patch.object(
                processor,
                "analyze_file",
                return_value=MagicMock(success=False, entities=[], relationships=[]),
            ):
                result = await processor.analyze_directory(str(tmp_path))
                assert "failed_files" in result.metadata

    @pytest.mark.asyncio
    async def test_analyze_directory_os_error(self):
        """Directory analysis OS error (lines 325-327)."""
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            with patch("os.walk", side_effect=OSError("permission denied")):
                result = await processor.analyze_directory("/locked/dir")
                assert result.success is False

    def test_regex_parse(self):
        """Test regex-based parsing (lines 376-456)."""
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            code = (
                "function hello() {}\n"
                "class MyClass {}\n"
                "const x = 1;\n"
                "let y = 2;\n"
                "var z = 3;\n"
                "import foo from 'bar';\n"
            )
            result = processor._regex_parse(code, "test.js")
            assert result["type"] == "Program"
            body = result["body"]
            types = [item["type"] for item in body]
            assert "function" in types
            assert "class" in types
            assert "variable" in types
            assert "import" in types

    def test_extract_entities(self):
        """Test entity extraction from AST (lines 458-500)."""
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            ast = {
                "type": "Program",
                "body": [
                    {"type": "function", "name": "hello", "line": 1},
                    {"type": "class", "name": "Foo", "line": 5},
                    {"type": "variable", "name": "x", "line": 10, "var_type": "const"},
                    {"type": "import", "module": "bar", "line": 15},
                ],
            }
            entities = processor._extract_entities(ast, "test.js")
            # import type is excluded from entities
            assert len(entities) == 3
            names = [e.name for e in entities]
            assert "hello" in names
            assert "Foo" in names

    def test_extract_relationships(self):
        """Test relationship extraction (lines 502-519)."""
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            ast = {
                "type": "Program",
                "body": [
                    {"type": "import", "module": "react", "line": 1},
                    {"type": "import", "module": "lodash", "line": 2},
                    {"type": "function", "name": "foo", "line": 3},
                ],
            }
            relations = processor._extract_relationships(ast, "test.js")
            assert len(relations) == 2
            assert relations[0].target_id == "react"
            assert relations[1].target_id == "lodash"

    def test_error_result(self):
        """Test error result creation (lines 521-539)."""
        with patch("processing.analyzers.javascript_analyzer.secure_runner") as mock_runner:
            mock_runner.run_command.return_value = MagicMock(returncode=0, stdout="", stderr="")
            processor = JSAnalyzerProcessor()
            result = processor._error_result("test.js", "some error")
            assert result.success is False
            assert result.error == "some error"
            assert result.file_path == "test.js"


# ---------------------------------------------------------------------------
# JSLanguageProcessor deeper coverage (javascript_processor.py)
# Covers lines: 37-120 (fallback class), 129-137, 155-159, 167, 196-200, 369-370
# ---------------------------------------------------------------------------

from processing.languages.javascript.javascript_processor import (
    LEGACY_COMPONENTS_AVAILABLE,
    JSLanguageProcessor,
)


class TestJSLanguageProcessorDeep:
    """Deeper coverage tests for JSLanguageProcessor."""

    @pytest.fixture
    def js_proc(self):
        return JSLanguageProcessor()

    @pytest.mark.asyncio
    async def test_parse_file_falls_back_to_regex(self, js_proc):
        """parse_file falls through to regex when legacy fails (lines 196-200)."""
        code = "const x = 1;\n"
        # Force legacy parser failure
        js_proc._use_legacy = True
        js_proc.legacy_parser = MagicMock()
        js_proc.legacy_parser.parse_code = AsyncMock(side_effect=ValueError("legacy fail"))

        result = await js_proc.parse_file(Path("test.js"), code)
        assert result.success is True
        assert result.parser_used.value == "regex_fallback"

    @pytest.mark.asyncio
    async def test_parse_file_typescript_uses_regex(self, js_proc):
        """TypeScript files skip legacy parser (line 188)."""
        code = "const x: number = 1;\n"
        result = await js_proc.parse_file(Path("test.ts"), code)
        assert result.success is True
        assert result.language in ("typescript", "javascript")

    @pytest.mark.asyncio
    async def test_extract_entities_fallback_regex(self, js_proc, tmp_path):
        """extract_entities falls back to regex when not tree-sitter AST (lines 369-370)."""
        js_file = tmp_path / "test.js"
        js_file.write_text('import foo from "bar";\n')
        entities = await js_proc.extract_entities(None, js_file)
        assert isinstance(entities, list)

    @pytest.mark.asyncio
    async def test_extract_entities_missing_file(self, js_proc, tmp_path):
        """extract_entities returns [] for missing files (line 375-376)."""
        missing = tmp_path / "nonexistent.js"
        entities = await js_proc.extract_entities(None, missing)
        assert entities == []

    def test_legacy_components_availability(self):
        """Verify LEGACY_COMPONENTS_AVAILABLE is set."""
        assert isinstance(LEGACY_COMPONENTS_AVAILABLE, bool)

    @pytest.mark.asyncio
    async def test_parse_file_no_legacy_no_tree_sitter(self, js_proc):
        """parse_file when both legacy and tree-sitter unavailable (line 202-203)."""
        js_proc._use_legacy = False
        js_proc.parser = None  # Ensure tree-sitter parser is None

        code = 'const express = require("express");\n'
        result = await js_proc.parse_file(Path("app.js"), code)
        assert result.success is True
        assert result.parser_used.value == "regex_fallback"

    @pytest.mark.asyncio
    async def test_regex_parse_complexity_metrics(self, js_proc):
        """Regex parse computes complexity metrics correctly (lines 307-327)."""
        code = "// comment\nfunction foo() {}\n\nconst bar = 1;\n/* block */\n"
        result = await js_proc._regex_parse(Path("test.js"), code, "javascript")
        assert result.lines_of_code == 5
        assert result.success is True

    @pytest.mark.asyncio
    async def test_init_impl_tree_sitter_available(self, js_proc):
        """_initialize_impl when tree-sitter available (line 163-164)."""
        # This exercises the conditional initialization path
        await js_proc._initialize_impl()

    def test_init_with_legacy_failure(self):
        """Init when legacy parser import fails (lines 155-159)."""
        with patch(
            "processing.languages.javascript.javascript_processor.JavaScriptParser",
            side_effect=TypeError("init fail"),
        ):
            proc = JSLanguageProcessor()
            assert proc._use_legacy is False
