"""Unit tests for analysis_utils module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.api.endpoints.analysis_modules.analysis_utils import (
    detect_language_from_code,
    detect_language_from_file_path,
    find_code_files,
    get_extension_for_language,
    populate_graph_database,
    resolve_container_files,
)

# DatabaseLock tests removed: the sqlite per-file lock was dropped in the
# postgres cutover (schema is Alembic-owned, rows scoped by code_graph_id).


# ---------------------------------------------------------------------------
# detect_language_from_code tests
# ---------------------------------------------------------------------------


class TestDetectLanguageFromCode:
    """Tests for detect_language_from_code."""

    def test_detect_python(self):
        assert detect_language_from_code("def hello():") == "python"
        assert detect_language_from_code("import os") == "python"
        assert detect_language_from_code("from pathlib import Path") == "python"
        assert detect_language_from_code("class MyClass:") == "python"
        assert detect_language_from_code('if __name__ == "__main__":') == "python"

    def test_detect_javascript(self):
        assert detect_language_from_code("function hello() {}") == "javascript"
        assert detect_language_from_code("const x = 5;") == "javascript"
        assert detect_language_from_code("let y = 10;") == "javascript"
        assert detect_language_from_code("var z = 15;") == "javascript"
        assert detect_language_from_code("console.log('hi')") == "javascript"

    def test_detect_typescript(self):
        code = "const x = 5;\ninterface Foo {}"
        assert detect_language_from_code(code) == "typescript"
        code2 = "let name: string = 'hello';"
        assert detect_language_from_code(code2) == "typescript"
        code3 = "const count: number = 42;"
        assert detect_language_from_code(code3) == "typescript"

    def test_detect_java(self):
        # Note: "public class" also contains "class " which triggers python first,
        # so test with unambiguous Java indicators.
        assert detect_language_from_code("public static void main(String[] args)") == "java"
        assert detect_language_from_code("System.out.println('hello')") == "java"
        # "private " is a Java indicator that doesn't overlap with Python
        assert detect_language_from_code("private int x = 5;") == "java"

    def test_detect_c(self):
        assert detect_language_from_code("#include <stdio.h>\nint main() {}") == "c"
        assert detect_language_from_code("printf('hello')") == "c"

    def test_detect_cpp(self):
        assert detect_language_from_code("#include <iostream>\nstd::cout") == "cpp"
        # "class " overlaps with Python detection, so use std:: without "class "
        assert detect_language_from_code("#include <vector>\nstd::vector<int> v;") == "cpp"

    def test_detect_go(self):
        # "package " and "func " are Go indicators without Python overlap
        assert detect_language_from_code("package main") == "go"
        assert detect_language_from_code("func main() {}") == "go"
        assert detect_language_from_code('fmt.Println("hello")') == "go"

    def test_detect_rust(self):
        # "fn " is a Rust indicator without overlap with earlier languages
        assert detect_language_from_code("fn main() {}") == "rust"
        assert detect_language_from_code('println!("hello")') == "rust"
        # "struct " is a Rust indicator without JS/Python/Java/C overlap
        assert detect_language_from_code("struct Foo { x: i32 }") == "rust"

    def test_unknown_returns_none(self):
        assert detect_language_from_code("hello world") is None
        assert detect_language_from_code("") is None
        assert detect_language_from_code("12345") is None


# ---------------------------------------------------------------------------
# get_extension_for_language tests
# ---------------------------------------------------------------------------


class TestGetExtensionForLanguage:
    """Tests for get_extension_for_language."""

    def test_python(self):
        assert get_extension_for_language("python") == ".py"

    def test_javascript(self):
        assert get_extension_for_language("javascript") == ".js"

    def test_typescript(self):
        assert get_extension_for_language("typescript") == ".ts"

    def test_java(self):
        assert get_extension_for_language("java") == ".java"

    def test_cpp(self):
        assert get_extension_for_language("cpp") == ".cpp"

    def test_c(self):
        assert get_extension_for_language("c") == ".c"

    def test_go(self):
        assert get_extension_for_language("go") == ".go"

    def test_rust(self):
        assert get_extension_for_language("rust") == ".rs"

    def test_csharp(self):
        assert get_extension_for_language("csharp") == ".cs"

    def test_php(self):
        assert get_extension_for_language("php") == ".php"

    def test_ruby(self):
        assert get_extension_for_language("ruby") == ".rb"

    def test_swift(self):
        assert get_extension_for_language("swift") == ".swift"

    def test_kotlin(self):
        assert get_extension_for_language("kotlin") == ".kt"

    def test_scala(self):
        assert get_extension_for_language("scala") == ".scala"

    def test_dart(self):
        assert get_extension_for_language("dart") == ".dart"

    def test_r(self):
        assert get_extension_for_language("r") == ".r"

    def test_lua(self):
        assert get_extension_for_language("lua") == ".lua"

    def test_perl(self):
        assert get_extension_for_language("perl") == ".pl"

    def test_shell(self):
        assert get_extension_for_language("shell") == ".sh"

    def test_powershell(self):
        assert get_extension_for_language("powershell") == ".ps1"

    def test_unknown_language(self):
        assert get_extension_for_language("brainfuck") == ".txt"

    def test_case_insensitive(self):
        assert get_extension_for_language("Python") == ".py"
        assert get_extension_for_language("JAVA") == ".java"
        assert get_extension_for_language("GoLang") == ".txt"  # "golang" not in map


# ---------------------------------------------------------------------------
# detect_language_from_file_path tests
# ---------------------------------------------------------------------------


class TestDetectLanguageFromFilePath:
    """Tests for detect_language_from_file_path."""

    def test_python(self):
        assert detect_language_from_file_path("main.py") == "python"

    def test_javascript(self):
        assert detect_language_from_file_path("app.js") == "javascript"
        assert detect_language_from_file_path("component.jsx") == "javascript"

    def test_typescript(self):
        assert detect_language_from_file_path("app.ts") == "typescript"
        assert detect_language_from_file_path("component.tsx") == "typescript"

    def test_java(self):
        assert detect_language_from_file_path("Main.java") == "java"

    def test_kotlin(self):
        assert detect_language_from_file_path("Main.kt") == "kotlin"

    def test_scala(self):
        assert detect_language_from_file_path("Main.scala") == "scala"

    def test_cpp(self):
        assert detect_language_from_file_path("main.cpp") == "cpp"
        assert detect_language_from_file_path("main.cc") == "cpp"
        assert detect_language_from_file_path("main.cxx") == "cpp"
        assert detect_language_from_file_path("header.hpp") == "cpp"

    def test_c(self):
        assert detect_language_from_file_path("main.c") == "c"
        assert detect_language_from_file_path("header.h") == "c"

    def test_csharp(self):
        assert detect_language_from_file_path("Program.cs") == "csharp"

    def test_go(self):
        assert detect_language_from_file_path("main.go") == "go"

    def test_rust(self):
        assert detect_language_from_file_path("main.rs") == "rust"

    def test_ruby(self):
        assert detect_language_from_file_path("app.rb") == "ruby"

    def test_php(self):
        assert detect_language_from_file_path("index.php") == "php"

    def test_swift(self):
        assert detect_language_from_file_path("main.swift") == "swift"

    def test_objectivec(self):
        assert detect_language_from_file_path("main.m") == "objectivec"

    def test_r(self):
        assert detect_language_from_file_path("script.r") == "r"

    def test_perl(self):
        assert detect_language_from_file_path("script.pl") == "perl"

    def test_shell(self):
        assert detect_language_from_file_path("run.sh") == "shell"
        assert detect_language_from_file_path("run.bash") == "shell"

    def test_powershell(self):
        assert detect_language_from_file_path("run.ps1") == "powershell"

    def test_lua(self):
        assert detect_language_from_file_path("script.lua") == "lua"

    def test_dart(self):
        assert detect_language_from_file_path("main.dart") == "dart"

    def test_elm(self):
        assert detect_language_from_file_path("Main.elm") == "elm"

    def test_elixir(self):
        assert detect_language_from_file_path("app.ex") == "elixir"

    def test_erlang(self):
        assert detect_language_from_file_path("app.erl") == "erlang"

    def test_clojure(self):
        assert detect_language_from_file_path("core.clj") == "clojure"

    def test_haskell(self):
        assert detect_language_from_file_path("Main.hs") == "haskell"

    def test_ocaml(self):
        assert detect_language_from_file_path("main.ml") == "ocaml"

    def test_vim(self):
        assert detect_language_from_file_path("plugin.vim") == "vim"

    def test_sql(self):
        assert detect_language_from_file_path("query.sql") == "sql"

    def test_julia(self):
        assert detect_language_from_file_path("script.jl") == "julia"

    def test_nim(self):
        assert detect_language_from_file_path("app.nim") == "nim"

    def test_crystal(self):
        assert detect_language_from_file_path("app.cr") == "crystal"

    def test_d(self):
        assert detect_language_from_file_path("app.d") == "d"

    def test_pascal(self):
        assert detect_language_from_file_path("app.pas") == "pascal"

    def test_ada(self):
        assert detect_language_from_file_path("app.ada") == "ada"

    def test_fortran(self):
        assert detect_language_from_file_path("sim.f90") == "fortran"

    def test_cobol(self):
        assert detect_language_from_file_path("prog.cob") == "cobol"

    def test_groovy(self):
        assert detect_language_from_file_path("build.groovy") == "groovy"

    def test_unknown_extension(self):
        assert detect_language_from_file_path("file.xyz") is None
        assert detect_language_from_file_path("Makefile") is None

    def test_full_path(self):
        assert detect_language_from_file_path("/home/user/project/src/main.py") == "python"

    def test_case_insensitive_extension(self):
        # .PY → suffix.lower() == ".py"
        assert detect_language_from_file_path("MAIN.PY") == "python"


# ---------------------------------------------------------------------------
# find_code_files tests
# ---------------------------------------------------------------------------


class TestFindCodeFiles:
    """Use project/ subdir to avoid find_code_files skipping /tmp paths on Linux CI."""

    @pytest.fixture
    def project_dir(self, tmp_path):
        d = tmp_path / "project"
        d.mkdir()
        return d

    def test_finds_python_files(self, project_dir):
        (project_dir / "main.py").write_text("print('hello')", encoding="utf-8")
        (project_dir / "utils.py").write_text("x = 1", encoding="utf-8")
        result = find_code_files(project_dir)
        assert len(result) == 2
        names = {f.name for f in result}
        assert names == {"main.py", "utils.py"}

    def test_finds_multiple_languages(self, project_dir):
        (project_dir / "app.py").write_text("# python", encoding="utf-8")
        (project_dir / "app.js").write_text("// js", encoding="utf-8")
        (project_dir / "app.go").write_text("// go", encoding="utf-8")
        (project_dir / "app.rs").write_text("// rust", encoding="utf-8")
        result = find_code_files(project_dir)
        assert len(result) == 4

    def test_skips_git_directory(self, project_dir):
        git_dir = project_dir / ".git"
        git_dir.mkdir()
        (git_dir / "config.py").write_text("# config", encoding="utf-8")
        (project_dir / "main.py").write_text("# main", encoding="utf-8")
        result = find_code_files(project_dir)
        assert len(result) == 1
        assert result[0].name == "main.py"

    def test_skips_node_modules(self, project_dir):
        nm = project_dir / "node_modules"
        nm.mkdir()
        (nm / "index.js").write_text("// module", encoding="utf-8")
        (project_dir / "app.js").write_text("// app", encoding="utf-8")
        result = find_code_files(project_dir)
        assert len(result) == 1
        assert result[0].name == "app.js"

    def test_skips_pycache(self, project_dir):
        cache = project_dir / "__pycache__"
        cache.mkdir()
        (cache / "module.py").write_text("cached", encoding="utf-8")
        (project_dir / "main.py").write_text("main", encoding="utf-8")
        result = find_code_files(project_dir)
        assert len(result) == 1

    def test_skips_venv(self, project_dir):
        venv = project_dir / "venv"
        venv.mkdir()
        (venv / "lib.py").write_text("venv lib", encoding="utf-8")
        (project_dir / "main.py").write_text("main", encoding="utf-8")
        result = find_code_files(project_dir)
        assert len(result) == 1

    def test_skips_build_dirs(self, project_dir):
        for d in ["build", "dist", "target", "bin", "obj"]:
            bd = project_dir / d
            bd.mkdir()
            (bd / "output.py").write_text("output", encoding="utf-8")
        (project_dir / "src.py").write_text("source", encoding="utf-8")
        result = find_code_files(project_dir)
        assert len(result) == 1

    def test_ignores_non_code_files(self, project_dir):
        (project_dir / "readme.md").write_text("# Readme", encoding="utf-8")
        (project_dir / "data.json").write_text("{}", encoding="utf-8")
        (project_dir / "image.png").write_bytes(b"\x89PNG")
        (project_dir / "main.py").write_text("# code", encoding="utf-8")
        result = find_code_files(project_dir)
        assert len(result) == 1
        assert result[0].name == "main.py"

    def test_returns_sorted_files(self, project_dir):
        (project_dir / "z_last.py").write_text("z", encoding="utf-8")
        (project_dir / "a_first.py").write_text("a", encoding="utf-8")
        (project_dir / "m_mid.py").write_text("m", encoding="utf-8")
        result = find_code_files(project_dir)
        names = [f.name for f in result]
        assert names == sorted(names)

    def test_empty_directory(self, project_dir):
        result = find_code_files(project_dir)
        assert result == []

    def test_recursive_search(self, project_dir):
        sub = project_dir / "src" / "pkg"
        sub.mkdir(parents=True)
        (sub / "mod.py").write_text("# module", encoding="utf-8")
        (project_dir / "main.py").write_text("# main", encoding="utf-8")
        result = find_code_files(project_dir)
        assert len(result) == 2

    def test_skips_large_files(self, project_dir):
        # Create a file > 10MB
        large = project_dir / "big.py"
        large.write_text("x = 1\n" * (10 * 1024 * 1024 // 6 + 1), encoding="utf-8")
        small = project_dir / "small.py"
        small.write_text("y = 2", encoding="utf-8")
        result = find_code_files(project_dir)
        names = [f.name for f in result]
        assert "small.py" in names
        assert "big.py" not in names

    def test_many_extensions(self, project_dir):
        exts = [".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".cs", ".rb", ".php"]
        for ext in exts:
            (project_dir / f"file{ext}").write_text("// code", encoding="utf-8")
        result = find_code_files(project_dir)
        assert len(result) == len(exts)


# ---------------------------------------------------------------------------
# populate_graph_database tests
# ---------------------------------------------------------------------------


class TestPopulateGraphDatabase:
    """Tests for populate_graph_database (postgres pool, scoped by code_graph_id)."""

    _GET_CONN = "app.api.endpoints.analysis_modules.analysis_utils.get_raw_connection"

    @staticmethod
    def _node_inserts(conn):
        return [
            (sql, params)
            for sql, params in conn.executed
            if "INSERT INTO nodes" in " ".join(sql.split())
        ]

    @staticmethod
    def _edge_inserts(conn):
        return [
            (sql, params)
            for sql, params in conn.executed
            if "INSERT INTO edges" in " ".join(sql.split())
        ]

    async def test_populate_with_nodes_and_edges(self, make_pg_conn):
        conn = make_pg_conn()
        analysis_data = {
            "nodes": [
                {"id": "node1", "type": "function", "name": "hello", "properties": {}},
                {"id": "node2", "type": "class", "name": "MyClass", "properties": {}},
            ],
            "edges": [
                {"source_id": "node1", "target_id": "node2", "type": "calls", "properties": {}},
            ],
        }
        with patch(self._GET_CONN, return_value=conn):
            result = await populate_graph_database("test-container", analysis_data)

        assert result is True
        node_inserts = self._node_inserts(conn)
        assert len(node_inserts) == 2
        # every write is scoped to the code_graph_id (== container name)
        for _sql, params in node_inserts:
            assert params[0] == "test-container"
        edge_inserts = self._edge_inserts(conn)
        assert len(edge_inserts) == 1
        assert edge_inserts[0][1][0] == "test-container"
        assert conn.commits >= 1

    async def test_populate_empty_data(self, make_pg_conn):
        conn = make_pg_conn()
        with patch(self._GET_CONN, return_value=conn):
            result = await populate_graph_database("test-container", {"nodes": [], "edges": []})
        assert result is True
        assert self._node_inserts(conn) == []
        assert self._edge_inserts(conn) == []

    async def test_populate_skips_duplicate_nodes(self, make_pg_conn):
        conn = make_pg_conn()
        analysis_data = {
            "nodes": [
                {"id": "node1", "type": "function", "properties": {}},
                {"id": "node1", "type": "function", "properties": {}},  # duplicate
                {"id": "node2", "type": "class", "properties": {}},
            ],
            "edges": [],
        }
        with patch(self._GET_CONN, return_value=conn):
            result = await populate_graph_database("test-container", analysis_data)
        assert result is True
        # the in-memory dedup prevents the duplicate INSERT entirely
        inserted_ids = [params[1] for _sql, params in self._node_inserts(conn)]
        assert inserted_ids == ["node1", "node2"]

    async def test_populate_skips_duplicate_edges(self, make_pg_conn):
        conn = make_pg_conn()
        analysis_data = {
            "nodes": [
                {"id": "n1", "type": "func", "properties": {}},
                {"id": "n2", "type": "func", "properties": {}},
            ],
            "edges": [
                {"source_id": "n1", "target_id": "n2", "type": "calls", "properties": {}},
                {"source_id": "n1", "target_id": "n2", "type": "calls", "properties": {}},  # dup
            ],
        }
        with patch(self._GET_CONN, return_value=conn):
            result = await populate_graph_database("test-container", analysis_data)
        assert result is True
        assert len(self._edge_inserts(conn)) == 1

    async def test_populate_skips_orphaned_edges(self, make_pg_conn):
        conn = make_pg_conn()
        analysis_data = {
            "nodes": [{"id": "n1", "type": "func", "properties": {}}],
            "edges": [
                # references n2 which doesn't exist -> orphaned, never inserted
                {"source_id": "n1", "target_id": "n2", "type": "calls", "properties": {}},
            ],
        }
        with patch(self._GET_CONN, return_value=conn):
            result = await populate_graph_database("test-container", analysis_data)
        assert result is True
        assert self._edge_inserts(conn) == []

    async def test_populate_node_with_name_in_properties(self, make_pg_conn):
        conn = make_pg_conn()
        analysis_data = {
            "nodes": [
                {
                    "id": "n1",
                    "type": "function",
                    "name": "my_func",
                    "file_path": "/src/main.py",
                    "properties": {},
                },
            ],
            "edges": [],
        }
        with patch(self._GET_CONN, return_value=conn):
            result = await populate_graph_database("test-container", analysis_data)
        assert result is True
        node_inserts = self._node_inserts(conn)
        assert len(node_inserts) == 1
        # name/file_path are folded into the JSONB properties payload
        props = node_inserts[0][1][3]
        assert props.obj["name"] == "my_func"
        assert props.obj["file_path"] == "/src/main.py"

    async def test_populate_returns_false_on_error(self):
        with patch(self._GET_CONN, side_effect=RuntimeError("boom")):
            result = await populate_graph_database("test-container", {"nodes": [], "edges": []})
        assert result is False

    async def test_populate_handles_node_missing_id(self, make_pg_conn):
        """Nodes without an 'id' key should be skipped without crashing."""
        conn = make_pg_conn()
        analysis_data = {
            "nodes": [
                {"type": "function", "properties": {}},  # missing id
                {"id": "n1", "type": "class", "properties": {}},
            ],
            "edges": [],
        }
        with patch(self._GET_CONN, return_value=conn):
            result = await populate_graph_database("test-container", analysis_data)
        assert result is True
        inserted_ids = [params[1] for _sql, params in self._node_inserts(conn)]
        assert inserted_ids == ["n1"]

    async def test_populate_handles_edge_missing_keys(self, make_pg_conn):
        """Edges missing required keys should be skipped without crashing."""
        conn = make_pg_conn()
        analysis_data = {
            "nodes": [
                {"id": "n1", "type": "func", "properties": {}},
                {"id": "n2", "type": "func", "properties": {}},
            ],
            "edges": [
                {"type": "calls"},  # missing source_id, target_id
            ],
        }
        with patch(self._GET_CONN, return_value=conn):
            result = await populate_graph_database("test-container", analysis_data)
        assert result is True
        assert self._edge_inserts(conn) == []


# ---------------------------------------------------------------------------
# resolve_container_files tests
# ---------------------------------------------------------------------------


_LOOKUP = "services.docker_service.get_docker_service"
_MATERIALIZE = "app.api.utils.workspace_files.materialize_workspace"


def _lookup_exact(*known: tuple[str, str]):
    """A coding-containers lookup that knows ``(name, tenant)`` pairs EXACTLY."""

    async def _find(name, tenant_id):
        for known_name, owner in known:
            if name == known_name:
                return {"name": known_name, "tenant_id": owner, "codebase_path": "/workspace"}
        return None

    svc = AsyncMock()
    svc.find_container_by_name = AsyncMock(side_effect=_find)
    return svc


class TestResolveContainerFiles:
    """Exact, tenant-scoped lookup + materialised live workspace; no fallbacks.

    Deleted (they pinned the defect, prod 2026-10-11):
    test_container_found_by_partial_name / _by_suffix_match (substring and
    suffix matching answered about a DIFFERENT container),
    test_resolves_absolute_file_paths (read an arbitrary path on THIS pod),
    test_raises_404_when_no_paths_accessible (exercised the /tmp/shared_repos,
    /app/shared_data, /shared, /app/data fallback chain), and the
    list_containers-based not-found/500/tenant tests (lookup is now exact via
    coding-containers).
    """

    async def test_container_not_found_raises_404(self):
        with patch(_LOOKUP, return_value=_lookup_exact()):
            with pytest.raises(HTTPException) as exc_info:
                await resolve_container_files("nonexistent-container", tenant_id="t1")
        assert exc_info.value.status_code == 404

    async def test_prefix_of_a_real_name_does_not_resolve(self):
        """Negative control: 'cv-ems' must NOT resolve to 'cv-ems-p4'."""
        svc = _lookup_exact(("cv-ems-p4", "t1"))
        with patch(_LOOKUP, return_value=svc), patch(_MATERIALIZE, new_callable=AsyncMock) as mat:
            with pytest.raises(HTTPException) as exc_info:
                await resolve_container_files("cv-ems", tenant_id="t1")
        assert exc_info.value.status_code == 404
        svc.find_container_by_name.assert_awaited_once_with("cv-ems", "t1")
        mat.assert_not_awaited()

    async def test_other_tenants_container_is_403(self):
        with patch(_LOOKUP, return_value=_lookup_exact(("cv-ems-p4", "owner"))):
            with pytest.raises(HTTPException) as exc_info:
                await resolve_container_files("cv-ems-p4", tenant_id="intruder")
        assert exc_info.value.status_code == 403

    async def test_exact_name_materialises_live_workspace(self, tmp_path):
        (tmp_path / "main.py").write_text("# code", encoding="utf-8")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "dep.js").write_text("//", encoding="utf-8")
        with (
            patch(_LOOKUP, return_value=_lookup_exact(("cv-ems-p4", "t1"))),
            patch(_MATERIALIZE, new_callable=AsyncMock, return_value=tmp_path) as mat,
        ):
            result = await resolve_container_files("cv-ems-p4", tenant_id="t1")

        assert [p.name for p in result] == ["main.py"]  # skip-dirs still applied
        assert result.materialized_root == tmp_path
        args, kwargs = mat.call_args
        assert args == ("cv-ems-p4",)
        assert kwargs["tenant_id"] == "t1"
        assert kwargs["root"] == "/workspace"
        assert "*.py" in kwargs["name_patterns"]
        result.release()
        assert not tmp_path.exists()

    async def test_resolves_specific_file_paths(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# main", encoding="utf-8")
        (tmp_path / "src" / "utils.py").write_text("# utils", encoding="utf-8")
        with (
            patch(_LOOKUP, return_value=_lookup_exact(("my-container", "t1"))),
            patch(_MATERIALIZE, new_callable=AsyncMock, return_value=tmp_path) as mat,
        ):
            result = await resolve_container_files(
                "my-container", file_paths=["src/main.py", "src/utils.py"], tenant_id="t1"
            )
        assert len(result) == 2
        assert mat.call_args.kwargs["rel_paths"] == ["src/main.py", "src/utils.py"]

    async def test_unreadable_workspace_is_422_never_a_fallback_dir(self):
        from app.api.utils.workspace_files import WorkspaceUnresolvableError

        with (
            patch(_LOOKUP, return_value=_lookup_exact(("my-container", "t1"))),
            patch(
                _MATERIALIZE,
                new_callable=AsyncMock,
                side_effect=WorkspaceUnresolvableError("my-container", "no readable files"),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await resolve_container_files("my-container", tenant_id="t1")
        assert exc_info.value.status_code == 422

    async def test_unconfined_path_is_400(self):
        from app.api.utils.workspace_files import InvalidWorkspacePathError

        with (
            patch(_LOOKUP, return_value=_lookup_exact(("my-container", "t1"))),
            patch(
                _MATERIALIZE,
                new_callable=AsyncMock,
                side_effect=InvalidWorkspacePathError("../../etc/passwd"),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await resolve_container_files(
                    "my-container", file_paths=["../../etc/passwd"], tenant_id="t1"
                )
        assert exc_info.value.status_code == 400
