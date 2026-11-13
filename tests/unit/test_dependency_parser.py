"""Unit tests for services.dependency_parser."""

from __future__ import annotations

from services.dependency_parser import DependencyParser

# ------------------------------------------------------------------
# parse_requirements
# ------------------------------------------------------------------


class TestParseRequirements:
    def test_basic_packages(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("requests==2.31.0\nflask>=3.0\nnumpy\n")
        parser = DependencyParser(str(tmp_path))
        result = parser.parse_requirements(str(req))

        assert len(result) == 3
        assert result[0] == {"name": "requests", "version": "==2.31.0", "extras": []}
        assert result[1] == {"name": "flask", "version": ">=3.0", "extras": []}
        assert result[2] == {"name": "numpy", "version": "", "extras": []}

    def test_extras(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("uvicorn[standard]>=0.30\nPyJWT[crypto]>=2.8.0\n")
        parser = DependencyParser(str(tmp_path))
        result = parser.parse_requirements(str(req))

        assert result[0]["name"] == "uvicorn"
        assert result[0]["extras"] == ["standard"]
        assert result[1]["name"] == "PyJWT"
        assert result[1]["extras"] == ["crypto"]

    def test_skips_comments_and_directives(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text(
            "# This is a comment\n"
            "-r base.txt\n"
            "-e git+https://github.com/...\n"
            "--index-url https://pypi.org/simple\n"
            "real-package>=1.0\n"
        )
        parser = DependencyParser(str(tmp_path))
        result = parser.parse_requirements(str(req))

        assert len(result) == 1
        assert result[0]["name"] == "real-package"

    def test_inline_comments_stripped(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("requests==2.31.0 # HTTP library\n")
        parser = DependencyParser(str(tmp_path))
        result = parser.parse_requirements(str(req))

        assert result[0]["version"] == "==2.31.0"

    def test_empty_file(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("")
        parser = DependencyParser(str(tmp_path))
        assert parser.parse_requirements(str(req)) == []

    def test_missing_file(self, tmp_path):
        parser = DependencyParser(str(tmp_path))
        assert parser.parse_requirements(str(tmp_path / "nope.txt")) == []

    def test_tilde_equals(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("django~=4.2\n")
        parser = DependencyParser(str(tmp_path))
        result = parser.parse_requirements(str(req))
        assert result[0]["version"] == "~=4.2"


# ------------------------------------------------------------------
# parse_pyproject
# ------------------------------------------------------------------


class TestParsePyproject:
    def test_basic(self, tmp_path):
        pp = tmp_path / "pyproject.toml"
        pp.write_text(
            '[project]\nname = "myapp"\n'
            'dependencies = [\n  "fastapi>=0.100",\n  "pydantic>=2.0",\n]\n'
        )
        parser = DependencyParser(str(tmp_path))
        result = parser.parse_pyproject(str(pp))

        assert len(result) == 2
        assert result[0]["name"] == "fastapi"
        assert result[0]["version"] == ">=0.100"

    def test_extras_in_pyproject(self, tmp_path):
        pp = tmp_path / "pyproject.toml"
        pp.write_text('[project]\ndependencies = ["sqlalchemy[asyncio]>=2.0"]\n')
        parser = DependencyParser(str(tmp_path))
        result = parser.parse_pyproject(str(pp))

        assert result[0]["name"] == "sqlalchemy"
        assert result[0]["extras"] == ["asyncio"]

    def test_no_project_section(self, tmp_path):
        pp = tmp_path / "pyproject.toml"
        pp.write_text("[tool.ruff]\nline-length = 100\n")
        parser = DependencyParser(str(tmp_path))
        assert parser.parse_pyproject(str(pp)) == []

    def test_missing_file(self, tmp_path):
        parser = DependencyParser(str(tmp_path))
        assert parser.parse_pyproject(str(tmp_path / "pyproject.toml")) == []

    def test_invalid_toml(self, tmp_path):
        pp = tmp_path / "pyproject.toml"
        pp.write_text("not valid toml {{{\n")
        parser = DependencyParser(str(tmp_path))
        assert parser.parse_pyproject(str(pp)) == []


# ------------------------------------------------------------------
# get_available_packages
# ------------------------------------------------------------------


class TestGetAvailablePackages:
    async def test_scans_requirements_and_pyproject(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("requests>=2.31\n")
        pp = tmp_path / "pyproject.toml"
        pp.write_text('[project]\ndependencies = ["flask>=3.0"]\n')

        parser = DependencyParser(str(tmp_path))
        pkgs = await parser.get_available_packages()

        assert "requests" in pkgs
        assert "flask" in pkgs

    async def test_caches_result(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("numpy\n")
        parser = DependencyParser(str(tmp_path))

        first = await parser.get_available_packages()
        # Remove the file to prove caching works
        req.unlink()
        second = await parser.get_available_packages()
        assert first is second

    async def test_requirements_subdir(self, tmp_path):
        req_dir = tmp_path / "requirements"
        req_dir.mkdir()
        (req_dir / "dev.txt").write_text("pytest>=8.0\n")

        parser = DependencyParser(str(tmp_path))
        pkgs = await parser.get_available_packages()
        assert "pytest" in pkgs

    async def test_empty_project(self, tmp_path):
        parser = DependencyParser(str(tmp_path))
        pkgs = await parser.get_available_packages()
        assert pkgs == {}


# ------------------------------------------------------------------
# get_package_info
# ------------------------------------------------------------------


class TestGetPackageInfo:
    async def test_found(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("requests>=2.31.0\n")
        parser = DependencyParser(str(tmp_path))
        await parser.get_available_packages()

        info = parser.get_package_info("requests")
        assert info is not None
        assert info["name"] == "requests"

    async def test_not_found(self, tmp_path):
        parser = DependencyParser(str(tmp_path))
        await parser.get_available_packages()
        assert parser.get_package_info("nonexistent") is None

    async def test_case_insensitive(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("Flask>=3.0\n")
        parser = DependencyParser(str(tmp_path))
        await parser.get_available_packages()

        assert parser.get_package_info("flask") is not None
        assert parser.get_package_info("FLASK") is not None


# ------------------------------------------------------------------
# get_similar_packages
# ------------------------------------------------------------------


class TestGetSimilarPackages:
    async def test_returns_close_matches(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("requests\nflask\nfastapi\n")
        parser = DependencyParser(str(tmp_path))
        await parser.get_available_packages()

        matches = parser.get_similar_packages("requsts")  # typo
        assert "requests" in matches

    async def test_no_matches(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("numpy\n")
        parser = DependencyParser(str(tmp_path))
        await parser.get_available_packages()

        assert parser.get_similar_packages("zzzzzzz") == []
