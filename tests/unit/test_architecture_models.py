"""Tests for models/architecture_models.py.

Validates Component validators, ModelMigrationTools migration chains,
model field constraints, and computed properties.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.architecture_models import (
    ArchitecturalMetric,
    ArchitectureGraphMetrics,
    ArchitectureGraphResponse,
    ArchitectureJobRequest,
    ArchitectureQualityIssue,
    CodeGraphData,
    Component,
    ComponentExtractionStats,
    DependencyRiskSummary,
    DetectedPattern,
    GraphEdge,
    GraphNode,
    HotspotAnalysisResult,
    HotspotPrediction,
    Layer,
    LayerViolation,
    ModelMigrationTools,
    QualityGateCheck,
    QualityGateResult,
    RepoConfig,
    RuntimeData,
    Vulnerability,
)

# ---------------------------------------------------------------------------
# Component validators
# ---------------------------------------------------------------------------


class TestComponentValidators:
    """Test Component model field validators."""

    @pytest.mark.parametrize(
        "comp_id",
        ["my-comp_1", "abc", "A-B-C", "under_score", "123"],
    )
    def test_valid_ids_accepted(self, comp_id: str):
        c = Component(id=comp_id, name="Test", files=["a.py"])
        assert c.id == comp_id

    @pytest.mark.parametrize(
        "bad_id",
        ["bad id!", "has space", "special@char", "semi;colon", "dot.dot"],
    )
    def test_invalid_ids_rejected(self, bad_id: str):
        with pytest.raises(ValidationError, match="alphanumeric"):
            Component(id=bad_id, name="Test", files=["a.py"])

    def test_empty_files_rejected(self):
        with pytest.raises(ValidationError, match="at least one file"):
            Component(id="comp", name="Test", files=[])

    def test_duplicate_dependencies_deduplicated(self):
        c = Component(
            id="comp",
            name="Test",
            files=["a.py"],
            dependencies=["dep1", "dep1", "dep2"],
        )
        assert sorted(c.dependencies) == ["dep1", "dep2"]

    def test_self_dependency_rejected(self):
        with pytest.raises(ValidationError, match="cannot depend on itself"):
            Component(
                id="comp",
                name="Test",
                files=["a.py"],
                dependencies=["comp"],
            )

    def test_layer_none_accepted(self):
        c = Component(id="comp", name="Test", files=["a.py"], layer=None)
        assert c.layer is None

    def test_layer_string_accepted(self):
        c = Component(id="comp", name="Test", files=["a.py"], layer="service")
        assert c.layer == "service"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            Component(id="comp", name="Test", files=["a.py"], bogus="nope")


class TestComponentGetFileExtensions:
    """Test Component.get_file_extensions()."""

    @pytest.mark.parametrize(
        ("files", "expected_exts"),
        [
            (["a.py", "b.js", "c.py"], {"py", "js"}),
            (["main.ts", "style.css"], {"ts", "css"}),
            (["noext"], set()),
            (["a.py", "b.py"], {"py"}),
        ],
    )
    def test_extensions_extracted(self, files: list[str], expected_exts: set[str]):
        c = Component(id="comp", name="Test", files=files)
        assert c.get_file_extensions() == expected_exts


class TestComponentLayerHierarchy:
    """Test Component._get_layer_hierarchy()."""

    def test_default_hierarchy_when_no_config_layers(self):
        c = Component(id="comp", name="Test", files=["a.py"])
        config = type("Config", (), {"layers": {}})()
        hierarchy = c._get_layer_hierarchy(config)
        assert hierarchy["infrastructure"] < hierarchy["domain"]
        assert hierarchy["domain"] < hierarchy["service"]

    def test_custom_layers_from_config(self):
        c = Component(id="comp", name="Test", files=["a.py"])
        config = type("Config", (), {"layers": {"api": [], "core": [], "data": []}})()
        hierarchy = c._get_layer_hierarchy(config)
        assert "data" in hierarchy
        assert "api" in hierarchy
        assert "core" in hierarchy

    def test_known_layer_names_get_default_levels(self):
        c = Component(id="comp", name="Test", files=["a.py"])
        config = type("Config", (), {"layers": {"infrastructure": [], "ui": []}})()
        hierarchy = c._get_layer_hierarchy(config)
        assert hierarchy["infrastructure"] < hierarchy["ui"]


# ---------------------------------------------------------------------------
# ArchitecturalMetric
# ---------------------------------------------------------------------------


class TestArchitecturalMetric:
    def test_basic_creation(self):
        m = ArchitecturalMetric(
            component_id="c1", metric_type="instability", value=0.5, description="desc"
        )
        assert m.component_id == "c1"
        assert m.value == 0.5
        assert m.instability is None

    def test_martin_metrics(self):
        m = ArchitecturalMetric(
            component_id="c1",
            metric_type="combined",
            value=0.7,
            description="test",
            instability=0.3,
            abstractness=0.5,
            distance=0.2,
        )
        assert m.instability == 0.3
        assert m.abstractness == 0.5
        assert m.distance == 0.2

    @pytest.mark.parametrize("field", ["instability", "abstractness", "distance"])
    def test_martin_metrics_bounded_0_to_1(self, field: str):
        with pytest.raises(ValidationError):
            ArchitecturalMetric(
                component_id="c1",
                metric_type="x",
                value=0.0,
                description="d",
                **{field: 1.5},
            )


# ---------------------------------------------------------------------------
# DetectedPattern
# ---------------------------------------------------------------------------


class TestDetectedPattern:
    def test_basic_creation(self):
        p = DetectedPattern(
            pattern_type="singleton",
            confidence=0.9,
            components=["c1", "c2"],
            description="Singleton pattern",
        )
        assert p.pattern_type == "singleton"
        assert p.confidence == 0.9
        assert len(p.components) == 2

    def test_confidence_must_be_0_to_1(self):
        with pytest.raises(ValidationError):
            DetectedPattern(pattern_type="x", confidence=1.5, components=[], description="d")


# ---------------------------------------------------------------------------
# Vulnerability / DependencyRiskSummary
# ---------------------------------------------------------------------------


class TestVulnerability:
    def test_all_fields(self):
        v = Vulnerability(
            id="CVE-2024-1234",
            severity="high",
            summary="XSS",
            details="details",
            affected_versions=["1.0"],
            fixed_versions=["1.1"],
            references=["https://example.com"],
            published_date="2024-01-01",
            modified_date="2024-01-02",
        )
        assert v.id == "CVE-2024-1234"
        assert v.severity == "high"
        assert v.fixed_versions == ["1.1"]


class TestDependencyRiskSummary:
    def test_counts(self):
        s = DependencyRiskSummary(
            total_packages=10,
            vulnerable_packages=2,
            total_vulnerabilities=3,
            critical_vulnerabilities=1,
            high_vulnerabilities=1,
            moderate_vulnerabilities=1,
            low_vulnerabilities=0,
        )
        assert s.total_packages == 10
        assert s.vulnerable_packages == 2
        assert s.critical_vulnerabilities + s.high_vulnerabilities == 2


# ---------------------------------------------------------------------------
# RepoConfig
# ---------------------------------------------------------------------------


class TestRepoConfig:
    def test_defaults(self):
        config = RepoConfig()
        assert config.schema_version == "2.0"
        assert config.extract_docstrings is True
        assert config.min_pattern_confidence == 0.7
        assert "folder" in config.detect_by
        assert "build_file" in config.detect_by

    def test_custom_layers(self):
        config = RepoConfig(
            layers={"api": ["src/api/*"], "core": ["src/core/*"]},
            detect_by=["folder"],
        )
        assert "api" in config.layers
        assert config.detect_by == ["folder"]

    def test_custom_build_files(self):
        config = RepoConfig(build_files=["setup.py", "pyproject.toml"])
        assert "setup.py" in config.build_files


# ---------------------------------------------------------------------------
# Frozen models
# ---------------------------------------------------------------------------


class TestFrozenModels:
    def test_component_extraction_stats_immutable(self):
        stats = ComponentExtractionStats(components_found=5, edges_scanned=20)
        assert stats.components_found == 5
        with pytest.raises(ValidationError):
            stats.components_found = 10

    def test_layer_violation_immutable(self):
        v = LayerViolation(component_id="c1", expected_layer="core", actual_layer="api")
        assert v.component_id == "c1"
        with pytest.raises(ValidationError):
            v.component_id = "c2"


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------


class TestLayer:
    def test_basic_creation(self):
        layer = Layer(name="service", components=["c1", "c2"], component_count=2)
        assert layer.component_count == 2
        assert layer.dependencies == []

    def test_with_dependencies(self):
        layer = Layer(
            name="api",
            components=["c1"],
            component_count=1,
            dependencies=["service"],
        )
        assert layer.dependencies == ["service"]


# ---------------------------------------------------------------------------
# QualityGateCheck / QualityGateResult
# ---------------------------------------------------------------------------


class TestQualityGateCheck:
    @pytest.mark.parametrize(
        ("actual", "threshold", "passed"),
        [(85.0, 80.0, True), (70.0, 80.0, False), (80.0, 80.0, True)],
    )
    def test_check_values(self, actual: float, threshold: float, passed: bool):
        check = QualityGateCheck(
            name="min_score",
            description="Minimum score",
            passed=passed,
            actual=actual,
            threshold=threshold,
        )
        assert check.passed is passed
        assert check.actual == actual


class TestQualityGateResult:
    def test_grade_and_score(self):
        result = QualityGateResult(
            passed=True,
            overall_score=90.0,
            grade="A",
            gate_checks=[],
            recommendations=["improve coverage"],
        )
        assert result.grade == "A"
        assert result.passed is True
        assert result.recommendations == ["improve coverage"]

    def test_overall_score_bounded(self):
        with pytest.raises(ValidationError):
            QualityGateResult(
                passed=True,
                overall_score=101.0,
                grade="A+",
                gate_checks=[],
                recommendations=[],
            )


# ---------------------------------------------------------------------------
# ModelMigrationTools
# ---------------------------------------------------------------------------


class TestModelMigrationTools:
    def test_same_version_no_op(self):
        tools = ModelMigrationTools(RepoConfig)
        data = {"schema_version": "2.0"}
        result = tools.migrate(data, "2.0", "2.0")
        assert result is data

    def test_direct_migration(self):
        tools = ModelMigrationTools(RepoConfig)

        def migrate_1_to_2(d):
            d = dict(d)
            d["schema_version"] = "2.0"
            return d

        tools.register_version_migration("1.0", "2.0", migrate_1_to_2)
        result = tools.migrate({"schema_version": "1.0"}, "1.0", "2.0")
        assert result["schema_version"] == "2.0"

    def test_two_step_migration(self):
        tools = ModelMigrationTools(RepoConfig)
        tools.register_version_migration("1.0", "1.5", lambda d: {**d, "schema_version": "1.5"})
        tools.register_version_migration("1.5", "2.0", lambda d: {**d, "schema_version": "2.0"})
        result = tools.migrate({"schema_version": "1.0"}, "1.0", "2.0")
        assert result["schema_version"] == "2.0"

    def test_no_migration_path_raises_value_error(self):
        tools = ModelMigrationTools(RepoConfig)
        with pytest.raises(ValueError, match="No migration path"):
            tools.migrate({"schema_version": "1.0"}, "1.0", "3.0")

    def test_to_latest_creates_model_instance(self):
        tools = ModelMigrationTools(RepoConfig)
        data = {"schema_version": "2.0"}
        result = tools.to_latest(data, "latest")
        assert isinstance(result, RepoConfig)
        assert result.schema_version == "2.0"

    def test_creation_timestamp_is_iso(self):
        tools = ModelMigrationTools(RepoConfig)
        ts = tools.get_creation_timestamp()
        assert isinstance(ts, str)
        assert "T" in ts

    def test_register_overwrites_existing(self):
        tools = ModelMigrationTools(RepoConfig)
        tools.register_version_migration("1.0", "2.0", lambda d: {**d, "extra": "a"})
        tools.register_version_migration("1.0", "2.0", lambda d: {**d, "extra": "b"})
        result = tools.migrate({}, "1.0", "2.0")
        assert result["extra"] == "b"


# ---------------------------------------------------------------------------
# Graph visualization models
# ---------------------------------------------------------------------------


class TestGraphModels:
    def test_graph_node_defaults(self):
        node = GraphNode(id="n1", name="API", type="component")
        assert node.layer is None
        assert node.metrics is None
        assert node.hotspotScore is None
        assert node.issues is None

    def test_graph_node_with_metrics(self):
        node = GraphNode(
            id="n1",
            name="API",
            type="service",
            layer="api",
            metrics={"complexity": 3.5},
            hotspotScore=0.8,
            issues=2,
        )
        assert node.metrics["complexity"] == 3.5
        assert node.hotspotScore == 0.8

    def test_runtime_data(self):
        rd = RuntimeData(call_count=100, avg_duration_ms=5.5, error_rate=0.01)
        assert rd.call_count == 100
        assert rd.error_rate == 0.01

    def test_graph_edge_default_weight(self):
        edge = GraphEdge(source="n1", target="n2", type="dependency")
        assert edge.weight == 1
        assert edge.runtime_data is None

    def test_graph_edge_with_runtime(self):
        rd = RuntimeData(call_count=50, avg_duration_ms=2.0, error_rate=0.0)
        edge = GraphEdge(source="n1", target="n2", type="runtime", weight=5, runtime_data=rd)
        assert edge.runtime_data.call_count == 50

    def test_architecture_graph_metrics(self):
        m = ArchitectureGraphMetrics(
            total_components=10,
            total_dependencies=25,
            avg_complexity=3.5,
            hotspot_count=2,
        )
        assert m.total_components == 10

    def test_architecture_graph_response_composite(self):
        nodes = [GraphNode(id="n1", name="A", type="component")]
        edges = [GraphEdge(source="n1", target="n2", type="dependency")]
        metrics = ArchitectureGraphMetrics(
            total_components=1,
            total_dependencies=1,
            avg_complexity=1.0,
            hotspot_count=0,
        )
        r = ArchitectureGraphResponse(nodes=nodes, edges=edges, layers=["core"], metrics=metrics)
        assert len(r.nodes) == 1
        assert len(r.edges) == 1
        assert r.layers == ["core"]


# ---------------------------------------------------------------------------
# Hotspot models
# ---------------------------------------------------------------------------


class TestHotspotModels:
    def test_hotspot_prediction_defaults(self):
        hp = HotspotPrediction(component_id="c1", score=0.8, confidence=0.9)
        assert hp.predicted_issues == []
        assert hp.risk_factors == []
        assert hp.recommended_actions == []

    def test_hotspot_prediction_with_data(self):
        hp = HotspotPrediction(
            component_id="c1",
            score=0.8,
            confidence=0.9,
            predicted_issues=["complexity"],
            risk_factors=["high coupling"],
            recommended_actions=["refactor"],
        )
        assert hp.score == 0.8
        assert "complexity" in hp.predicted_issues

    def test_hotspot_score_bounded(self):
        with pytest.raises(ValidationError):
            HotspotPrediction(component_id="c1", score=1.5, confidence=0.5)

    def test_hotspot_analysis_result(self):
        r = HotspotAnalysisResult(
            codebase_id="cb1",
            generated_at="2024-01-01",
            hotspots=[],
            analysis_ai_model_version="1.0",
            overall_confidence=0.85,
        )
        assert r.overall_confidence == 0.85
        assert r.hotspots == []


# ---------------------------------------------------------------------------
# CodeGraphData
# ---------------------------------------------------------------------------


class TestCodeGraphData:
    def test_basic_creation(self):
        d = CodeGraphData(
            nodes=[{"id": "n1", "type": "module"}],
            edges=[{"source": "n1", "target": "n2"}],
        )
        assert len(d.nodes) == 1
        assert d.edges[0]["source"] == "n1"

    def test_empty_graph(self):
        d = CodeGraphData(nodes=[], edges=[])
        assert d.nodes == []
        assert d.edges == []


# ---------------------------------------------------------------------------
# ArchitectureJobRequest
# ---------------------------------------------------------------------------


class TestArchitectureJobRequest:
    def test_valid_request(self):
        req = ArchitectureJobRequest(
            codebase_id="my-repo", repo_path="/tmp/repo", tenant_id="tenant-a"
        )
        assert req.codebase_id == "my-repo"
        assert req.tenant_id == "tenant-a"
        assert req.force_rebuild is False

    def test_invalid_codebase_id(self):
        with pytest.raises(ValidationError):
            ArchitectureJobRequest(
                codebase_id="bad id!", repo_path="/tmp/repo", tenant_id="tenant-a"
            )

    def test_tenant_id_is_required(self):
        # tenant_id is mandatory so a graph read can never run unscoped.
        with pytest.raises(ValidationError):
            ArchitectureJobRequest(codebase_id="my-repo", repo_path="/tmp/repo")


# ---------------------------------------------------------------------------
# ArchitectureQualityIssue
# ---------------------------------------------------------------------------


class TestArchitectureQualityIssue:
    def test_creation(self):
        issue = ArchitectureQualityIssue(
            id="issue-1",
            severity="high",
            message="Unused variable",
            rule_id="W001",
            rule_name="unused-variable",
            file_path="src/main.py",
            line_number=10,
            tool="ruff",
            category="style",
        )
        assert issue.severity == "high"
        assert issue.line_number == 10
        assert issue.column_number is None
