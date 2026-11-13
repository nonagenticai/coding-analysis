from processing.contract_analysis.checks.response_check import check_response_fields
from processing.contract_analysis.checks.types import ViolationSeverity
from processing.contract_analysis.parsers.types import (
    ClientCall,
    ModelField,
    PydanticModel,
    RouteDefinition,
    ServiceSchema,
)


def test_all_response_reads_exist():
    calls = [
        ClientCall(
            method="POST",
            path="/api/analyze",
            response_reads=["job_id", "status"],
            caller="start()",
        )
    ]
    routes = [
        RouteDefinition(method="POST", path="/api/analyze", response_model="AnalysisResponse")
    ]
    schema = ServiceSchema(
        models={
            "AnalysisResponse": PydanticModel(
                name="AnalysisResponse",
                bases=["BaseModel"],
                fields=[
                    ModelField(name="job_id", type_annotation="str", required=True),
                    ModelField(name="status", type_annotation="str", required=True),
                ],
            )
        }
    )
    violations = check_response_fields(calls, routes, schema)
    assert len(violations) == 0


def test_reading_nonexistent_field():
    calls = [
        ClientCall(
            method="POST", path="/api/analyze", response_reads=["missing_field"], caller="start()"
        )
    ]
    routes = [
        RouteDefinition(method="POST", path="/api/analyze", response_model="AnalysisResponse")
    ]
    schema = ServiceSchema(
        models={
            "AnalysisResponse": PydanticModel(
                name="AnalysisResponse",
                bases=["BaseModel"],
                fields=[
                    ModelField(name="job_id", type_annotation="str", required=True),
                ],
            )
        }
    )
    violations = check_response_fields(calls, routes, schema)
    assert len(violations) == 1
    assert violations[0].severity == ViolationSeverity.FAIL
    assert "missing_field" in violations[0].message


def test_route_aware_response_check():
    """Field exists on Model A but NOT on route's actual response Model B."""
    calls = [
        ClientCall(
            method="GET",
            path="/api/items/{id}",
            response_reads=["extra_field"],
            caller="get()",
        )
    ]
    routes = [
        RouteDefinition(method="GET", path="/api/items/{item_id}", response_model="ItemResponse")
    ]
    schema = ServiceSchema(
        models={
            "ItemResponse": PydanticModel(
                name="ItemResponse",
                fields=[
                    ModelField(name="id", type_annotation="str", required=True),
                    ModelField(name="name", type_annotation="str", required=True),
                ],
            ),
            "OtherResponse": PydanticModel(
                name="OtherResponse",
                fields=[
                    ModelField(name="extra_field", type_annotation="str", required=True),
                ],
            ),
        }
    )
    violations = check_response_fields(calls, routes, schema)
    assert len(violations) == 1
    assert violations[0].severity == ViolationSeverity.FAIL
    assert "extra_field" in violations[0].message
    assert "ItemResponse" in violations[0].message


def test_no_response_model_fallback_warn():
    """Route with no response_model falls back to pooled check with WARN."""
    calls = [
        ClientCall(method="GET", path="/api/foo", response_reads=["nonexistent"], caller="get()")
    ]
    routes = [RouteDefinition(method="GET", path="/api/foo")]
    schema = ServiceSchema(
        models={
            "SomeResponse": PydanticModel(
                name="SomeResponse",
                fields=[
                    ModelField(name="id", type_annotation="str", required=True),
                ],
            )
        }
    )
    violations = check_response_fields(calls, routes, schema)
    assert len(violations) == 1
    assert violations[0].severity == ViolationSeverity.WARN
