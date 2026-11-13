from processing.contract_analysis.checks.request_check import check_request_fields
from processing.contract_analysis.checks.types import ViolationSeverity
from processing.contract_analysis.parsers.types import (
    ClientCall,
    ModelField,
    PydanticModel,
    RouteDefinition,
    ServiceSchema,
)


def _schema_with_request(fields: list[ModelField]) -> ServiceSchema:
    return ServiceSchema(
        models={
            "AnalyzeRequest": PydanticModel(
                name="AnalyzeRequest",
                bases=["BaseModel"],
                fields=fields,
            )
        }
    )


def test_all_required_fields_sent():
    calls = [
        ClientCall(
            method="POST",
            path="/api/analyze",
            payload_fields=["pages", "keep_lang"],
            caller="start()",
        )
    ]
    routes = [RouteDefinition(method="POST", path="/api/analyze", request_model="AnalyzeRequest")]
    schema = _schema_with_request(
        [
            ModelField(name="pages", type_annotation="list", required=True),
            ModelField(name="keep_lang", type_annotation="str", required=True),
        ]
    )
    violations = check_request_fields(calls, routes, schema)
    assert len(violations) == 0


def test_missing_required_field():
    calls = [
        ClientCall(method="POST", path="/api/analyze", payload_fields=["pages"], caller="start()")
    ]
    routes = [RouteDefinition(method="POST", path="/api/analyze", request_model="AnalyzeRequest")]
    schema = _schema_with_request(
        [
            ModelField(name="pages", type_annotation="list", required=True),
            ModelField(name="config", type_annotation="dict", required=True),
        ]
    )
    violations = check_request_fields(calls, routes, schema)
    assert len(violations) == 1
    assert violations[0].severity == ViolationSeverity.FAIL
    assert "config" in violations[0].message


def test_optional_field_not_flagged():
    calls = [
        ClientCall(method="POST", path="/api/analyze", payload_fields=["pages"], caller="start()")
    ]
    routes = [RouteDefinition(method="POST", path="/api/analyze", request_model="AnalyzeRequest")]
    schema = _schema_with_request(
        [
            ModelField(name="pages", type_annotation="list", required=True),
            ModelField(name="user_context", type_annotation="dict | None", required=False),
        ]
    )
    violations = check_request_fields(calls, routes, schema)
    assert len(violations) == 0


def test_opaque_payload_no_violation():
    """No request_field violation when payload_opaque is True."""
    calls = [ClientCall(method="POST", path="/api/foo", payload_opaque=True)]
    routes = [RouteDefinition(method="POST", path="/api/foo", request_model="FooRequest")]
    schema = ServiceSchema(
        models={
            "FooRequest": PydanticModel(
                name="FooRequest",
                fields=[
                    ModelField(name="required_field", type_annotation="str", required=True),
                ],
            )
        }
    )
    violations = check_request_fields(calls, routes, schema)
    assert len(violations) == 0


def test_payload_model_resolves_fields():
    """payload_model uses model fields instead of payload_fields."""
    calls = [
        ClientCall(
            method="POST",
            path="/api/foo",
            payload_opaque=True,
            payload_model="CreateRequest",
        )
    ]
    routes = [RouteDefinition(method="POST", path="/api/foo", request_model="FooRequest")]
    schema = ServiceSchema(
        models={
            "FooRequest": PydanticModel(
                name="FooRequest",
                fields=[
                    ModelField(name="name", type_annotation="str", required=True),
                    ModelField(name="value", type_annotation="int", required=True),
                ],
            ),
            "CreateRequest": PydanticModel(
                name="CreateRequest",
                fields=[
                    ModelField(name="name", type_annotation="str", required=True),
                    ModelField(name="value", type_annotation="int", required=True),
                ],
            ),
        }
    )
    violations = check_request_fields(calls, routes, schema)
    assert len(violations) == 0


def test_payload_model_missing_field_violation():
    """payload_model with missing required field produces violation."""
    calls = [
        ClientCall(
            method="POST",
            path="/api/foo",
            payload_opaque=True,
            payload_model="PartialRequest",
        )
    ]
    routes = [RouteDefinition(method="POST", path="/api/foo", request_model="FooRequest")]
    schema = ServiceSchema(
        models={
            "FooRequest": PydanticModel(
                name="FooRequest",
                fields=[
                    ModelField(name="name", type_annotation="str", required=True),
                    ModelField(name="value", type_annotation="int", required=True),
                ],
            ),
            "PartialRequest": PydanticModel(
                name="PartialRequest",
                fields=[
                    ModelField(name="name", type_annotation="str", required=True),
                ],
            ),
        }
    )
    violations = check_request_fields(calls, routes, schema)
    assert len(violations) == 1
    assert "value" in violations[0].message
