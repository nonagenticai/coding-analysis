from processing.contract_analysis.checks.type_check import (
    _base_type,
    _normalize_type,
    _types_compatible,
    check_types,
)
from processing.contract_analysis.checks.types import ViolationSeverity
from processing.contract_analysis.parsers.types import (
    ClientCall,
    ModelField,
    PydanticModel,
    ServiceSchema,
)


def test_compatible_types_no_warnings():
    calls = [ClientCall(method="POST", path="/api/x", payload_fields=["name"], caller="go()")]
    schema = ServiceSchema(
        models={
            "XRequest": PydanticModel(
                name="XRequest",
                bases=["BaseModel"],
                fields=[
                    ModelField(name="name", type_annotation="str", required=True),
                ],
            )
        }
    )
    violations = check_types(calls, schema)
    assert len(violations) == 0


def test_no_crash_on_empty():
    violations = check_types([], ServiceSchema())
    assert len(violations) == 0


def test_payload_model_type_mismatch():
    """Phase 1: client model has str for a field, service model has int."""
    calls = [
        ClientCall(method="POST", path="/api/x", payload_model="ClientPayload", caller="do_thing()")
    ]
    schema = ServiceSchema(
        models={
            "ClientPayload": PydanticModel(
                name="ClientPayload",
                bases=["BaseModel"],
                fields=[ModelField(name="count", type_annotation="str", required=True)],
            ),
            "CreateRequest": PydanticModel(
                name="CreateRequest",
                bases=["BaseModel"],
                fields=[ModelField(name="count", type_annotation="int", required=True)],
            ),
        }
    )
    violations = check_types(calls, schema)
    assert len(violations) == 1
    assert violations[0].severity == ViolationSeverity.WARN
    assert violations[0].check == "type_mismatch"
    assert "count" in violations[0].message


def test_payload_model_compatible_types_no_warning():
    """Phase 1: int vs float are in the same compatibility group."""
    calls = [ClientCall(method="POST", path="/api/x", payload_model="ClientPayload", caller="f()")]
    schema = ServiceSchema(
        models={
            "ClientPayload": PydanticModel(
                name="ClientPayload",
                bases=["BaseModel"],
                fields=[ModelField(name="value", type_annotation="int", required=True)],
            ),
            "UpdateRequest": PydanticModel(
                name="UpdateRequest",
                bases=["BaseModel"],
                fields=[ModelField(name="value", type_annotation="float", required=True)],
            ),
        }
    )
    violations = check_types(calls, schema)
    assert len(violations) == 0


def test_payload_model_optional_compatible():
    """Phase 1: str vs Optional[str] are compatible."""
    calls = [ClientCall(method="POST", path="/api/x", payload_model="ClientPayload", caller="f()")]
    schema = ServiceSchema(
        models={
            "ClientPayload": PydanticModel(
                name="ClientPayload",
                bases=["BaseModel"],
                fields=[ModelField(name="name", type_annotation="str", required=True)],
            ),
            "CreateRequest": PydanticModel(
                name="CreateRequest",
                bases=["BaseModel"],
                fields=[ModelField(name="name", type_annotation="Optional[str]", required=False)],
            ),
        }
    )
    violations = check_types(calls, schema)
    assert len(violations) == 0


def test_payload_model_any_compatible():
    """Phase 1: Any is compatible with everything."""
    calls = [ClientCall(method="POST", path="/api/x", payload_model="ClientPayload", caller="f()")]
    schema = ServiceSchema(
        models={
            "ClientPayload": PydanticModel(
                name="ClientPayload",
                bases=["BaseModel"],
                fields=[ModelField(name="data", type_annotation="Any", required=True)],
            ),
            "CreateRequest": PydanticModel(
                name="CreateRequest",
                bases=["BaseModel"],
                fields=[ModelField(name="data", type_annotation="dict[str, int]", required=True)],
            ),
        }
    )
    violations = check_types(calls, schema)
    assert len(violations) == 0


def test_payload_model_same_model_skipped():
    """Phase 1: same model name is not compared against itself."""
    calls = [ClientCall(method="POST", path="/api/x", payload_model="CreateRequest", caller="f()")]
    schema = ServiceSchema(
        models={
            "CreateRequest": PydanticModel(
                name="CreateRequest",
                bases=["BaseModel"],
                fields=[ModelField(name="count", type_annotation="str", required=True)],
            ),
        }
    )
    violations = check_types(calls, schema)
    assert len(violations) == 0


def test_response_type_conflict():
    """Phase 2: two response models with same field name but incompatible types."""
    calls = [
        ClientCall(method="GET", path="/api/x", response_reads=["status"], caller="get_status()")
    ]
    schema = ServiceSchema(
        models={
            "StatusResponse": PydanticModel(
                name="StatusResponse",
                bases=["BaseModel"],
                fields=[ModelField(name="status", type_annotation="str", required=True)],
            ),
            "DetailResult": PydanticModel(
                name="DetailResult",
                bases=["BaseModel"],
                fields=[ModelField(name="status", type_annotation="int", required=True)],
            ),
        }
    )
    violations = check_types(calls, schema)
    assert len(violations) == 1
    assert "conflicting types" in violations[0].message


def test_response_type_compatible_no_warning():
    """Phase 2: compatible types across response models do not trigger."""
    calls = [ClientCall(method="GET", path="/api/x", response_reads=["count"], caller="f()")]
    schema = ServiceSchema(
        models={
            "AResponse": PydanticModel(
                name="AResponse",
                bases=["BaseModel"],
                fields=[ModelField(name="count", type_annotation="int", required=True)],
            ),
            "BResult": PydanticModel(
                name="BResult",
                bases=["BaseModel"],
                fields=[ModelField(name="count", type_annotation="float", required=True)],
            ),
        }
    )
    violations = check_types(calls, schema)
    assert len(violations) == 0


def test_response_fallback_to_all_models():
    """Phase 2: when no models match response naming, all models are used."""
    calls = [ClientCall(method="GET", path="/api/x", response_reads=["val"], caller="f()")]
    schema = ServiceSchema(
        models={
            "Alpha": PydanticModel(
                name="Alpha",
                bases=["BaseModel"],
                fields=[ModelField(name="val", type_annotation="str", required=True)],
            ),
            "Beta": PydanticModel(
                name="Beta",
                bases=["BaseModel"],
                fields=[ModelField(name="val", type_annotation="int", required=True)],
            ),
        }
    )
    violations = check_types(calls, schema)
    assert len(violations) == 1


def test_heuristic_count_field_wrong_type():
    """Phase 3: field named 'count' with type 'str' in service model should warn."""
    calls = [ClientCall(method="POST", path="/api/x", payload_fields=["count"], caller="f()")]
    schema = ServiceSchema(
        models={
            "SearchRequest": PydanticModel(
                name="SearchRequest",
                bases=["BaseModel"],
                fields=[ModelField(name="count", type_annotation="str", required=True)],
            ),
        }
    )
    violations = check_types(calls, schema)
    assert len(violations) == 1
    assert "count" in violations[0].message
    assert "expected numeric/bool type" in violations[0].message


def test_heuristic_page_field_correct_type():
    """Phase 3: field named 'page' with type 'int' produces no warning."""
    calls = [ClientCall(method="POST", path="/api/x", payload_fields=["page"], caller="f()")]
    schema = ServiceSchema(
        models={
            "ListRequest": PydanticModel(
                name="ListRequest",
                bases=["BaseModel"],
                fields=[ModelField(name="page", type_annotation="int", required=True)],
            ),
        }
    )
    violations = check_types(calls, schema)
    assert len(violations) == 0


def test_heuristic_skipped_when_payload_model_set():
    """Phase 3: when payload_model is set, heuristic phase is skipped."""
    calls = [
        ClientCall(
            method="POST",
            path="/api/x",
            payload_fields=["count"],
            payload_model="MyModel",
            caller="f()",
        )
    ]
    schema = ServiceSchema(
        models={
            "SearchRequest": PydanticModel(
                name="SearchRequest",
                bases=["BaseModel"],
                fields=[ModelField(name="count", type_annotation="str", required=True)],
            ),
        }
    )
    violations = check_types(calls, schema)
    # payload_model is set so Phase 3 skips (even though MyModel isn't in schema)
    assert len(violations) == 0


def test_empty_schema_no_crash():
    calls = [ClientCall(method="POST", path="/api/x", payload_fields=["a"], caller="f()")]
    violations = check_types(calls, ServiceSchema(models={}))
    assert len(violations) == 0


def test_normalize_type_optional():
    assert _normalize_type("Optional[str]") == "str | None"
    assert _normalize_type("  str  ") == "str"


def test_base_type_generic():
    assert _base_type("list[str]") == "list"
    assert _base_type("dict[str, int]") == "dict"
    assert _base_type("str | None") == "str"
    assert _base_type("str") == "str"


def test_types_compatible_groups():
    assert _types_compatible("int", "float")
    assert _types_compatible("str", "str | None")
    assert not _types_compatible("str", "int")
    assert _types_compatible("Any", "dict[str, int]")
