from processing.contract_analysis.checks.flow_check import check_flow
from processing.contract_analysis.flow.registry import FlowEdge
from processing.contract_analysis.parsers.types import ModelField, PydanticModel, ServiceSchema


def _schema(models: dict[str, list[tuple[str, str, bool]]]) -> ServiceSchema:
    """Helper: build ServiceSchema from {name: [(field, type, required)]}."""
    result = ServiceSchema()
    for name, fields in models.items():
        result.models[name] = PydanticModel(
            name=name,
            fields=[ModelField(n, t, r) for n, t, r in fields],
        )
    return result


def test_required_field_missing_is_fail():
    producer = _schema({"Foo": [("url", "str", True)]})
    consumer = _schema({"Bar": [("url", "str", True), ("title", "str", True)]})
    edge = FlowEdge("Foo", "Bar", label="test")
    violations = check_flow(edge, producer, consumer)
    fails = [v for v in violations if v.severity == "FAIL"]
    assert any("title" in v.message for v in fails)


def test_matching_fields_no_violations():
    producer = _schema({"Foo": [("url", "str", True), ("title", "str", True)]})
    consumer = _schema({"Bar": [("url", "str", True), ("title", "str", True)]})
    edge = FlowEdge("Foo", "Bar", label="test")
    violations = check_flow(edge, producer, consumer)
    assert len(violations) == 0


def test_extra_producer_fields_ignored():
    producer = _schema({"Foo": [("url", "str", True), ("extra", "str", True)]})
    consumer = _schema({"Bar": [("url", "str", True)]})
    edge = FlowEdge("Foo", "Bar", label="test")
    violations = check_flow(edge, producer, consumer)
    assert len(violations) == 0


def test_type_mismatch_is_fail():
    producer = _schema({"Foo": [("items", "str", True)]})
    consumer = _schema({"Bar": [("items", "list[str]", True)]})
    edge = FlowEdge("Foo", "Bar", label="test")
    violations = check_flow(edge, producer, consumer)
    fails = [v for v in violations if v.severity == "FAIL"]
    assert len(fails) >= 1


def test_enum_to_str_is_warn():
    producer = _schema({"Foo": [("page_type", "PageType", True)]})
    consumer = _schema({"Bar": [("page_type", "str", True)]})
    edge = FlowEdge("Foo", "Bar", label="test")
    violations = check_flow(edge, producer, consumer)
    assert all(v.severity == "WARN" for v in violations)


def test_optional_missing_is_warn_not_fail():
    producer = _schema({"Foo": [("url", "str", True)]})
    consumer = _schema({"Bar": [("url", "str", True), ("meta", "str", False)]})
    edge = FlowEdge("Foo", "Bar", label="test")
    violations = check_flow(edge, producer, consumer)
    warns = [v for v in violations if v.severity == "WARN"]
    fails = [v for v in violations if v.severity == "FAIL"]
    assert len(fails) == 0
    assert any("meta" in v.message for v in warns)


def test_field_name_mismatch_detected():
    """navigation vs navigation_elements - the real w2v bug."""
    producer = _schema(
        {
            "PageAnalysisResponse": [
                ("page_id", "str", True),
                ("navigation", "list[NavigationElementResponse]", False),
            ]
        }
    )
    consumer = _schema(
        {
            "PageAnalysisInput": [
                ("page_id", "str", True),
                ("navigation_elements", "list[dict]", False),
            ]
        }
    )
    edge = FlowEdge("PageAnalysisResponse", "PageAnalysisInput", label="test")
    violations = check_flow(edge, producer, consumer)
    warns = [v for v in violations if v.severity == "WARN"]
    assert any("navigation_elements" in v.message for v in warns)


def test_missing_model_warns():
    producer = _schema({})
    consumer = _schema({"Bar": [("url", "str", True)]})
    edge = FlowEdge("Foo", "Bar", label="test")
    violations = check_flow(edge, producer, consumer)
    assert len(violations) == 1
    assert "not found" in violations[0].message


def test_forwarded_fields_filters():
    """Only forwarded fields are checked; non-forwarded required fields FAIL."""
    producer = _schema({"Foo": [("url", "str", True), ("title", "str", True)]})
    consumer = _schema(
        {"Bar": [("url", "str", True), ("title", "str", True), ("extra", "str", True)]}
    )
    edge = FlowEdge("Foo", "Bar", label="test", forwarded_fields={"url", "title"})
    violations = check_flow(edge, producer, consumer)
    fails = [v for v in violations if v.severity == "FAIL"]
    assert any("extra" in v.message for v in fails)
