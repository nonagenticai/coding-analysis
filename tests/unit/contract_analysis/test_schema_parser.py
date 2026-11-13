from processing.contract_analysis.parsers.schema_parser import parse_schemas

SAMPLE_SCHEMA = """
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    pages: list[dict] = Field(..., min_length=1)
    user_context: dict | None = None
    keep_lang: str = "en"
    config: dict = Field(default_factory=dict)

class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    pages: list[dict] = []
    error: str | None = None
"""


def test_parse_models():
    result = parse_schemas(SAMPLE_SCHEMA)
    assert len(result.models) == 2
    assert "AnalyzeRequest" in result.models
    assert "AnalysisResponse" in result.models


def test_required_fields():
    result = parse_schemas(SAMPLE_SCHEMA)
    req = result.models["AnalyzeRequest"]
    fields_by_name = {f.name: f for f in req.fields}
    assert fields_by_name["pages"].required is True
    assert fields_by_name["user_context"].required is False
    assert fields_by_name["keep_lang"].required is False


def test_field_types():
    result = parse_schemas(SAMPLE_SCHEMA)
    resp = result.models["AnalysisResponse"]
    fields_by_name = {f.name: f for f in resp.fields}
    assert fields_by_name["job_id"].type_annotation == "str"
    assert fields_by_name["pages"].type_annotation == "list[dict]"
    assert fields_by_name["error"].type_annotation == "str | None"


def test_inheritance():
    src = """
from pydantic import BaseModel

class Base(BaseModel):
    id: str
    name: str

class Extended(Base):
    extra: int
"""
    result = parse_schemas(src)
    ext = result.models["Extended"]
    field_names = [f.name for f in ext.fields]
    assert "id" in field_names
    assert "name" in field_names
    assert "extra" in field_names


def test_empty_source():
    result = parse_schemas("")
    assert len(result.models) == 0
    assert len(result.parse_errors) == 0


def test_optional_syntax():
    """Parse Optional[X] type annotations."""
    source = """
from typing import Optional
from pydantic import BaseModel

class FooModel(BaseModel):
    name: str
    description: Optional[str] = None
    count: Optional[int]
"""
    result = parse_schemas(source)
    model = result.models["FooModel"]
    fields_by_name = {f.name: f for f in model.fields}
    # Optional[str] should normalize to "str | None"
    assert fields_by_name["description"].type_annotation == "str | None"
    assert fields_by_name["description"].required is False
    # Optional[int] without default is still required in Pydantic v2
    assert fields_by_name["count"].type_annotation == "int | None"
    assert fields_by_name["count"].required is True


def test_optional_complex_type():
    """Parse Optional[list[str]] correctly."""
    source = """
from typing import Optional
from pydantic import BaseModel

class BarModel(BaseModel):
    tags: Optional[list[str]] = None
"""
    result = parse_schemas(source)
    model = result.models["BarModel"]
    fields_by_name = {f.name: f for f in model.fields}
    assert fields_by_name["tags"].type_annotation == "list[str] | None"
    assert fields_by_name["tags"].required is False


def test_field_positional_default():
    """Field(None, ...) uses positional default — field is optional."""
    source = """
from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Required prompt")
    provider: str | None = Field(None, description="Optional provider")
    width: int | None = Field(None, ge=64, le=4096)
    budget: float = Field(default=5.0, description="Budget limit")
"""
    result = parse_schemas(source)
    model = result.models["GenerateRequest"]
    fields_by_name = {f.name: f for f in model.fields}
    assert fields_by_name["prompt"].required is True
    assert fields_by_name["provider"].required is False
    assert fields_by_name["width"].required is False
    assert fields_by_name["budget"].required is False


def test_annotated_field_patterns():
    """Parse Annotated[X, Field(...)] patterns."""
    source = """
from typing import Annotated
from pydantic import BaseModel, Field

class AnnotatedModel(BaseModel):
    name: str
    desc: Annotated[str, Field(default="")]
    count: Annotated[int, Field(...)]
    label: Annotated[str | None, Field(default=None)]
"""
    result = parse_schemas(source)
    model = result.models["AnnotatedModel"]
    fields_by_name = {f.name: f for f in model.fields}
    assert fields_by_name["name"].required is True
    assert fields_by_name["desc"].required is False
    assert fields_by_name["count"].required is True
    assert fields_by_name["label"].required is False


def test_model_validator_detected():
    """Detect @model_validator decorated methods."""
    source = """
from pydantic import BaseModel, model_validator

class ValidatedModel(BaseModel):
    name: str
    value: int

    @model_validator(mode="after")
    def check_value(self):
        return self
"""
    result = parse_schemas(source)
    assert result.models["ValidatedModel"].has_model_validator is True


def test_model_validator_absent():
    """Models without @model_validator have has_model_validator=False."""
    source = """
from pydantic import BaseModel

class SimpleModel(BaseModel):
    name: str
"""
    result = parse_schemas(source)
    assert result.models["SimpleModel"].has_model_validator is False


def test_resolve_false_skips_inheritance():
    """parse_schemas(resolve=False) does not merge parent fields."""
    source = """
from pydantic import BaseModel

class Base(BaseModel):
    id: str

class Child(Base):
    name: str
"""
    result = parse_schemas(source, resolve=False)
    child = result.models["Child"]
    field_names = [f.name for f in child.fields]
    assert "id" not in field_names
    assert "name" in field_names
