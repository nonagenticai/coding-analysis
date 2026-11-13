"""POST /api/analysis/symbols: what one file defines and imports.

coding-gateway calls this to validate function/class references in every
language the coding app supports, not just Python. The caller sends the file
content it read from ITS OWN workspace; this endpoint never resolves a
container, because container lookup fuzzy-matches names and can answer about
the wrong tree. A missing grammar is a 503, never an empty 200: an empty symbol
list must only ever mean "this file defines nothing".
"""

from __future__ import annotations

import logging

from coding_client.auth.models import User
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from auth.tenant import get_current_user_with_tenant
from processing.symbols import SUPPORTED_LANGUAGES, extract_symbols, resolve_language

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_CODE_BYTES = 2 * 1024 * 1024


class SymbolsRequest(BaseModel):
    code: str = Field(..., max_length=MAX_CODE_BYTES, description="File content (max 2 MiB)")
    language: str | None = Field(None, description=f"One of {', '.join(SUPPORTED_LANGUAGES)}")
    file_path: str | None = Field(
        None, description="Infers the language; marks a Python package __init__.py"
    )


class SymbolEntry(BaseModel):
    name: str
    kind: str
    line: int
    end_line: int
    parent: str | None = None
    bases: list[str] | None = Field(
        None, description="Class-like symbols only: base names as written, type args stripped"
    )


class ImportEntry(BaseModel):
    module: str
    names: list[str]
    line: int


class ReexportEntry(BaseModel):
    name: str = Field(..., description='Re-exported name, or "*" for a star re-export')
    from_module: str | None = Field(None, description="Source module; None when unknown")
    alias: str | None = None
    line: int


class ImplEntry(BaseModel):
    type: str
    trait: str
    line: int


class SymbolsResponse(BaseModel):
    language: str
    parsed: bool
    parse_error_line: int | None = None
    symbols: list[SymbolEntry]
    imports: list[ImportEntry]
    reexports: list[ReexportEntry] = []
    impls: list[ImplEntry] = []


@router.post(
    "/symbols",
    response_model=SymbolsResponse,
    summary="Symbols and imports defined by one file",
    description=(
        "Deterministic tree-sitter extraction of the functions, methods, classes, "
        "interfaces, types, enums, structs, traits, concepts and namespaces one file "
        "defines, plus its imports. Class-like symbols carry `bases` (written base "
        "names) when they have any; `reexports` lists JS/TS `export ... from`, Python "
        "package `__init__.py` from-imports and Rust `pub use`; `impls` lists Rust "
        "`impl Trait for Type`. Send the file content; the language comes from "
        "`language` or the `file_path` extension, and `file_path` also marks a "
        "Python `__init__.py`."
    ),
    responses={
        422: {"description": "Missing or unsupported language, or code over 2 MiB"},
        503: {"description": "Grammar not available for the language"},
    },
    operation_id="extract_symbols",
)
async def extract_file_symbols(
    request: SymbolsRequest,
    user: User = Depends(get_current_user_with_tenant),
) -> SymbolsResponse:
    """Extract symbols and imports from submitted file content. Requires authentication."""
    content = request.code.encode("utf-8")
    if len(content) > MAX_CODE_BYTES:
        raise HTTPException(status_code=422, detail="code exceeds 2 MiB")
    try:
        grammar = resolve_language(request.language, request.file_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = await run_in_threadpool(
        extract_symbols, content, language=grammar, file_path=request.file_path
    )
    if not result["parsed"]:
        logger.warning("symbols: grammar not available for %s", grammar)
        raise HTTPException(status_code=503, detail=f"grammar not available: {grammar}")
    return SymbolsResponse(**result)
