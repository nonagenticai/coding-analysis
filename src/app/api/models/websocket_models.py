"""WebSocket models stub.

TODO: Implement WebSocket API models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RealtimeAnalysisRequest(BaseModel):
    """Realtime analysis request."""

    code: str
    language: str = "python"
    analysis_type: str = "full"


class RealtimeAnalysisResponse(BaseModel):
    """Realtime analysis response."""

    results: dict[str, Any] = {}
    analysis_type: str = ""


class RealtimeErrorResponse(BaseModel):
    """Realtime error response."""

    error: str = ""
    code: int = 500
