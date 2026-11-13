"""Common response models.

TODO: Consolidate with app.api.models response models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LegacyBaseResponse(BaseModel):
    """Legacy base response model for backward compatibility.

    TODO: Migrate all usages to app.api.models.common_api_models.BaseResponse.
    """

    success: bool = True
    message: str = ""
    data: dict[str, Any] | None = None
    errors: list[str] = []
