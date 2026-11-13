"""FastAPI dependency injection for coding-analysis."""

from __future__ import annotations

from functools import lru_cache

from processing.multi_language_factory import MultiLanguageProcessorFactory


@lru_cache
def get_multi_language_factory() -> MultiLanguageProcessorFactory:
    """Get or create the multi-language processor factory singleton."""
    return MultiLanguageProcessorFactory()
