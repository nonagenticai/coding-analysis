"""Computed structural facts about a repository: module graph, API surface, inventories.

Facts only (no recommendations, no LLM). Each module takes ``{repo-relative path:
source}`` and returns plain dataclasses, so the API layer can run it in an
isolated worker on tenant source (``processing.contract_analysis.isolation``).
"""
