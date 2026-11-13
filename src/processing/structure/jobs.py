"""Structure extraction jobs, run in an isolated worker (``contract_analysis.isolation``).

Module-level functions with picklable arguments and results; tenant source is
parsed here, never in the API process.
"""

from __future__ import annotations

from .api_surface import api_surface
from .db_tables import TableInventory, table_inventory
from .entrypoints import EntrypointInventory, entrypoint_inventory
from .module_graph import module_graph
from .service_graph import ServiceGraph, build_service_graph
from .types import ApiSurface, ModuleGraph


def module_graph_job(files: dict[str, str], language: str, root: str) -> ModuleGraph:
    return module_graph(files, language, root)


def api_surface_job(files: dict[str, str], language: str, root: str) -> ApiSurface:
    return api_surface(files, language, root)


def tables_job(repo_files: dict[str, dict[str, str]]) -> dict[str, TableInventory]:
    return {repo: table_inventory(files) for repo, files in repo_files.items()}


def entrypoints_job(repo_files: dict[str, dict[str, str]]) -> dict[str, EntrypointInventory]:
    return {repo: entrypoint_inventory(files) for repo, files in repo_files.items()}


def service_graph_job(
    repo_files: dict[str, dict[str, str]], hints: dict[str, str], include_tests: bool
) -> ServiceGraph:
    return build_service_graph(repo_files, hints, include_tests)
