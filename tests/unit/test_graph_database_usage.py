"""Unit tests for services/code_graph/examples/graph_database_usage.py.

This is example code but still needs test coverage. Tests verify the
example functions run correctly with mocked dependencies.
"""

from __future__ import annotations

from unittest.mock import patch

import services.code_graph.examples.graph_database_usage as _usage_mod
import services.code_graph.graph_database_populator as _populator_mod

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExamplePopulateGraphDatabase:
    """Tests for example_populate_graph_database."""

    @patch.object(_populator_mod, "get_database_stats")
    @patch.object(_populator_mod, "verify_database_integrity")
    @patch.object(_populator_mod, "populate_graph_database")
    async def test_successful_population(self, mock_populate, mock_verify, mock_stats):
        """Should run through success path when population succeeds."""
        mock_populate.return_value = True
        mock_verify.return_value = {"valid": True}
        mock_stats.return_value = {"nodes": 3, "edges": 2}

        from services.code_graph.examples.graph_database_usage import (
            example_populate_graph_database,
        )

        await example_populate_graph_database()

        mock_populate.assert_called_once()
        mock_verify.assert_called_once()
        mock_stats.assert_called_once()

    @patch.object(_populator_mod, "populate_graph_database")
    async def test_failed_population(self, mock_populate):
        """Should handle failure gracefully."""
        mock_populate.return_value = False

        from services.code_graph.examples.graph_database_usage import (
            example_populate_graph_database,
        )

        # Should not raise
        await example_populate_graph_database()

    @patch.object(_populator_mod, "populate_graph_database")
    async def test_exception_handling(self, mock_populate):
        """Should handle exceptions gracefully."""
        mock_populate.side_effect = RuntimeError("DB down")

        from services.code_graph.examples.graph_database_usage import (
            example_populate_graph_database,
        )

        # Should not raise
        await example_populate_graph_database()


class TestExampleLargeDatasetPopulation:
    """Tests for example_large_dataset_population."""

    @patch.object(_populator_mod, "get_database_stats")
    @patch.object(_populator_mod, "populate_graph_database")
    async def test_successful_large_dataset(self, mock_populate, mock_stats):
        """Should handle large dataset population."""
        mock_populate.return_value = True
        mock_stats.return_value = {"nodes": 2100, "edges": 800}

        from services.code_graph.examples.graph_database_usage import (
            example_large_dataset_population,
        )

        await example_large_dataset_population()

        mock_populate.assert_called_once()
        # Verify it generated the expected number of nodes/edges
        call_args = mock_populate.call_args
        analysis_data = call_args[0][1]
        # 100 modules + 100*5 classes + 100*5*3 methods = 100 + 500 + 1500 = 2100 nodes
        assert len(analysis_data["nodes"]) == 2100

    @patch.object(_populator_mod, "populate_graph_database")
    async def test_failed_large_dataset(self, mock_populate):
        """Should handle failure in large dataset population."""
        mock_populate.return_value = False

        from services.code_graph.examples.graph_database_usage import (
            example_large_dataset_population,
        )

        await example_large_dataset_population()


class TestRunExamples:
    """Tests for run_examples."""

    @patch.object(_usage_mod, "example_large_dataset_population")
    @patch.object(_usage_mod, "example_populate_graph_database")
    def test_run_examples(self, mock_populate, mock_large):
        """run_examples should call both example functions via asyncio.run."""
        # The functions are coroutines so asyncio.run needs them to return
        mock_populate.return_value = None
        mock_large.return_value = None

        from services.code_graph.examples.graph_database_usage import run_examples

        run_examples()

        mock_populate.assert_called_once()
        mock_large.assert_called_once()
