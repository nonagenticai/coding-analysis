"""
Example usage of the graph database population functionality.

This script demonstrates how to use the optimized batch processing
for populating graph databases with code analysis data.
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_populate_graph_database() -> None:
    """Example demonstrating graph database population with sample data."""
    from services.code_graph.graph_database_populator import (
        get_database_stats,
        populate_graph_database,
        verify_database_integrity,
    )

    container_name = "example_project_12345"
    analysis_data = {
        "nodes": [
            {
                "id": "module_main_py",
                "type": "Module",
                "name": "main",
                "properties": {
                    "name": "main",
                    "file_path": "src/main.py",
                    "docstring": "Main application module",
                },
            },
            {
                "id": "class_user_manager",
                "type": "Class",
                "name": "UserManager",
                "properties": {
                    "name": "UserManager",
                    "file_path": "src/user_manager.py",
                    "docstring": "Manages user operations",
                },
            },
            {
                "id": "func_create_user",
                "type": "Function",
                "name": "create_user",
                "properties": {
                    "name": "create_user",
                    "file_path": "src/user_manager.py",
                    "parameters": ["name", "email"],
                    "docstring": "Creates a new user",
                },
            },
        ],
        "edges": [
            {
                "source_id": "module_main_py",
                "target_id": "class_user_manager",
                "type": "imports",
                "properties": {"import_type": "class"},
            },
            {
                "source_id": "class_user_manager",
                "target_id": "func_create_user",
                "type": "defines",
                "properties": {"method_type": "instance"},
            },
        ],
    }
    try:
        logger.info("🚀 Starting database population example...")
        success = await populate_graph_database(container_name, analysis_data)
        if success:
            logger.info("✅ Database population completed successfully!")
            logger.info("🔍 Verifying database integrity...")
            integrity_result = await verify_database_integrity(container_name)
            logger.info(f"Integrity check result: {integrity_result}")
            logger.info("📊 Getting database statistics...")
            stats = get_database_stats(container_name)
            logger.info(f"Database stats: {stats}")
        else:
            logger.error("❌ Database population failed!")
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        logger.exception("Error details:")


async def example_large_dataset_population() -> None:
    """Example with a larger dataset to demonstrate batch processing efficiency."""
    from services.code_graph.graph_database_populator import (
        populate_graph_database,
    )

    container_name = "large_project_67890"
    nodes = []
    edges = []
    for i in range(100):
        nodes.append(
            {
                "id": f"module_{i}",
                "type": "Module",
                "name": f"module_{i}",
                "properties": {
                    "name": f"module_{i}",
                    "file_path": f"src/modules/module_{i}.py",
                },
            }
        )
        for j in range(5):
            class_id = f"class_{i}_{j}"
            nodes.append(
                {
                    "id": class_id,
                    "type": "Class",
                    "name": f"Class_{j}",
                    "properties": {
                        "name": f"Class_{j}",
                        "file_path": f"src/modules/module_{i}.py",
                    },
                }
            )
            edges.append(
                {
                    "source_id": f"module_{i}",
                    "target_id": class_id,
                    "type": "defines",
                    "properties": {},
                }
            )
            for k in range(3):
                method_id = f"method_{i}_{j}_{k}"
                nodes.append(
                    {
                        "id": method_id,
                        "type": "Function",
                        "name": f"method_{k}",
                        "properties": {
                            "name": f"method_{k}",
                            "file_path": f"src/modules/module_{i}.py",
                        },
                    }
                )
                edges.append(
                    {
                        "source_id": class_id,
                        "target_id": method_id,
                        "type": "defines",
                        "properties": {},
                    }
                )
    analysis_data = {"nodes": nodes, "edges": edges}
    logger.info(f"🎯 Generated sample dataset: {len(nodes)} nodes, {len(edges)} edges")
    import time

    start_time = time.time()
    success = await populate_graph_database(container_name, analysis_data)
    end_time = time.time()
    duration = end_time - start_time
    if success:
        logger.info(f"⚡ Large dataset population completed in {duration:.2f} seconds")
        from services.code_graph.graph_database_populator import (
            get_database_stats,
        )

        stats = get_database_stats(container_name)
        logger.info(f"Final database stats: {stats}")
    else:
        logger.error("❌ Large dataset population failed!")


def run_examples() -> None:
    """Run all examples."""
    logger.info("🎮 Running graph database population examples...")
    asyncio.run(example_populate_graph_database())
    asyncio.run(example_large_dataset_population())
    logger.info("🏁 Examples completed!")


if __name__ == "__main__":
    run_examples()
