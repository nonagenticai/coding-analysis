""".vulture whitelist module."""

# Vulture whitelist for analyzers module
# These are abstract methods that must be implemented

# JavaScriptCodeProcessor is used by analyzer_factory.py
JavaScriptCodeProcessor  # type: ignore[unresolved-reference]

# Abstract methods from BaseCodeProcessor interface
get_language_name  # type: ignore[unresolved-reference]
analyze_directory  # type: ignore[unresolved-reference]

# Pytest discovers this function automatically
test_static_analysis_integration  # type: ignore[unresolved-reference]

# FileAnalyzer is used by incremental_updater.py
FileAnalyzer  # type: ignore[unresolved-reference]

# TypeSystemAnalyzer and DataFlowAnalyzer are used in semantic_analyzer.py
TypeSystemAnalyzer  # type: ignore[unresolved-reference]
DataFlowAnalyzer  # type: ignore[unresolved-reference]
