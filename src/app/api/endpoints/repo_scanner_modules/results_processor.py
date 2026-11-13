"""Scanning Results Processing Module.

This module handles the processing, formatting, and aggregation of scan results
from various scanning operations. It provides utilities for result transformation,
statistical analysis, and report generation.

The module provides:
- Result aggregation and summarization
- Statistical analysis of scan data
- Report formatting and structure
- Error consolidation and reporting
- Performance metrics calculation
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def process_scan_results(
    scan_type: str,
    raw_results: dict[str, Any],
    start_time: datetime,
    container_name: str,
    file_count: int | None = None,
) -> dict[str, Any]:
    """Process and format raw scan results into structured output.

    This function takes raw scanning results from various scan types and
    transforms them into a standardized, structured format suitable for
    API responses and further analysis.

    Processing Steps:
    1. Calculate scan duration and performance metrics
    2. Aggregate and summarize scan data
    3. Generate statistical insights
    4. Format results according to scan type
    5. Add metadata and context information

    Args:
        scan_type (str): Type of scan performed ('dependencies', 'imports', 'validation', 'full_scan').
        raw_results (Dict[str, Any]): Raw results from scanning operations.
        start_time (datetime): When the scan operation started.
        container_name (str): Name of the container that was scanned.
        file_count (Optional[int]): Number of files processed, if applicable.

    Returns:
        Dict[str, Any]: Processed results containing:
            - results: Formatted scan-specific results
            - metadata: Scan metadata (duration, timing, container info)
            - statistics: Statistical summary of scan results
            - performance: Performance metrics and timing data
            - errors: Consolidated error information

    Scan Type Processing:
        - dependencies: Aggregates package information, version analysis
        - imports: Processes import relationships, categorization
        - validation: Summarizes validation status, error patterns
        - full_scan: Combines all scan types with cross-references

    Example:
        ```python
        start_time = datetime.now(timezone.utc)
        raw_results = await scan_dependencies("container-name")

        processed = process_scan_results(
            scan_type="dependencies",
            raw_results=raw_results,
            start_time=start_time,
            container_name="container-name",
            file_count=0
        )

        print(f"Scan took {processed['metadata']['duration_seconds']} seconds")
        print(f"Found {processed['statistics']['total_items']} items")
        ```

    Error Handling:
        - Invalid scan types are handled gracefully
        - Missing or malformed data doesn't stop processing
        - Errors are collected and reported in consolidated format
        - Performance metrics are calculated even for failed scans
    """
    end_time = datetime.now(UTC)
    duration_seconds = (end_time - start_time).total_seconds()

    try:
        # Initialize base processed results structure
        processed_results = {
            "results": {},
            "metadata": {
                "scan_type": scan_type,
                "container_name": container_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration_seconds,
                "file_count": file_count or 0,
            },
            "statistics": {},
            "performance": {},
            "errors": [],
        }

        # Process results based on scan type
        if scan_type == "dependencies":
            processed_results.update(_process_dependency_results(raw_results))
        elif scan_type == "imports":
            processed_results.update(_process_import_results(raw_results))
        elif scan_type == "validation":
            processed_results.update(_process_validation_results(raw_results))
        elif scan_type == "full_scan":
            processed_results.update(_process_full_scan_results(raw_results))
        else:
            logger.warning(f"Unknown scan type for processing: {scan_type}")
            processed_results["results"] = raw_results
            processed_results["errors"].append(f"Unknown scan type: {scan_type}")

        # Add performance metrics
        processed_results["performance"] = _calculate_performance_metrics(
            raw_results, duration_seconds, file_count
        )

        # Collect and consolidate errors
        processed_results["errors"].extend(_extract_errors(raw_results))

        logger.info(
            f"Processed {scan_type} scan results: "
            f"duration={duration_seconds:.2f}s, "
            f"items={processed_results['statistics'].get('total_items', 0)}"
        )

        return processed_results

    except Exception as e:
        logger.error(f"Error processing {scan_type} scan results: {e}")
        # Return basic structure even if processing fails
        return {
            "results": raw_results,
            "metadata": {
                "scan_type": scan_type,
                "container_name": container_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration_seconds,
                "file_count": file_count or 0,
            },
            "statistics": {"total_items": 0},
            "performance": {"items_per_second": 0},
            "errors": [f"Processing error: {str(e)}"],
        }


def _process_dependency_results(raw_results: dict[str, Any]) -> dict[str, Any]:
    """Process dependency scan results with enhanced analysis.

    Args:
        raw_results: Raw dependency scan data.

    Returns:
        Dict containing processed dependency information with statistics.
    """
    dependencies = raw_results.get("dependencies", [])

    # Group by source
    sources = {}
    for dep in dependencies:
        source = dep.get("source", "unknown")
        if source not in sources:
            sources[source] = []
        sources[source].append(dep)

    # Analyze versions
    version_patterns = _analyze_version_patterns(dependencies)

    return {
        "results": {
            "dependencies": dependencies,
            "sources": sources,
            "version_analysis": version_patterns,
        },
        "statistics": {
            "total_items": len(dependencies),
            "total_dependencies": raw_results.get("dependency_count", 0),
            "sources_found": len(sources),
            "invalid_dependencies": len(raw_results.get("invalid_dependencies", [])),
        },
    }


def _process_import_results(raw_results: dict[str, Any]) -> dict[str, Any]:
    """Process import scan results with relationship analysis.

    Args:
        raw_results: Raw import scan data.

    Returns:
        Dict containing processed import information with analysis.
    """
    imports = raw_results.get("imports", [])

    # Categorize imports
    import_categories = _categorize_imports(imports)

    # Analyze import patterns
    import_patterns = _analyze_import_patterns(imports)

    return {
        "results": {
            "imports": imports,
            "categories": import_categories,
            "patterns": import_patterns,
        },
        "statistics": {
            "total_items": len(imports),
            "total_imports": raw_results.get("import_count", 0),
            "invalid_imports": len(raw_results.get("invalid_imports", [])),
            "unique_modules": len(set(imp.get("module", "") for imp in imports)),
        },
    }


def _process_validation_results(raw_results: dict[str, Any]) -> dict[str, Any]:
    """Process validation results with error analysis.

    Args:
        raw_results: Raw validation scan data.

    Returns:
        Dict containing processed validation information with insights.
    """
    validation_results = raw_results.get("validation_results", [])

    # Analyze issues and warnings
    issue_analysis = _analyze_validation_issues(validation_results)

    # Calculate quality metrics
    quality_metrics = _calculate_quality_metrics(validation_results)

    return {
        "results": {
            "validation_results": validation_results,
            "issue_analysis": issue_analysis,
            "quality_metrics": quality_metrics,
        },
        "statistics": {
            "total_items": len(validation_results),
            "valid_files": raw_results.get("valid_files", 0),
            "invalid_files": raw_results.get("invalid_files", 0),
            "total_lines": sum(result.get("line_count", 0) for result in validation_results),
        },
    }


def _process_full_scan_results(raw_results: dict[str, Any]) -> dict[str, Any]:
    """Process comprehensive full scan results.

    Args:
        raw_results: Raw full scan data containing all scan types.

    Returns:
        Dict containing processed comprehensive scan information.
    """
    # Process each scan type individually
    dependencies_data = _process_dependency_results(raw_results.get("dependencies", {}))
    imports_data = _process_import_results(raw_results.get("imports", {}))
    validation_data = _process_validation_results(raw_results.get("validation", {}))

    # Create cross-references and correlations
    correlations = _analyze_cross_scan_correlations(
        dependencies_data, imports_data, validation_data
    )

    return {
        "results": {
            "dependencies": dependencies_data["results"],
            "imports": imports_data["results"],
            "validation": validation_data["results"],
            "correlations": correlations,
            "summary": raw_results.get("summary", {}),
        },
        "statistics": {
            "total_items": (
                dependencies_data["statistics"]["total_items"]
                + imports_data["statistics"]["total_items"]
                + validation_data["statistics"]["total_items"]
            ),
            "dependencies": dependencies_data["statistics"],
            "imports": imports_data["statistics"],
            "validation": validation_data["statistics"],
        },
    }


def _analyze_version_patterns(dependencies: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze version patterns in dependencies.

    Args:
        dependencies: List of dependency dictionaries.

    Returns:
        Dict containing version pattern analysis.
    """
    version_types = {"exact": 0, "range": 0, "unknown": 0}

    for dep in dependencies:
        version = dep.get("version", "")
        if "==" in version:
            version_types["exact"] += 1
        elif any(op in version for op in [">=", "<=", ">", "<", "~="]):
            version_types["range"] += 1
        else:
            version_types["unknown"] += 1

    return {
        "version_types": version_types,
        "total_versioned": sum(version_types.values()),
    }


def _categorize_imports(imports: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Categorize imports by type and source.

    Args:
        imports: List of import dictionaries.

    Returns:
        Dict with categorized import lists.
    """
    categories = {
        "standard_library": [],
        "third_party": [],
        "local": [],
        "relative": [],
    }

    for imp in imports:
        module = imp.get("module", "")
        if not module:
            continue

        # Simple categorization logic
        if module.startswith("."):
            categories["relative"].append(module)
        elif module in ["os", "sys", "json", "re", "datetime", "pathlib"]:
            categories["standard_library"].append(module)
        elif "/" in module or module.startswith(("app", "src")):
            categories["local"].append(module)
        else:
            categories["third_party"].append(module)

    return categories


def _analyze_import_patterns(imports: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze patterns in import usage.

    Args:
        imports: List of import dictionaries.

    Returns:
        Dict containing import pattern analysis.
    """
    patterns = {
        "import_types": {"import": 0, "from_import": 0},
        "aliased_imports": 0,
        "most_imported_modules": {},
    }

    for imp in imports:
        import_type = imp.get("import_type", "")
        if import_type in patterns["import_types"]:
            patterns["import_types"][import_type] += 1

        if imp.get("alias"):
            patterns["aliased_imports"] += 1

        module = imp.get("module", "")
        if module:
            patterns["most_imported_modules"][module] = (
                patterns["most_imported_modules"].get(module, 0) + 1
            )

    # Get top 10 most imported modules
    patterns["most_imported_modules"] = dict(
        sorted(
            patterns["most_imported_modules"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
    )

    return patterns


def _analyze_validation_issues(
    validation_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Analyze validation issues and patterns.

    Args:
        validation_results: List of validation result dictionaries.

    Returns:
        Dict containing issue analysis.
    """
    analysis = {
        "issue_types": {},
        "warning_types": {},
        "files_with_issues": 0,
        "total_issues": 0,
        "total_warnings": 0,
    }

    for result in validation_results:
        issues = result.get("issues", [])
        warnings = result.get("warnings", [])

        if issues:
            analysis["files_with_issues"] += 1
            analysis["total_issues"] += len(issues)

        analysis["total_warnings"] += len(warnings)

        # Categorize issue types
        for issue in issues:
            issue_type = "SyntaxError" if "SyntaxError" in issue else "Other"
            analysis["issue_types"][issue_type] = analysis["issue_types"].get(issue_type, 0) + 1

        # Categorize warning types
        for warning in warnings:
            warning_type = "LineLength" if "Line too long" in warning else "Other"
            analysis["warning_types"][warning_type] = (
                analysis["warning_types"].get(warning_type, 0) + 1
            )

    return analysis


def _calculate_quality_metrics(
    validation_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate code quality metrics from validation results.

    Args:
        validation_results: List of validation result dictionaries.

    Returns:
        Dict containing quality metrics.
    """
    if not validation_results:
        return {"quality_score": 0, "metrics": {}}

    total_files = len(validation_results)
    valid_files = sum(1 for result in validation_results if result.get("is_valid", False))
    total_lines = sum(result.get("line_count", 0) for result in validation_results)

    quality_score = (valid_files / total_files) * 100 if total_files > 0 else 0

    return {
        "quality_score": round(quality_score, 2),
        "metrics": {
            "syntax_validity_rate": quality_score,
            "average_file_size": total_lines / total_files if total_files > 0 else 0,
            "files_analyzed": total_files,
            "total_lines_analyzed": total_lines,
        },
    }


def _analyze_cross_scan_correlations(
    dependencies_data: dict[str, Any],
    imports_data: dict[str, Any],
    validation_data: dict[str, Any],
) -> dict[str, Any]:
    """Analyze correlations between different scan types.

    Args:
        dependencies_data: Processed dependency scan data.
        imports_data: Processed import scan data.
        validation_data: Processed validation scan data.

    Returns:
        Dict containing correlation analysis.
    """
    correlations: dict[str, dict[str, Any] | int | float] = {
        "dependency_import_correlation": {},
        "validation_import_correlation": {},
        "overall_health_score": 0,
    }

    # Simple correlation analysis
    dep_count = dependencies_data["statistics"].get("total_dependencies", 0)
    import_count = imports_data["statistics"].get("total_imports", 0)
    valid_files = validation_data["statistics"].get("valid_files", 0)
    total_files = validation_data["statistics"].get("total_items", 1)

    # Calculate basic health score
    syntax_score = (valid_files / total_files) * 100 if total_files > 0 else 0
    dependency_score = min(dep_count * 2, 100)  # Simple scoring
    import_score = min(import_count * 0.5, 100)  # Simple scoring

    overall_health = (syntax_score + dependency_score + import_score) / 3

    correlations["overall_health_score"] = round(overall_health, 2)
    correlations["dependency_import_correlation"] = {
        "dependencies_per_import": dep_count / import_count if import_count > 0 else 0
    }

    return correlations


def _calculate_performance_metrics(
    raw_results: dict[str, Any], duration_seconds: float, file_count: int | None
) -> dict[str, Any]:
    """Calculate performance metrics for the scan operation.

    Args:
        raw_results: Raw scan results.
        duration_seconds: Scan duration in seconds.
        file_count: Number of files processed.

    Returns:
        Dict containing performance metrics.
    """
    metrics = {
        "duration_seconds": duration_seconds,
        "files_per_second": 0,
        "items_per_second": 0,
    }

    if file_count and file_count > 0:
        metrics["files_per_second"] = file_count / duration_seconds

    # Try to calculate items per second based on scan content
    total_items = 0
    if "dependencies" in raw_results:
        total_items += len(raw_results["dependencies"])
    if "imports" in raw_results:
        total_items += len(raw_results["imports"])
    if "validation_results" in raw_results:
        total_items += len(raw_results["validation_results"])

    if total_items > 0:
        metrics["items_per_second"] = total_items / duration_seconds

    return metrics


def _extract_errors(raw_results: dict[str, Any]) -> list[str]:
    """Extract and consolidate errors from raw scan results.

    Args:
        raw_results: Raw scan results that may contain errors.

    Returns:
        List of error messages found in the results.
    """
    errors = []

    # Check for direct error field
    if "error" in raw_results:
        errors.append(raw_results["error"])

    # Check for validation errors
    if "validation_error" in raw_results:
        errors.append(raw_results["validation_error"])

    # Check validation results for individual file errors
    validation_results = raw_results.get("validation_results", [])
    for result in validation_results:
        if "error" in result:
            errors.append(f"File {result.get('file_path', 'unknown')}: {result['error']}")

    return errors


def generate_scan_summary(processed_results: dict[str, Any]) -> dict[str, Any]:
    """Generate a high-level summary of scan results.

    This function creates a concise summary suitable for dashboards,
    reports, and quick overview displays.

    Args:
        processed_results: Fully processed scan results from process_scan_results.

    Returns:
        Dict[str, Any]: Summary containing:
            - overview: High-level scan overview
            - key_metrics: Most important metrics
            - recommendations: Suggested actions based on results
            - status: Overall scan status (success, warning, error)

    Example:
        ```python
        processed = process_scan_results(...)
        summary = generate_scan_summary(processed)

        print(f"Scan Status: {summary['status']}")
        print(f"Key Issues: {summary['key_metrics']['issues_found']}")
        for rec in summary['recommendations']:
            print(f"- {rec}")
        ```
    """
    try:
        metadata = processed_results.get("metadata", {})
        statistics = processed_results.get("statistics", {})
        errors = processed_results.get("errors", [])

        # Determine overall status
        if errors:
            status = "error"
        elif statistics.get("invalid_files", 0) > 0 or statistics.get("invalid_imports", 0) > 0:
            status = "warning"
        else:
            status = "success"

        # Generate key metrics
        key_metrics = {
            "scan_type": metadata.get("scan_type", "unknown"),
            "duration": f"{metadata.get('duration_seconds', 0):.2f}s",
            "items_processed": statistics.get("total_items", 0),
            "issues_found": len(errors),
        }

        # Add scan-specific metrics
        if "dependencies" in statistics:
            key_metrics["dependencies_found"] = statistics["dependencies"].get(
                "total_dependencies", 0
            )
        if "imports" in statistics:
            key_metrics["imports_analyzed"] = statistics["imports"].get("total_imports", 0)
        if "validation" in statistics:
            key_metrics["files_validated"] = statistics["validation"].get("total_items", 0)

        # Generate recommendations
        recommendations = _generate_recommendations(processed_results)

        return {
            "overview": {
                "scan_type": metadata.get("scan_type"),
                "container": metadata.get("container_name"),
                "timestamp": metadata.get("end_time"),
                "status": status,
            },
            "key_metrics": key_metrics,
            "recommendations": recommendations,
            "status": status,
        }

    except Exception as e:
        logger.error(f"Error generating scan summary: {e}")
        return {
            "overview": {"status": "error"},
            "key_metrics": {},
            "recommendations": ["Review scan results for errors"],
            "status": "error",
        }


def _generate_recommendations(processed_results: dict[str, Any]) -> list[str]:
    """Generate actionable recommendations based on scan results.

    Args:
        processed_results: Processed scan results.

    Returns:
        List of recommendation strings.
    """
    recommendations = []
    statistics = processed_results.get("statistics", {})
    errors = processed_results.get("errors", [])

    # Error-based recommendations
    if errors:
        recommendations.append("Review and fix scan errors before proceeding")

    # Dependency recommendations
    if "dependencies" in statistics:
        dep_stats = statistics["dependencies"]
        if dep_stats.get("invalid_dependencies", 0) > 0:
            recommendations.append("Review invalid dependencies and fix requirements")

    # Import recommendations
    if "imports" in statistics:
        import_stats = statistics["imports"]
        if import_stats.get("invalid_imports", 0) > 0:
            recommendations.append("Fix invalid import statements")

    # Validation recommendations
    if "validation" in statistics:
        val_stats = statistics["validation"]
        if val_stats.get("invalid_files", 0) > 0:
            recommendations.append("Fix syntax errors in Python files")

    # Performance recommendations
    performance = processed_results.get("performance", {})
    if performance.get("duration_seconds", 0) > 60:
        recommendations.append("Consider optimizing scan performance for large codebases")

    if not recommendations:
        recommendations.append("No issues found - codebase appears healthy")

    return recommendations
