"""
Documentation Generation Utilities Module
Handles template management, content generation, and documentation compilation.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import jinja2

from models.static_analysis_models import (
    DocumentationSection,
    DocumentationType,
)

from ..ast_analysis.function_extractor import FunctionExtractor
from .call_graph_analyzer import CallGraphGenerator
from .code_analyzer import CodeAnalyzer
from .doc_extractor import DocumentationExtractor

logger = logging.getLogger(__name__)


class DocumentationGenerator:
    """Handles documentation generation using templates and content compilation."""

    def __init__(self, templates_dir: str | None = None):
        """Initialize the documentation generator."""
        self.templates_dir = templates_dir or os.path.join(os.path.dirname(__file__), "templates")
        self.jinja_env = self._setup_jinja_environment()
        self.code_analyzer = CodeAnalyzer()
        self.doc_extractor = DocumentationExtractor()

    def _setup_jinja_environment(self) -> jinja2.Environment:
        """Setup Jinja2 environment with templates."""
        # Create templates directory if it doesn't exist
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir)
            self._create_default_templates()

        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

    async def generate_architecture_documentation(
        self, repository_path: str, include_diagrams: bool = True
    ) -> DocumentationSection:
        """Generate architecture documentation with diagrams."""
        try:
            # Generate call graph for architecture visualization
            call_graph_analyzer = CallGraphGenerator()
            call_graph_result = await call_graph_analyzer.generate_comprehensive_call_graph(
                repository_path
            )

            # Analyze project structure
            structure = self.code_analyzer.analyze_project_structure(repository_path)

            # Generate architecture content using template
            template = self.jinja_env.get_template("architecture.md.j2")
            content = template.render(
                project_structure=structure,
                call_graph=call_graph_result.call_graph,
                metrics=call_graph_result.metrics,
                critical_paths=call_graph_result.critical_paths,
            )

            diagrams = []
            if include_diagrams:
                # Note: Diagram generation is handled by format_converter module
                diagrams = ["architecture_diagram_placeholder"]

            return DocumentationSection(content=content, diagrams=diagrams, coverage=0.85)

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error generating architecture documentation: %s", e)
            return DocumentationSection(
                content=f"Error generating architecture documentation: {str(e)}",
                diagrams=[],
                coverage=0.0,
            )

    async def generate_api_reference_documentation(
        self, repository_path: str
    ) -> DocumentationSection:
        """Generate API reference documentation."""
        try:
            # Extract function signatures and API information
            extractor = FunctionExtractor()

            # Find all Python files
            from pathlib import Path

            python_files = list(Path(repository_path).rglob("*.py"))

            api_data: dict[str, list[Any]] = {
                "modules": [],
                "functions": [],
                "classes": [],
            }

            for file_path in python_files[:20]:  # Limit processing
                try:
                    file_analysis = extractor.extract_comprehensive_signatures(
                        str(file_path), repository_path
                    )

                    if "functions" in file_analysis:
                        api_data["functions"].extend(file_analysis["functions"])

                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning("Error analyzing %s for API docs: %s", file_path, e)

            # Generate API reference content using template
            template = self.jinja_env.get_template("api_reference.md.j2")
            content = template.render(api_data=api_data)

            # Calculate coverage based on documented functions
            documented_functions = sum(1 for func in api_data["functions"] if func.get("docstring"))
            total_functions = len(api_data["functions"])
            coverage = documented_functions / total_functions if total_functions > 0 else 0.0

            return DocumentationSection(content=content, diagrams=[], coverage=coverage)

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error generating API reference documentation: %s", e)
            return DocumentationSection(
                content=f"Error generating API reference: {str(e)}",
                diagrams=[],
                coverage=0.0,
            )

    async def generate_change_summary_documentation(
        self, repository_path: str
    ) -> DocumentationSection:
        """Generate change summary documentation."""
        try:
            # Analyze recent changes
            changes = await self.code_analyzer.analyze_recent_changes(repository_path)

            # Generate change summary using template
            template = self.jinja_env.get_template("change_summary.md.j2")
            content = template.render(changes=changes)

            return DocumentationSection(
                content=content,
                diagrams=[],
                coverage=0.9,  # Change summaries are typically well-covered
            )

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error generating change summary: %s", e)
            return DocumentationSection(
                content=f"Error generating change summary: {str(e)}",
                diagrams=[],
                coverage=0.0,
            )

    async def generate_user_guide_documentation(self, repository_path: str) -> DocumentationSection:
        """Generate user guide documentation."""
        try:
            # Extract project data for user guide
            project_data = await self.code_analyzer.generate_project_documentation_data(
                repository_path
            )

            # Find existing documentation files
            readme_content = ""
            try:
                from pathlib import Path

                readme_files = ["README.md", "README.rst", "README.txt"]
                for readme_file in readme_files:
                    readme_path = Path(repository_path) / readme_file
                    if readme_path.exists():
                        with open(readme_path, encoding="utf-8") as f:
                            readme_content = f.read()
                        break
            except (OSError, UnicodeDecodeError):  # fmt: skip
                pass

            # Generate user guide using template
            template = self.jinja_env.get_template("user_guide.md.j2")
            content = template.render(
                project_data=project_data,
                readme_content=readme_content,
                structure=project_data.get("project_structure", {}),
                dependencies=project_data.get("dependencies", {}),
            )

            return DocumentationSection(
                content=content,
                diagrams=[],
                coverage=0.7,
            )

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error generating user guide: %s", e)
            return DocumentationSection(
                content=f"Error generating user guide: {str(e)}",
                diagrams=[],
                coverage=0.0,
            )

    async def generate_project_overview_documentation(
        self, repository_path: str
    ) -> DocumentationSection:
        """Generate project overview documentation."""
        try:
            # Analyze project structure and metadata
            structure = self.code_analyzer.analyze_project_structure(repository_path)
            dependencies = {
                "python": await self.code_analyzer.extract_python_dependencies(repository_path),
                "javascript": await self.code_analyzer.extract_javascript_dependencies(
                    repository_path
                ),
            }

            # Extract key documentation files
            key_docs = {}
            try:
                from pathlib import Path

                doc_files = ["README.md", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE"]
                for doc_file in doc_files:
                    doc_path = Path(repository_path) / doc_file
                    if doc_path.exists():
                        with open(doc_path, encoding="utf-8") as f:
                            key_docs[doc_file] = f.read()[:500]  # First 500 chars
            except (OSError, UnicodeDecodeError):  # fmt: skip
                pass

            # Generate overview using template
            template = self.jinja_env.get_template("project_overview.md.j2")
            content = template.render(
                structure=structure,
                dependencies=dependencies,
                key_docs=key_docs,
            )

            return DocumentationSection(
                content=content,
                diagrams=[],
                coverage=0.8,
            )

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error generating project overview: %s", e)
            return DocumentationSection(
                content=f"Error generating project overview: {str(e)}",
                diagrams=[],
                coverage=0.0,
            )

    async def generate_documentation_by_type(
        self, doc_type: DocumentationType, repository_path: str, **kwargs
    ) -> DocumentationSection:
        """Generate documentation by type."""
        try:
            if doc_type == DocumentationType.ARCHITECTURE:
                return await self.generate_architecture_documentation(
                    repository_path, kwargs.get("include_diagrams", True)
                )
            elif doc_type == DocumentationType.API_REFERENCE:
                return await self.generate_api_reference_documentation(repository_path)
            elif doc_type == DocumentationType.CHANGE_SUMMARY:
                return await self.generate_change_summary_documentation(repository_path)
            elif doc_type == DocumentationType.USER_GUIDE:
                return await self.generate_user_guide_documentation(repository_path)
            else:
                # Default to project overview
                return await self.generate_project_overview_documentation(repository_path)

        except Exception as e:
            logger.exception("Error generating documentation for type %s: %s", doc_type, e)
            return DocumentationSection(
                content=f"Error generating {doc_type.value} documentation: {str(e)}",
                diagrams=[],
                coverage=0.0,
            )

    def _create_default_templates(self):
        """Create default Jinja2 templates."""
        templates = {
            "architecture.md.j2": self._get_architecture_template(),
            "api_reference.md.j2": self._get_api_reference_template(),
            "change_summary.md.j2": self._get_change_summary_template(),
            "user_guide.md.j2": self._get_user_guide_template(),
            "project_overview.md.j2": self._get_project_overview_template(),
        }

        for template_name, template_content in templates.items():
            template_path = os.path.join(self.templates_dir, template_name)
            with open(template_path, "w", encoding="utf-8") as f:
                f.write(template_content)

    def _get_architecture_template(self) -> str:
        """Get architecture documentation template."""
        return """# Architecture Overview

## Project Structure
{% if project_structure.directories %}
### Directories
{% for directory in project_structure.directories %}
- {{ directory }}
{% endfor %}
{% endif %}

## Languages Used
{% for language in project_structure.languages %}
- {{ language }}
{% endfor %}

## Size Metrics
- **Total Files:** {{ project_structure.size_metrics.total_files }}
- **Total Directories:** {{ project_structure.size_metrics.total_directories }}
- **Total Size:** {{ project_structure.size_metrics.total_size_mb }} MB

## File Types
{% for ext, count in project_structure.file_types.items() %}
- **{{ ext or 'No extension' }}:** {{ count }} files
{% endfor %}

## Call Graph Metrics
{% if metrics %}
- **Total Functions:** {{ metrics.total_functions }}
- **Max Call Depth:** {{ metrics.max_call_depth }}
- **Complexity Score:** {{ metrics.complexity_score }}
{% endif %}

## Critical Paths
{% for path in critical_paths %}
- {{ path }}
{% endfor %}

## Architecture Diagrams
{% if diagrams %}
Diagrams available:
{% for diagram in diagrams %}
- {{ diagram }}
{% endfor %}
{% else %}
No architecture diagrams generated.
{% endif %}
"""

    def _get_api_reference_template(self) -> str:
        """Get API reference documentation template."""
        return """# API Reference

## Overview
This document provides a comprehensive reference for all public APIs in the project.

## Functions
{% for function in api_data.functions %}
### {{ function.name }}

**Module:** {{ function.module }}

**Signature:** `{{ function.signature }}`

{% if function.docstring %}
{{ function.docstring }}
{% else %}
*No documentation available*
{% endif %}

**Parameters:**
{% if function.parameters %}
{% for param in function.parameters %}
- `{{ param.name }}` ({{ param.type or 'Any' }}): {{ param.description or 'No description' }}
{% endfor %}
{% else %}
No parameters documented.
{% endif %}

**Returns:** {{ function.return_type or 'Unknown' }}

---
{% endfor %}

## Coverage Statistics
- **Total Functions:** {{ api_data.functions|length }}
- **Documented Functions:** {{ api_data.functions|selectattr('docstring')|list|length }}
- **Coverage:** {{ ((api_data.functions|selectattr('docstring')|list|length / api_data.functions|length) * 100)|round(1) if api_data.functions else 0 }}%
"""

    def _get_change_summary_template(self) -> str:
        """Get change summary documentation template."""
        return """# Change Summary

## Overview
**Current Branch:** {{ changes.current_branch }}
**Analysis Period:** {{ changes.analysis_period }}

{% if changes.change_statistics %}
## Change Statistics
- **Files Changed:** {{ changes.change_statistics.files_changed }}
- **Lines Added:** {{ changes.change_statistics.insertions }}
- **Lines Removed:** {{ changes.change_statistics.deletions }}
{% endif %}

## Recent Commits
{% for commit in changes.recent_commits %}
### {{ commit.hash }} - {{ commit.date }}
**Author:** {{ commit.author }} ({{ commit.email }})
**Message:** {{ commit.message }}

{% endfor %}

## Files Modified
{% if changes.files_changed %}
{% for file in changes.files_changed %}
- `{{ file }}`
{% endfor %}
{% else %}
No files modified in the analysis period.
{% endif %}

## Impact Analysis
{{ changes.impact_analysis }}

{% if changes.error %}
## Analysis Errors
{{ changes.error }}
{% endif %}
"""

    def _get_user_guide_template(self) -> str:
        """Get user guide documentation template."""
        return """# User Guide

## Introduction
Welcome to the project! This guide will help you get started and understand how to use this software.

{% if readme_content %}
## Getting Started (from README)
{{ readme_content[:1000] }}
{% if readme_content|length > 1000 %}
... (see README for full details)
{% endif %}
{% endif %}

## Project Structure
The project is organized as follows:

### Languages
{% for language in structure.languages %}
- **{{ language }}**
{% endfor %}

### Key Directories
{% for directory in structure.directories[:10] %}
- `{{ directory }}`
{% endfor %}
{% if structure.directories|length > 10 %}
... and {{ structure.directories|length - 10 }} more directories
{% endif %}

## Dependencies
{% if dependencies.python %}
### Python Dependencies
{% for dep, info in dependencies.python.items() %}
- **{{ dep }}** {{ info.version }}{% if info.dev %} (dev){% endif %}
{% endfor %}
{% endif %}

{% if dependencies.javascript %}
### JavaScript Dependencies
{% for dep, info in dependencies.javascript.items() %}
- **{{ dep }}** {{ info.version }}{% if info.dev %} (dev){% endif %}
{% endfor %}
{% endif %}

## Installation
Please refer to the project's README or setup documentation for installation instructions.

## Usage
For detailed usage instructions, please refer to the API reference and examples provided in the project documentation.
"""

    def _get_project_overview_template(self) -> str:
        """Get project overview documentation template."""
        return """# Project Overview

## Summary
This document provides a high-level overview of the project structure, dependencies, and key documentation.

## Project Statistics
- **Total Files:** {{ structure.size_metrics.total_files }}
- **Total Size:** {{ structure.size_metrics.total_size_mb }} MB
- **Languages:** {{ structure.languages|join(', ') }}

## File Distribution
{% for ext, count in structure.file_types.items() %}
- **{{ ext or 'No extension' }}:** {{ count }} files
{% endfor %}

## Dependencies Overview
{% if dependencies.python %}
### Python Dependencies
- **Total:** {{ dependencies.python|length }} packages
- **Production:** {{ dependencies.python.values()|rejectattr('dev')|list|length }}
- **Development:** {{ dependencies.python.values()|selectattr('dev')|list|length }}
{% endif %}

{% if dependencies.javascript %}
### JavaScript Dependencies
- **Total:** {{ dependencies.javascript|length }} packages
- **Production:** {{ dependencies.javascript.values()|rejectattr('dev')|list|length }}
- **Development:** {{ dependencies.javascript.values()|selectattr('dev')|list|length }}
{% endif %}

## Key Documentation Files
{% for filename, content in key_docs.items() %}
### {{ filename }}
{{ content }}
{% if content|length >= 500 %}
... (truncated)
{% endif %}
{% endfor %}

## Directory Structure
```
{% for directory in structure.directories[:20] %}
{{ directory }}/
{% endfor %}
{% if structure.directories|length > 20 %}
... and {{ structure.directories|length - 20 }} more directories
{% endif %}
```
"""
