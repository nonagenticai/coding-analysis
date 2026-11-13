"""
Format Conversion and Output Module
Handles diagram generation, file output, and format conversions for documentation.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FormatConverter:
    """Handles format conversion, diagram generation, and file output."""

    def __init__(self, output_dir: str | None = None):
        """Initialize the format converter."""
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="docs_output_")

    async def generate_architecture_diagrams(
        self, repository_path: str, call_graph_result: Any
    ) -> list[str]:
        """Generate architecture diagrams using Graphviz."""
        diagrams = []

        try:
            # Generate call graph diagram
            if hasattr(call_graph_result, "visual_output") and call_graph_result.visual_output:
                diagram_path = await self.save_diagram(
                    call_graph_result.visual_output, "call_graph.svg"
                )
                if diagram_path:
                    diagrams.append(diagram_path)

            # Generate dependency diagram
            dependency_diagram = await self.generate_dependency_diagram(repository_path)
            if dependency_diagram:
                diagrams.append(dependency_diagram)

            # Generate module structure diagram
            structure_diagram = await self.generate_structure_diagram(repository_path)
            if structure_diagram:
                diagrams.append(structure_diagram)

        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Error generating architecture diagrams: %s", e)

        return diagrams

    async def save_diagram(self, base64_content: str, filename: str) -> str | None:
        """Save base64 diagram content to file."""
        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            file_path = os.path.join(self.output_dir, filename)

            # Decode and save
            diagram_data = base64.b64decode(base64_content)
            with open(file_path, "wb") as f:
                f.write(diagram_data)

            return file_path

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error saving diagram: %s", e)
            return None

    async def generate_dependency_diagram(self, repository_path: str) -> str | None:
        """Generate dependency diagram showing project dependencies."""
        try:
            import graphviz

            # Create a directed graph for dependencies
            dot = graphviz.Digraph(comment="Project Dependencies", format="svg", engine="dot")

            # Configure graph appearance
            dot.attr(rankdir="LR", bgcolor="white")
            dot.attr(
                "node",
                shape="box",
                style="rounded,filled",
                fillcolor="lightblue",
            )
            dot.attr("edge", color="gray")

            # Import code analyzer for dependency extraction
            from .code_analyzer import CodeAnalyzer

            analyzer = CodeAnalyzer()

            # Analyze dependencies based on file type
            dependencies = {}

            # Python dependencies
            python_deps = await analyzer.extract_python_dependencies(repository_path)
            if python_deps:
                dependencies["Python"] = python_deps

            # JavaScript/TypeScript dependencies
            js_deps = await analyzer.extract_javascript_dependencies(repository_path)
            if js_deps:
                dependencies["JavaScript"] = js_deps

            # Create nodes for each language and its dependencies
            for language, deps in dependencies.items():
                # Add language node
                dot.node(language, language, shape="ellipse", fillcolor="yellow")

                # Add dependency nodes and edges (limit to prevent overcrowding)
                for i, (dep_name, dep_info) in enumerate(list(deps.items())[:10]):
                    node_id = f"{language}_{dep_name}"
                    label = f"{dep_name}\n{dep_info.get('version', '')}"

                    # Color code by dependency type
                    if dep_info.get("dev", False):
                        color = "lightgray"
                    elif dep_info.get("optional", False):
                        color = "lightyellow"
                    else:
                        color = "lightgreen"

                    dot.node(node_id, label, fillcolor=color)
                    dot.edge(language, node_id)

                # Add note if there are more dependencies
                if len(deps) > 10:
                    more_node = f"{language}_more"
                    dot.node(
                        more_node,
                        f"... and {len(deps) - 10} more",
                        fillcolor="lightcyan",
                        fontsize="10",
                    )
                    dot.edge(language, more_node, style="dashed")

            # Save the diagram
            output_path = os.path.join(self.output_dir, "dependency_diagram")

            # Render the diagram
            dot.render(output_path, cleanup=True)

            return f"{output_path}.svg"

        except ImportError:
            logger.warning("Graphviz not installed, skipping dependency diagram generation")
            return None
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Error generating dependency diagram: %s", e)
            return None

    async def generate_structure_diagram(self, repository_path: str) -> str | None:
        """Generate project structure diagram."""
        try:
            import graphviz

            # Create a directed graph for project structure
            dot = graphviz.Digraph(comment="Project Structure", format="svg", engine="dot")

            # Configure graph appearance
            dot.attr(rankdir="TB", bgcolor="white")
            dot.attr(
                "node",
                shape="folder",
                style="filled",
                fillcolor="lightblue",
            )
            dot.attr("edge", color="gray")

            # Import code analyzer for structure analysis
            from .code_analyzer import CodeAnalyzer

            analyzer = CodeAnalyzer()

            structure = analyzer.analyze_project_structure(repository_path)

            # Add root node
            root_name = Path(repository_path).name
            dot.node("root", root_name, shape="house", fillcolor="gold")

            # Add directory nodes (limit to prevent overcrowding)
            directories = structure.get("directories", [])[:15]
            for directory in directories:
                # Create simplified path for readability
                dir_parts = directory.split(os.sep)
                if len(dir_parts) > 2:
                    display_name = f"{dir_parts[0]}/.../{dir_parts[-1]}"
                else:
                    display_name = directory

                # Color code by directory type
                if any(keyword in directory.lower() for keyword in ["test", "tests"]):
                    color = "lightcoral"
                elif any(keyword in directory.lower() for keyword in ["src", "lib", "app"]):
                    color = "lightgreen"
                elif any(keyword in directory.lower() for keyword in ["doc", "docs"]):
                    color = "lightyellow"
                else:
                    color = "lightblue"

                dot.node(f"dir_{directory}", display_name, fillcolor=color)

                # Connect to parent or root
                parent_parts = dir_parts[:-1]
                if parent_parts:
                    parent_dir = os.sep.join(parent_parts)
                    if parent_dir in directories:
                        dot.edge(f"dir_{parent_dir}", f"dir_{directory}")
                    else:
                        dot.edge("root", f"dir_{directory}")
                else:
                    dot.edge("root", f"dir_{directory}")

            # Add file type summary nodes
            file_types = structure.get("file_types", {})
            for ext, count in list(file_types.items())[:8]:  # Limit file types shown
                if count > 1:  # Only show file types with multiple files
                    ext_name = ext or "no_ext"
                    label = f"{ext_name}\n{count} files"

                    # Color code by file type
                    if ext in [".py"]:
                        color = "lightgreen"
                    elif ext in [".js", ".ts"]:
                        color = "lightyellow"
                    elif ext in [".md", ".rst"]:
                        color = "lightcyan"
                    else:
                        color = "lightgray"

                    dot.node(
                        f"type_{ext_name}",
                        label,
                        shape="note",
                        fillcolor=color,
                        fontsize="10",
                    )

            # Save the diagram
            output_path = os.path.join(self.output_dir, "structure_diagram")

            # Render the diagram
            dot.render(output_path, cleanup=True)

            return f"{output_path}.svg"

        except ImportError:
            logger.warning("Graphviz not installed, skipping structure diagram generation")
            return None
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Error generating structure diagram: %s", e)
            return None

    async def convert_markdown_to_html(self, markdown_content: str) -> str:
        """Convert markdown content to HTML."""
        try:
            # Try to use markdown library if available
            try:
                import markdown

                md = markdown.Markdown(extensions=["extra", "codehilite", "toc"])
                html_content = md.convert(markdown_content)
                return html_content

            except ImportError:
                # Fallback to basic conversion
                return self._basic_markdown_to_html(markdown_content)

        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Error converting markdown to HTML: %s", e)
            return f"<pre>{markdown_content}</pre>"

    def _basic_markdown_to_html(self, markdown_content: str) -> str:
        """Basic markdown to HTML conversion without external dependencies."""
        import re

        html = markdown_content

        # Headers
        html = re.sub(r"^# (.*$)", r"<h1>\1</h1>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.*$)", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"^### (.*$)", r"<h3>\1</h3>", html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)
        html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html)

        # Code blocks
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)

        # Links
        html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)

        # Line breaks
        html = html.replace("\n\n", "</p><p>")
        html = f"<p>{html}</p>"

        return html

    async def save_documentation_files(
        self, documentation: dict[str, Any], base_filename: str = "documentation"
    ) -> list[str]:
        """Save documentation in multiple formats."""
        saved_files = []

        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            # Save as JSON
            json_path = os.path.join(self.output_dir, f"{base_filename}.json")
            import json

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(documentation, f, indent=2, ensure_ascii=False)
            saved_files.append(json_path)

            # Save as Markdown
            md_path = os.path.join(self.output_dir, f"{base_filename}.md")
            md_content = await self._convert_to_markdown(documentation)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            saved_files.append(md_path)

            # Save as HTML
            html_path = os.path.join(self.output_dir, f"{base_filename}.html")
            html_content = await self.convert_markdown_to_html(md_content)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self._wrap_html(html_content, base_filename))
            saved_files.append(html_path)

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error saving documentation files: %s", e)

        return saved_files

    async def _convert_to_markdown(self, documentation: dict[str, Any]) -> str:
        """Convert documentation dictionary to markdown format."""
        md_lines = []

        # Title
        md_lines.append("# Project Documentation")
        md_lines.append("")

        # Process each section
        for section_name, section_data in documentation.items():
            md_lines.append(f"## {section_name.replace('_', ' ').title()}")
            md_lines.append("")

            if isinstance(section_data, dict):
                if "content" in section_data:
                    md_lines.append(section_data["content"])
                else:
                    # Convert dict to readable format
                    for key, value in section_data.items():
                        if isinstance(value, str):
                            md_lines.append(f"**{key}:** {value}")
                        elif isinstance(value, (list, dict)):
                            md_lines.append(f"**{key}:**")
                            md_lines.append("```json")
                            import json

                            md_lines.append(json.dumps(value, indent=2))
                            md_lines.append("```")
                        else:
                            md_lines.append(f"**{key}:** {str(value)}")
            elif isinstance(section_data, str):
                md_lines.append(section_data)
            else:
                md_lines.append(str(section_data))

            md_lines.append("")

        return "\n".join(md_lines)

    def _wrap_html(self, content: str, title: str) -> str:
        """Wrap HTML content with basic HTML structure."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    {content}
</body>
</html>"""

    async def generate_pdf_output(self, html_content: str, filename: str) -> str | None:
        """Generate PDF output from HTML content."""
        try:
            # Try to use weasyprint if available
            try:
                import weasyprint

                pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
                weasyprint.HTML(string=html_content).write_pdf(pdf_path)
                return pdf_path

            except ImportError:
                logger.warning("weasyprint not available, skipping PDF generation")
                return None

        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Error generating PDF: %s", e)
            return None

    def cleanup_temp_files(self):
        """Clean up temporary files and directories."""
        try:
            import shutil

            if self.output_dir.startswith(tempfile.gettempdir()):
                shutil.rmtree(self.output_dir, ignore_errors=True)
        except (OSError, ValueError) as e:
            logger.warning("Error cleaning up temp files: %s", e)
