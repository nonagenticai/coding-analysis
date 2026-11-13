""".vulture whitelist module."""

# Vulture whitelist for processing module
# Abstract base classes and their methods are intended to be inherited

# code_document_processor.py
logger  # Standard logger pattern  # type: ignore[unresolved-reference]
CodeDocumentProcessor  # Abstract base class  # type: ignore[unresolved-reference]
language  # Used by subclasses  # type: ignore[unresolved-reference]
process_document  # Abstract method  # type: ignore[unresolved-reference]
code  # Optional parameter used by implementations  # type: ignore[unresolved-reference]
get_supported_extensions  # Abstract method  # type: ignore[unresolved-reference]
_get_module_name  # Protected method used by subclasses  # type: ignore[unresolved-reference]
