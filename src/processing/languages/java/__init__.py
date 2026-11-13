"""Java language processing module.

This module provides Java-specific code processing capabilities including
parsing, entity extraction, and relationship analysis.
"""

from .java_config import JAVA_CONFIG
from .java_processor import JavaProcessor

__all__ = ["JavaProcessor", "JAVA_CONFIG"]
