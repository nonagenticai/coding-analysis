"""
Configuration for Python language processing.
"""

from ..base import LanguageConfig, LanguageFeatures, ParserType

# Python configuration
PYTHON_CONFIG = LanguageConfig(
    name="python",
    display_name="Python",
    extensions=[".py", ".pyw"],
    parser_type=ParserType.AST_NATIVE,  # Python has native AST module
    fallback_parsers=[ParserType.REGEX_FALLBACK],
    tree_sitter_grammar="python",
    external_command=None,
    language_server_command=None,
    features=LanguageFeatures(
        classes=True,
        inheritance=True,
        interfaces=False,  # Python uses duck typing, no interfaces
        abstract_classes=True,
        multiple_inheritance=True,
        functions=True,
        lambdas=True,
        closures=True,
        higher_order_functions=True,
        static_typing=True,  # Optional with type hints
        generics=True,  # Type hints support generics
        type_inference=True,
        nullable_types=True,  # Optional type hints
        modules=True,
        packages=True,
        namespaces=False,
        imports=True,
        async_support=True,  # async/await
        coroutines=True,
        annotations=True,  # Type annotations
        decorators=True,
        macros=False,
        garbage_collected=True,
        manual_memory=False,
        reference_counting=True,  # CPython uses reference counting + GC
    ),
    ignore_patterns=[
        "**/__pycache__/**",
        "**/*.pyc",
        "**/*.pyo",
        "**/venv/**",
        "**/env/**",
        "**/.venv/**",
    ],
    test_patterns=[
        "**/test_*.py",
        "**/*_test.py",
        "**/tests/**",
        "**/test/**",
    ],
    required_packages=[],  # No additional packages needed for native AST
    max_file_size_mb=5,
    timeout_seconds=30,
)
