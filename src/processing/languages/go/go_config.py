"""
Configuration for Go language processing.
"""

from ..base import LanguageConfig, LanguageFeatures, ParserType

# Go configuration
GO_CONFIG = LanguageConfig(
    name="go",
    display_name="Go",
    extensions=[".go"],
    parser_type=ParserType.TREE_SITTER,
    fallback_parsers=[ParserType.REGEX_FALLBACK],
    tree_sitter_grammar="go",
    external_command=None,
    language_server_command=None,
    features=LanguageFeatures(
        classes=False,
        inheritance=False,
        interfaces=True,
        abstract_classes=False,
        functions=True,
        static_typing=True,
        generics=True,
        modules=True,
        packages=True,
        annotations=False,
        async_support=True,  # Go has goroutines and channels
        garbage_collected=True,
    ),
    ignore_patterns=["**/vendor/**", "**/*.test", "**/go.sum"],
    test_patterns=["**/*_test.go"],
    required_packages=["tree-sitter", "tree-sitter-go"],
    max_file_size_mb=5,
    timeout_seconds=30,
)
