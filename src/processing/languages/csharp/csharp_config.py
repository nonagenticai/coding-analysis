"""
Configuration for C# language processing.
"""

from ..base import LanguageConfig, LanguageFeatures, ParserType

# C# configuration
CSHARP_CONFIG = LanguageConfig(
    name="csharp",
    display_name="C#",
    extensions=[".cs"],
    parser_type=ParserType.TREE_SITTER,
    fallback_parsers=[ParserType.REGEX_FALLBACK],
    tree_sitter_grammar="c_sharp",
    external_command=None,
    language_server_command=None,
    features=LanguageFeatures(
        classes=True,
        inheritance=True,
        interfaces=True,
        abstract_classes=True,
        multiple_inheritance=False,  # C# has single inheritance + interfaces
        functions=True,
        lambdas=True,
        closures=True,
        higher_order_functions=True,
        static_typing=True,
        generics=True,
        type_inference=True,  # var keyword
        nullable_types=True,
        modules=False,
        packages=False,
        namespaces=True,
        imports=True,
        async_support=True,  # async/await
        coroutines=True,
        annotations=True,  # attributes
        decorators=False,
        macros=False,
        garbage_collected=True,
        manual_memory=False,
        reference_counting=False,
    ),
    ignore_patterns=[
        "**/bin/**",
        "**/obj/**",
        "**/*.dll",
        "**/*.exe",
        "**/packages/**",
    ],
    test_patterns=[
        "**/*Test.cs",
        "**/Test*.cs",
        "**/Tests/**",
        "**/*Tests.cs",
    ],
    required_packages=["tree-sitter", "tree-sitter-c-sharp"],
    max_file_size_mb=5,
    timeout_seconds=30,
)
