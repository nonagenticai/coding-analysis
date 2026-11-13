"""
Configuration for JavaScript/TypeScript language processing.
"""

from ..base import LanguageConfig, LanguageFeatures, ParserType

# JavaScript configuration
JAVASCRIPT_CONFIG = LanguageConfig(
    name="javascript",
    display_name="JavaScript",
    extensions=[".js", ".jsx", ".mjs", ".cjs"],
    parser_type=ParserType.TREE_SITTER,
    fallback_parsers=[ParserType.REGEX_FALLBACK],
    tree_sitter_grammar="javascript",
    external_command=None,
    language_server_command=None,
    features=LanguageFeatures(
        classes=True,
        inheritance=True,
        interfaces=False,  # JavaScript uses duck typing
        abstract_classes=False,
        multiple_inheritance=False,
        functions=True,
        lambdas=True,  # Arrow functions
        closures=True,
        higher_order_functions=True,
        static_typing=False,
        generics=False,
        type_inference=True,  # Engine type inference
        nullable_types=True,  # null, undefined
        modules=True,
        packages=True,  # npm packages
        namespaces=False,
        imports=True,  # import/export
        async_support=True,  # async/await
        coroutines=True,
        annotations=False,
        decorators=True,  # Experimental decorators
        macros=False,
        garbage_collected=True,
        manual_memory=False,
        reference_counting=False,
    ),
    ignore_patterns=[
        "**/node_modules/**",
        "**/dist/**",
        "**/build/**",
        "**/*.min.js",
    ],
    test_patterns=[
        "**/*.test.js",
        "**/*.spec.js",
        "**/test/**",
        "**/tests/**",
    ],
    required_packages=["tree-sitter", "tree-sitter-javascript"],
    max_file_size_mb=5,
    timeout_seconds=30,
)

# TypeScript configuration
TYPESCRIPT_CONFIG = LanguageConfig(
    name="typescript",
    display_name="TypeScript",
    extensions=[".ts", ".tsx"],
    parser_type=ParserType.TREE_SITTER,
    fallback_parsers=[ParserType.REGEX_FALLBACK],
    tree_sitter_grammar="typescript",
    external_command=None,
    language_server_command=None,
    features=LanguageFeatures(
        classes=True,
        inheritance=True,
        interfaces=True,
        abstract_classes=True,
        multiple_inheritance=False,
        functions=True,
        lambdas=True,  # Arrow functions
        closures=True,
        higher_order_functions=True,
        static_typing=True,
        generics=True,
        type_inference=True,
        nullable_types=True,
        modules=True,
        packages=True,
        namespaces=True,
        imports=True,
        async_support=True,
        coroutines=True,
        annotations=True,  # Type annotations
        decorators=True,
        macros=False,
        garbage_collected=True,
        manual_memory=False,
        reference_counting=False,
    ),
    ignore_patterns=[
        "**/node_modules/**",
        "**/dist/**",
        "**/build/**",
        "**/*.d.ts",
    ],
    test_patterns=[
        "**/*.test.ts",
        "**/*.spec.ts",
        "**/test/**",
        "**/tests/**",
    ],
    required_packages=["tree-sitter", "tree-sitter-typescript"],
    max_file_size_mb=5,
    timeout_seconds=30,
)
