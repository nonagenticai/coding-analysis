"""Configuration settings for Rust language processing."""

from ..base import LanguageConfig, LanguageFeatures, ParserType

# Rust language configuration
RUST_CONFIG = LanguageConfig(
    name="rust",
    display_name="Rust",
    extensions=[".rs"],
    parser_type=ParserType.TREE_SITTER,
    fallback_parsers=[ParserType.REGEX_FALLBACK],
    tree_sitter_grammar="rust",
    external_command=None,
    language_server_command=None,
    features=LanguageFeatures(
        # Object-oriented features (limited in Rust)
        classes=False,  # Rust uses structs instead
        inheritance=False,  # Rust uses traits for behavior sharing
        interfaces=False,  # Rust uses traits
        abstract_classes=False,
        multiple_inheritance=False,
        # Functional features (strong in Rust)
        functions=True,
        lambdas=True,  # Closures
        closures=True,
        higher_order_functions=True,
        # Type system features (very strong in Rust)
        static_typing=True,
        generics=True,
        type_inference=True,
        nullable_types=True,  # Option<T>
        # Module system features
        modules=True,
        packages=True,  # Crates
        namespaces=False,  # Uses modules
        imports=True,  # use statements
        # Advanced features
        async_support=True,  # async/await
        coroutines=True,
        annotations=True,  # Attributes
        decorators=True,  # Procedural macros
        macros=True,  # Declarative and procedural macros
        # Memory management (unique to Rust)
        garbage_collected=False,
        manual_memory=True,  # Ownership system
        reference_counting=False,  # No traditional GC
    ),
    ignore_patterns=["**/target/**", "**/*.rlib", "**/Cargo.lock"],
    test_patterns=["**/tests/**/*.rs", "**/*_test.rs", "**/test_*.rs"],
    required_packages=["tree-sitter", "tree-sitter-rust"],
    max_file_size_mb=10,  # Rust files can be larger
    timeout_seconds=45,  # Rust parsing may take longer
)
