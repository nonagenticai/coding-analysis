"""Configuration settings for Java language processing."""

from ..base import LanguageConfig, LanguageFeatures, ParserType

# Java language configuration
JAVA_CONFIG = LanguageConfig(
    name="java",
    display_name="Java",
    extensions=[".java"],
    parser_type=ParserType.TREE_SITTER,
    fallback_parsers=[ParserType.REGEX_FALLBACK],
    tree_sitter_grammar="java",
    external_command=None,
    language_server_command=None,
    features=LanguageFeatures(
        classes=True,
        inheritance=True,
        interfaces=True,
        abstract_classes=True,
        functions=True,
        static_typing=True,
        generics=True,
        modules=True,
        packages=True,
        annotations=True,
        async_support=False,  # Java has CompletableFuture but not native async/await
        garbage_collected=True,
    ),
    ignore_patterns=["**/target/**", "**/*.class", "**/build/**"],
    test_patterns=["**/*Test.java", "**/Test*.java", "**/src/test/**"],
    required_packages=["tree-sitter", "tree-sitter-java"],
    max_file_size_mb=5,
    timeout_seconds=30,
)
