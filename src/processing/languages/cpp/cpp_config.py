"""
Configuration for C/C++ language processing.
"""

from ..base import LanguageConfig, LanguageFeatures, ParserType

# C++ configuration
CPP_CONFIG = LanguageConfig(
    name="cpp",
    display_name="C++",
    extensions=[
        ".cpp",
        ".cc",
        ".cxx",
        ".c++",
        ".hpp",
        ".hh",
        ".hxx",
        ".h++",
        ".c",
        ".h",
    ],
    parser_type=ParserType.TREE_SITTER,
    fallback_parsers=[ParserType.REGEX_FALLBACK],
    tree_sitter_grammar="cpp",
    external_command=None,
    language_server_command=None,
    features=LanguageFeatures(
        classes=True,
        inheritance=True,
        interfaces=False,  # C++ has pure virtual classes instead
        abstract_classes=True,
        multiple_inheritance=True,
        functions=True,
        lambdas=True,
        closures=True,
        higher_order_functions=True,
        static_typing=True,
        generics=True,  # Templates
        type_inference=True,  # auto keyword
        nullable_types=False,  # Pointers can be null but no nullable types
        modules=True,  # C++20 modules
        packages=False,
        namespaces=True,
        imports=True,  # #include
        async_support=False,  # No native async/await (but has std::async)
        coroutines=True,  # C++20 coroutines
        annotations=False,
        decorators=False,
        macros=True,
        garbage_collected=False,
        manual_memory=True,
        reference_counting=False,
    ),
    ignore_patterns=[
        "**/build/**",
        "**/cmake-build-*/**",
        "**/*.o",
        "**/*.obj",
        "**/*.a",
        "**/*.lib",
        "**/*.exe",
    ],
    test_patterns=[
        "**/*_test.cpp",
        "**/test_*.cpp",
        "**/tests/**",
        "**/*Test.cpp",
    ],
    required_packages=["tree-sitter", "tree-sitter-cpp"],
    max_file_size_mb=10,  # C++ files can be larger
    timeout_seconds=45,  # C++ parsing can be slower
)
