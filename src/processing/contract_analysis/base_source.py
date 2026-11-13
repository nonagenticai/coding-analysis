"""Where an outbound call's base URL comes from: a literal, an env var, or a config key.

A call site's ``base`` is the expression text that was stripped off the front of
the target (``f"{B_URL}/items"`` -> ``B_URL``). On its own that names a variable,
not a service. This module classifies it, from the file's own text only:

* ``literal``: the call target is an absolute URL (``host`` is set), or the base
  is a name bound once in the file to a string literal URL;
* ``env``: ``os.environ[...]`` / ``os.getenv(...)`` / ``os.environ.get(...)``
  (Python) or ``std::env::var(...)`` / ``env::var(...)`` / ``env!(...)`` (Rust),
  either inline or through a name bound once in the file; ``name`` is the env var
  NAME. A literal default URL (``os.getenv("X", "http://svc-b:8080")``) is
  reported as ``default`` -- it names the service a deployment usually points at;
* ``config``: an attribute read off a settings/config object
  (``settings.svc_b_url``, ``self.config.base_url``); ``name`` is the last
  attribute (the config key);
* ``unknown``: anything else (a parameter, a call result); ``name`` is the
  expression as written.

Never guessed: a name bound more than once is ``unknown``. All patterns are
linear regexes over bounded text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ENV_PY = re.compile(
    r"""os\.(?:getenv|environ\.get)\(\s*['"]([A-Za-z_][\w.-]*)['"]\s*(?:,\s*(['"])([^'"]*)\2)?"""
    r"""|os\.environ\[\s*['"]([A-Za-z_][\w.-]*)['"]\s*\]"""
)
_ENV_RS = re.compile(
    r"""(?:std::)?env::var(?:_os)?\(\s*"([A-Za-z_][\w.-]*)"\s*\)"""
    r"""|(?:option_)?env!\(\s*"([A-Za-z_][\w.-]*)"\s*\)"""
    r"""|dotenvy?::var\(\s*"([A-Za-z_][\w.-]*)"\s*\)"""
)
_ENV_JS = re.compile(
    r"""(?:process\.env|import\.meta\.env)(?:\.([A-Za-z_]\w*)|\[\s*['"]([A-Za-z_][\w.-]*)['"]\s*\])"""
)
#: ``unwrap_or("http://...")`` / ``unwrap_or_else(|_| "http://...".into())`` after env::var.
_RS_DEFAULT = re.compile(r"""unwrap_or(?:_else)?\([^"\n]{0,40}"([^"\n]{1,300})\"""")
_CONFIG_ROOT = re.compile(r"(?i)(^|\.)(settings|config|cfg|conf|configuration|options|opts)\b")
_ATTR_CHAIN = re.compile(r"^[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+$")
_NAME = re.compile(r"^[A-Za-z_]\w*$")
_URL = re.compile(r"^(?:https?|wss?)://([^/?#\s]*)", re.I)
_MAX_TEXT = 400


@dataclass(frozen=True)
class BaseSource:
    kind: str  # literal | env | config | unknown
    name: str | None = None  # env var name, config key, or the expression
    default: str | None = None  # literal default URL of an env lookup


def _env_in(expr: str, language: str) -> BaseSource | None:
    if language == "rust":
        m = _ENV_RS.search(expr)
        if m is None:
            return None
        dflt = _RS_DEFAULT.search(expr, m.end())
        return BaseSource("env", next(g for g in m.groups() if g), dflt.group(1) if dflt else None)
    m = _ENV_PY.search(expr)
    if m is None:
        js = _ENV_JS.search(expr)
        return BaseSource("env", js.group(1) or js.group(2)) if js else None
    return BaseSource("env", m.group(1) or m.group(4), m.group(3))


_BINDING_RS = re.compile(
    r"(?m)^[ \t]*(?:pub(?:\([^)\n]{0,40}\))?[ \t]+)?(?:let(?:[ \t]+mut)?|const|static)[ \t]+"
    rf"([A-Za-z_]\w*)\b[^=\n]{{0,120}}=(?!=)[ \t]*([^\n]{{0,{_MAX_TEXT}}})"
)
_BINDING_PY = re.compile(
    r"(?m)^[ \t]*(?:self\.)?([A-Za-z_]\w*)[ \t]*(?::[^=\n]{0,120})?=(?!=)"
    rf"[ \t]*([^\n]{{0,{_MAX_TEXT}}})"
)


def binding_index(source: str, language: str) -> dict[str, list[str]]:
    """``{name: right-hand side of each binding}`` of the whole file, in ONE scan.

    Per-name scans (one full-file regex per call site) made a file with N call
    sites cost O(N x file size). Every pattern is anchored at a line start with
    bounded repeats that never cross a newline.
    """
    pattern = _BINDING_RS if language == "rust" else _BINDING_PY
    out: dict[str, list[str]] = {}
    for m in pattern.finditer(source):
        out.setdefault(m.group(1), []).append(m.group(2))
    return out


class BaseClassifier:
    """Classifies the call sites of ONE file; the binding index is built once, lazily."""

    def __init__(self, source: str, language: str) -> None:
        self.source = source
        self.language = language
        self._index: dict[str, list[str]] | None = None

    def bindings(self, name: str) -> list[str]:
        if self._index is None:
            self._index = binding_index(self.source, self.language)
        return self._index.get(name, [])

    def classify(self, base: str | None, host: str | None) -> BaseSource | None:
        """Classify a call site's base URL (``None`` when it has neither base nor host)."""
        language = self.language
        if base is None:
            return BaseSource("literal", host) if host else None
        expr = base.strip()[:_MAX_TEXT]
        inline = _env_in(expr, language)
        if inline is not None:
            return inline
        bare = expr.removeprefix("&").removeprefix("self.")
        if _NAME.match(bare):
            bound = self.bindings(bare)
            if len(bound) == 1:
                rhs = bound[0]
                env = _env_in(rhs, language)
                if env is not None:
                    return env
                lit = re.match(r"""\s*(?:r?["'])([^"']*)["']""", rhs)
                if lit and _URL.match(lit.group(1)):
                    return BaseSource("literal", lit.group(1))
                if _ATTR_CHAIN.match(rhs.strip().rstrip(";").removesuffix(".clone()")):
                    return _config(rhs.strip().rstrip(";").removesuffix(".clone()")) or BaseSource(
                        "unknown", bare
                    )
            return BaseSource("unknown", bare)
        chain = (
            expr.removeprefix("&").split("(", 1)[0].removesuffix(".as_str").removesuffix(".clone")
        )
        if _ATTR_CHAIN.match(chain):
            found = _config(chain)
            if found is not None:
                return found
        return BaseSource("unknown", expr)


def classify_base(
    base: str | None, host: str | None, source: str, language: str
) -> BaseSource | None:
    """One-off :meth:`BaseClassifier.classify`; classify many sites of a file with ONE
    ``BaseClassifier`` instead (this builds the file's binding index per call)."""
    return BaseClassifier(source, language).classify(base, host)


def _config(chain: str) -> BaseSource | None:
    if _CONFIG_ROOT.search(chain.rsplit(".", 1)[0]):
        return BaseSource("config", chain.rsplit(".", 1)[-1])
    return None


def host_of(url: str | None) -> str | None:
    """``host[:port]`` of an absolute URL, else None."""
    if not url:
        return None
    m = _URL.match(url)
    return m.group(1) if m and m.group(1) else None
