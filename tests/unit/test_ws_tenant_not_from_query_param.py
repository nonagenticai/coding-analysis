"""The quality WS tenant must come from the authenticated context, not the query string.

Regression test for the 2026-07-29 finding. The ingress Lua filter
(gitops-production .../envoy-filters/ext-authz-enforce.yaml:338-354) strips a
client-supplied ``x-tenant-id`` off every inbound request and re-injects only
what license-authz resolved — which is what makes the header trustworthy. But it
strips *headers*; it cannot strip a query string. While ``?tenant_id=`` was
honoured, a client denied ``X-Tenant-Id: victim`` could pass the same value as a
query param and read another tenant's quality stream.

This is the source-level guard: it fails if the pattern is reintroduced anywhere
under ``src/``. Three sibling sites in coding-realtime had the identical bug, so
a repo-wide guard is the thing that actually holds the class closed.
"""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def test_no_source_file_reads_tenant_id_from_query_params():
    offenders = [
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if 'query_params.get("tenant_id")' in p.read_text()
    ]
    assert offenders == [], (
        f"tenant_id read from query params in: {offenders}. The edge filter "
        f"strips headers but cannot strip a query string, so ?tenant_id= is a "
        f"cross-tenant spoof vector. Read the X-Tenant-Id header instead."
    )


def test_quality_ws_acts_on_should_close_and_keeps_no_raw_header_fallback():
    """Pin the specific calls, so a partial revert is caught as well as a rewrite.

    This test has been wrong twice, in the same way both times — it asserted a
    string that is present for the wrong reason.

    * It was `..._resolves_tenant_from_header_only` and asserted the raw header
      read was present. That became false-by-omission when the primary path
      became `resolve_websocket_identity`.
    * It then asserted `"Outcome.REFUSED" in text`. That string now appears only
      inside a COMMENT warning against re-deriving the decision, so the
      assertion would pass against a handler that had lost the guard entirely —
      `"" in <str>` energy: a substring check proves nothing about which code
      runs.

    So assert the two things that are actually load-bearing, on the FUNCTION
    body rather than the whole file:

    1. the decision is taken from `identity.should_close`, not re-derived from
       the enum — a handler testing `outcome is Outcome.REFUSED` itself serves
       every uncredentialed handshake, which is the hole coding-client 1.15.0
       closed; and
    2. there is no raw `X-Tenant-Id` read left to fall back to.
    """
    text = (SRC / "app/api/endpoints/quality.py").read_text()
    assert "resolve_websocket_identity(websocket)" in text
    assert 'query_params.get("tenant_id")' not in text

    body = _quality_websocket_body(text)
    assert 'websocket.headers.get("X-Tenant-Id")' not in body, (
        "the uncredentialed raw-header fallback is closed — a handshake with no "
        "verifiable licence must be refused, not served on its own header"
    )

    # The FIRST branch after resolving the identity is the close decision, and
    # it must read `should_close`. The enum may still be read further in — it
    # labels the log line — but it must not be what gates the close, because a
    # handler testing `outcome is Outcome.REFUSED` serves every uncredentialed
    # handshake, which is precisely the hole coding-client 1.15.0 closed.
    after_resolve = body[body.index("resolve_websocket_identity(websocket)") :]
    first_branch = next(
        line.strip() for line in after_resolve.splitlines() if line.strip().startswith("if ")
    )
    assert first_branch == "if identity.should_close:", first_branch


def _quality_websocket_body(text: str) -> str:
    """The source of `quality_websocket` alone.

    Scoping matters: the file-wide substring checks this test used to make were
    satisfiable by comments and by unrelated functions.
    """
    marker = "async def quality_websocket("
    start = text.index(marker)
    rest = text[start + len(marker) :]
    # Ends at the next top-level `async def` / `def` / decorator.
    for terminator in ("\n@router.", "\nasync def ", "\ndef "):
        idx = rest.find(terminator)
        if idx != -1:
            rest = rest[:idx]
    assert len(rest) > 500, "positive control: the body must actually be sliced out"
    return rest
