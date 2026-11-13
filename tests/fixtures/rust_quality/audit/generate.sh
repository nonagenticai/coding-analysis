#!/bin/sh
# Regenerate Cargo.lock and the recorded `cargo audit --json` report with the tools.
# smallvec 1.6.0 carries a published RustSec advisory (RUSTSEC-2021-0003).
set -eu
cd "$(dirname "$0")"
rm -f Cargo.lock
cargo generate-lockfile
cargo update -p smallvec --precise 1.6.0
cargo audit --json > cargo-audit.json || true  # exits 1 when it finds vulnerabilities
