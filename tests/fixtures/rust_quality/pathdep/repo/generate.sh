#!/bin/sh
# Regenerate Cargo.lock with cargo itself (rust-version 1.75 -> lockfile v3).
set -eu
cd "$(dirname "$0")"
rm -f Cargo.lock
cargo generate-lockfile --offline
