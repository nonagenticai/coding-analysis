#!/bin/sh
# Regenerate this fixture's Cargo.lock with cargo itself (never edit it by hand).
# The crate's rust-version (1.75) makes cargo write lockfile format v3, which the
# CI toolchain (Ubuntu noble apt cargo 1.75, via the archetype's extra_apt) reads.
set -eu
cd "$(dirname "$0")"
rm -f Cargo.lock
cargo generate-lockfile --offline
