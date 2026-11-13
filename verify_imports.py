#!/usr/bin/env python3
"""Quick import verification script."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from main import app

    print(f"IMPORT: PASS (app.title={app.title})")
except Exception as e:
    print(f"IMPORT: FAIL ({e})")
    import traceback

    traceback.print_exc()
    sys.exit(1)
