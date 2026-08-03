#!/usr/bin/env python3
"""Development entry point for vllm-cli-tui."""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from vllm_tui.app import main

if __name__ == "__main__":
    main()
