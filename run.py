#!/usr/bin/env python3
"""Development entry point for vllm-cli-tui."""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Re-exec with the project's venv when present: a bare ./run.py from a shell
# without the venv activated would otherwise run on the system python and
# miss the dependencies (same convention as the ./vllm-cli-tui launcher).
# abspath, not realpath: the venv python symlinks to the system interpreter,
# so realpath would call them equal and skip the exec.
venv_python = HERE / ".venv" / "bin" / "python"
if venv_python.exists() and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
    os.execv(str(venv_python), [str(venv_python), str(HERE / "run.py"), *sys.argv[1:]])

# Add src to path for development
sys.path.insert(0, str(HERE / "src"))

from vllm_tui.app import main

if __name__ == "__main__":
    main()