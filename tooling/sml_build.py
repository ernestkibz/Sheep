#!/usr/bin/env python3
"""The Sheep Quantizer's former name, kept so older scripts and commands still
run. The quantizer is tooling/sml_quantize.py."""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sml_quantize import *  # noqa: E402,F401,F403
from sml_quantize import chat_block, promote_order  # noqa: E402,F401  (named for importers)

if __name__ == "__main__":
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sml_quantize.py"), run_name="__main__")
