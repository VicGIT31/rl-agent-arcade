"""Ensure the project root is importable during tests (``import envs`` etc.)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
