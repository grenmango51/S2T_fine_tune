"""Stable local artifact paths, independent of the calling module."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
