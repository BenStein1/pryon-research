"""Explicit local assets; no user, device or workstation defaults."""
import os
from pathlib import Path
DEFAULT_SOURCE=Path(os.environ.get("PRYON_MODEL", "private-assets/source.onnx"))
