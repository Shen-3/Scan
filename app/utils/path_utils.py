from __future__ import annotations
from pathlib import Path
from typing import Union


def normalize_settings_path(path_value: Union[str, Path, None], default: Union[str, Path]) -> Path:
    """
    Convert a settings path string (may contain backslashes) into a usable Path.
    Falls back to the provided default when empty.
    """
    raw = default if path_value in (None, "", ".") else path_value
    normalized = str(raw).replace("\\", "/")
    return Path(normalized).expanduser()

def settings_path_str(path: Path) -> str:
    """Store paths in settings using POSIX separators for cross-platform consistency."""
    return path.as_posix()