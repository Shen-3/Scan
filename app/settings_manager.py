from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_SETTINGS: Dict[str, Any] = {
    "version": 1,
    "camera_profiles": {},
    "active_camera_id": 0,
    "processing": {
        "target_resolution": [1920, 1080],
        "min_hole_diameter_mm": 4.5,
        "max_hole_diameter_mm": 12.0,
        "bullet_diameter_mm": 6.0,
        "use_adaptive_threshold": False,
        "gaussian_sigma": 1.0,
        "clahe_clip_limit": 2.0,
        "adaptive_block_size": 11,
        "adaptive_c": 0.0,
    },
    "calibration": {
        "grid_step_mm": 10.0,
        "mask_path": "",
        "template_path": "app/data/template.png",
        "mm_per_pixel": 0.05,
    },
    "export": {
        "output_dir": "results",
        "author": "",
        "organization": "",
        "default_formats": ["csv", "pdf", "docx"],
    },
    "ui": {
        "language": "ru",
        "show_problem_candidates": False,
        "show_r50": True,
        "show_r90": False,
    },
}


class SettingsManager:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return json.loads(json.dumps(DEFAULT_SETTINGS))
        try:
            text = self.path.read_text(encoding="utf-8")
            return _merge_settings(json.loads(text), DEFAULT_SETTINGS)
        except (json.JSONDecodeError, OSError):
            return json.loads(json.dumps(DEFAULT_SETTINGS))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def get_camera_profile(self, camera_id: int) -> Dict[str, Any]:
        profiles = self.data.setdefault("camera_profiles", {})
        return profiles.setdefault(str(camera_id), {})

    def update_camera_profile(self, camera_id: int, values: Dict[str, Any]) -> None:
        profiles = self.data.setdefault("camera_profiles", {})
        profile = profiles.setdefault(str(camera_id), {})
        profile.update(values)
        self.save()


def _merge_settings(current: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for key, default_value in defaults.items():
        if key not in current:
            merged[key] = default_value
            continue
        if isinstance(default_value, dict) and isinstance(current[key], dict):
            merged[key] = _merge_settings(current[key], default_value)
        else:
            merged[key] = current[key]
    for key, value in current.items():
        if key not in merged:
            merged[key] = value
    return merged
