from pathlib import Path

from app.settings_manager import DEFAULT_SETTINGS, SettingsManager


def test_settings_manager_loads_defaults(tmp_path: Path):
    settings_path = tmp_path / "settings.json"
    manager = SettingsManager(settings_path)
    assert manager.get("version") == DEFAULT_SETTINGS["version"]
    manager.set("active_camera_id", 2)
    manager.save()
    reloaded = SettingsManager(settings_path)
    assert reloaded.get("active_camera_id") == 2

