from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.settings_manager import SettingsManager


class SettingsDialog(QDialog):
    """Dialog for editing processing and export preferences."""

    def __init__(self, settings_manager: SettingsManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.setWindowTitle("Настройки")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        self.camera_group = self._create_camera_group()
        self.processing_group = self._create_processing_group()
        self.export_group = self._create_export_group()
        self.ui_group = self._create_ui_group()

        layout.addWidget(self.camera_group)
        layout.addWidget(self.processing_group)
        layout.addWidget(self.export_group)
        layout.addWidget(self.ui_group)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._load_settings()

    def _create_camera_group(self) -> QGroupBox:
        group = QGroupBox("Камера")
        layout = QFormLayout(group)
        self.camera_id_spin = QSpinBox()
        self.camera_id_spin.setRange(0, 20)
        self.resolution_w_spin = QSpinBox()
        self.resolution_w_spin.setRange(320, 7680)
        self.resolution_w_spin.setSingleStep(160)
        self.resolution_h_spin = QSpinBox()
        self.resolution_h_spin.setRange(240, 4320)
        self.resolution_h_spin.setSingleStep(90)
        layout.addRow("ID камеры", self.camera_id_spin)
        res_widget = QWidget()
        res_layout = QHBoxLayout(res_widget)
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.addWidget(self.resolution_w_spin)
        res_layout.addWidget(self.resolution_h_spin)
        layout.addRow("Разрешение (W/H)", res_widget)
        return group

    def _create_processing_group(self) -> QGroupBox:
        group = QGroupBox("Обработка")
        layout = QFormLayout(group)
        self.min_diameter_spin = QDoubleSpinBox()
        self.min_diameter_spin.setRange(1.0, 20.0)
        self.min_diameter_spin.setDecimals(1)
        self.max_diameter_spin = QDoubleSpinBox()
        self.max_diameter_spin.setRange(1.0, 30.0)
        self.max_diameter_spin.setDecimals(1)
        self.bullet_diameter_spin = QDoubleSpinBox()
        self.bullet_diameter_spin.setRange(1.0, 30.0)
        self.bullet_diameter_spin.setDecimals(1)
        self.gaussian_sigma_spin = QDoubleSpinBox()
        self.gaussian_sigma_spin.setRange(0.0, 5.0)
        self.gaussian_sigma_spin.setDecimals(1)
        self.clahe_clip_spin = QDoubleSpinBox()
        self.clahe_clip_spin.setRange(0.5, 10.0)
        self.clahe_clip_spin.setDecimals(1)
        self.use_adaptive_checkbox = QCheckBox("Адаптивный порог")
        self.adaptive_block_spin = QSpinBox()
        self.adaptive_block_spin.setRange(3, 99)
        self.adaptive_block_spin.setSingleStep(2)
        self.adaptive_c_spin = QDoubleSpinBox()
        self.adaptive_c_spin.setRange(-20.0, 20.0)
        self.adaptive_c_spin.setDecimals(1)
        self.mm_per_pixel_spin = QDoubleSpinBox()
        self.mm_per_pixel_spin.setRange(0.001, 2.0)
        self.mm_per_pixel_spin.setDecimals(4)
        layout.addRow("Мин. диаметр (мм)", self.min_diameter_spin)
        layout.addRow("Макс. диаметр (мм)", self.max_diameter_spin)
        layout.addRow("Диаметр пули (мм)", self.bullet_diameter_spin)
        layout.addRow("Sigma blur", self.gaussian_sigma_spin)
        layout.addRow("CLAHE clip", self.clahe_clip_spin)
        layout.addRow(self.use_adaptive_checkbox)
        layout.addRow("Размер окна адаптивного порога", self.adaptive_block_spin)
        layout.addRow("Смещение порога (C)", self.adaptive_c_spin)
        layout.addRow("Мм/пикс", self.mm_per_pixel_spin)
        return group

    def _create_export_group(self) -> QGroupBox:
        group = QGroupBox("Экспорт")
        layout = QFormLayout(group)
        self.output_dir_edit = QLineEdit()
        browse_button = QPushButton("…")
        browse_button.clicked.connect(self._browse_output_dir)
        output_widget = QWidget()
        output_layout = QHBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(browse_button)
        self.organization_edit = QLineEdit()
        self.author_edit = QLineEdit()
        layout.addRow("Папка отчётов", output_widget)
        layout.addRow("Организация", self.organization_edit)
        layout.addRow("Оператор", self.author_edit)
        return group

    def _create_ui_group(self) -> QGroupBox:
        group = QGroupBox("Интерфейс")
        layout = QVBoxLayout(group)
        self.show_r50_checkbox = QCheckBox("Показывать R50")
        self.show_r90_checkbox = QCheckBox("Показывать R90")
        self.show_problem_checkbox = QCheckBox("Показывать кандидатов")
        layout.addWidget(self.show_r50_checkbox)
        layout.addWidget(self.show_r90_checkbox)
        layout.addWidget(self.show_problem_checkbox)
        return group

    def _browse_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Выберите папку экспорта", self.output_dir_edit.text() or ".")
        if directory:
            self.output_dir_edit.setText(directory)

    def _load_settings(self) -> None:
        camera_id = self.settings_manager.get("active_camera_id", 0)
        self.camera_id_spin.setValue(int(camera_id or 0))
        processing = self.settings_manager.get("processing", {})
        resolution = processing.get("target_resolution", [1920, 1080])
        self.resolution_w_spin.setValue(int(resolution[0]))
        self.resolution_h_spin.setValue(int(resolution[1]))
        self.min_diameter_spin.setValue(float(processing.get("min_hole_diameter_mm", 4.5)))
        self.max_diameter_spin.setValue(float(processing.get("max_hole_diameter_mm", 12.0)))
        self.bullet_diameter_spin.setValue(float(processing.get("bullet_diameter_mm", 6.0)))
        self.gaussian_sigma_spin.setValue(float(processing.get("gaussian_sigma", 1.0)))
        self.clahe_clip_spin.setValue(float(processing.get("clahe_clip_limit", 2.0)))
        self.use_adaptive_checkbox.setChecked(bool(processing.get("use_adaptive_threshold", True)))
        block_size = int(processing.get("adaptive_block_size", 11))
        if block_size % 2 == 0:
            block_size += 1
        self.adaptive_block_spin.setValue(max(3, block_size))
        self.adaptive_c_spin.setValue(float(processing.get("adaptive_c", 0.0)))
        calibration = self.settings_manager.get("calibration", {})
        self.mm_per_pixel_spin.setValue(float(calibration.get("mm_per_pixel", 0.05)))
        export = self.settings_manager.get("export", {})
        self.output_dir_edit.setText(str(export.get("output_dir", "results")))
        self.organization_edit.setText(export.get("organization", ""))
        self.author_edit.setText(export.get("author", ""))
        ui_settings = self.settings_manager.get("ui", {})
        self.show_r50_checkbox.setChecked(bool(ui_settings.get("show_r50", True)))
        self.show_r90_checkbox.setChecked(bool(ui_settings.get("show_r90", False)))
        self.show_problem_checkbox.setChecked(bool(ui_settings.get("show_problem_candidates", False)))

    def _on_accept(self) -> None:
        processing = dict(self.settings_manager.get("processing", {}))
        processing["target_resolution"] = [self.resolution_w_spin.value(), self.resolution_h_spin.value()]
        processing["min_hole_diameter_mm"] = self.min_diameter_spin.value()
        processing["max_hole_diameter_mm"] = self.max_diameter_spin.value()
        processing["bullet_diameter_mm"] = self.bullet_diameter_spin.value()
        processing["gaussian_sigma"] = self.gaussian_sigma_spin.value()
        processing["clahe_clip_limit"] = self.clahe_clip_spin.value()
        processing["use_adaptive_threshold"] = self.use_adaptive_checkbox.isChecked()
        processing["adaptive_block_size"] = self.adaptive_block_spin.value()
        processing["adaptive_c"] = self.adaptive_c_spin.value()
        self.settings_manager.set("processing", processing)
        self.settings_manager.set("active_camera_id", self.camera_id_spin.value())

        calibration = dict(self.settings_manager.get("calibration", {}))
        calibration["mm_per_pixel"] = self.mm_per_pixel_spin.value()
        self.settings_manager.set("calibration", calibration)

        export = dict(self.settings_manager.get("export", {}))
        export["output_dir"] = self.output_dir_edit.text() or "results"
        export["organization"] = self.organization_edit.text()
        export["author"] = self.author_edit.text()
        self.settings_manager.set("export", export)

        ui_settings = dict(self.settings_manager.get("ui", {}))
        ui_settings["show_r50"] = self.show_r50_checkbox.isChecked()
        ui_settings["show_r90"] = self.show_r90_checkbox.isChecked()
        ui_settings["show_problem_candidates"] = self.show_problem_checkbox.isChecked()
        self.settings_manager.set("ui", ui_settings)

        self.accept()


__all__ = ["SettingsDialog"]
