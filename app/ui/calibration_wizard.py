from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
    QMessageBox,
)
from app.settings_manager import SettingsManager
from app.processing.scale import mm_per_pixel_from_grid
from app.utils.image_io import imread, imwrite, resize_to_max_edge

MAX_CALIBRATION_DISPLAY_EDGE = 900
PREVIEW_TEMPLATE_EDGE = 2048

def _np_to_pixmap(image: np.ndarray) -> QPixmap:
    if image.ndim == 2:
        gray = np.ascontiguousarray(image)
        height, width = gray.shape
        qimage = QImage(gray.data, width, height, width, QImage.Format.Format_Grayscale8)
    else:
        bgr = image
        if image.shape[2] == 3:
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888)
        elif image.shape[2] == 4:
            rgba = np.ascontiguousarray(image[..., [2, 1, 0, 3]])
            qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888)
        else:
            raise ValueError("Unsupported image shape")
    return QPixmap.fromImage(qimage.copy())


class CalibrationImageWidget(QLabel):
    """Interactive label supporting measurement clicks."""

    changed = pyqtSignal()

    def __init__(
        self,
        max_display_edge: int = MAX_CALIBRATION_DISPLAY_EDGE,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self._display_image: Optional[np.ndarray] = None
        self._current_pair: List[Tuple[int, int]] = []
        self._measurements: List[float] = []
        self._scale_ratio_x = 1.0
        self._scale_ratio_y = 1.0
        self._max_display_edge = max_display_edge
        self.setMouseTracking(True)

    def set_image(self, image: np.ndarray) -> None:
        resized = image.copy()
        self._scale_ratio_x = 1.0
        self._scale_ratio_y = 1.0
        if self._max_display_edge and max(image.shape[:2]) > self._max_display_edge:
            height, width = image.shape[:2]
            scale = self._max_display_edge / float(max(height, width))
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
            self._scale_ratio_x = width / new_width
            self._scale_ratio_y = height / new_height

        self._display_image = resized.copy()
        pixmap = _np_to_pixmap(self._display_image)
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())
        self.reset()

    def reset(self) -> None:
        self._current_pair.clear()
        self._measurements.clear()
        if self._display_image is not None:
            self._update_overlay()
        self.changed.emit()

    def measurements(self) -> List[float]:
        return list(self._measurements)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._display_image is None:
            return
        self._handle_measure_click(event)

    def _handle_measure_click(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = int(event.position().x())
        y = int(event.position().y())
        self._current_pair.append((x, y))
        if len(self._current_pair) == 2:
            (x1, y1), (x2, y2) = self._current_pair
            sx1, sy1 = self._to_source_coords(x1, y1)
            sx2, sy2 = self._to_source_coords(x2, y2)
            distance = float(((sx1 - sx2) ** 2 + (sy1 - sy2) ** 2) ** 0.5)
            if distance > 1.0:
                self._measurements.append(distance)
            self._current_pair.clear()
        self._update_overlay()
        self.changed.emit()

    def _to_source_coords(self, x: int, y: int) -> Tuple[float, float]:
        return x * self._scale_ratio_x, y * self._scale_ratio_y

    def _update_overlay(self) -> None:
        if self._display_image is None:
            return
        base = self._display_image.copy()
        for x, y in self._current_pair:
            cv2.drawMarker(base, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
        pixmap = _np_to_pixmap(base)
        self.setPixmap(pixmap)


class TemplatePreviewPage(QWizardPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Эталон")
        layout = QVBoxLayout(self)
        self.preview = QLabel("Эталон не найден.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(400, 300)
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reload_button = QPushButton("Перечитать эталон")
        self.reload_button.clicked.connect(self._reload_template)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.info_label)
        layout.addWidget(self.reload_button)

    def initializePage(self) -> None:
        self._reload_template()

    def _reload_template(self) -> None:
        wizard: CalibrationWizard = self.wizard()  # type: ignore[assignment]
        template = None
        original_shape: Optional[Tuple[int, int]] = None
        if wizard.template_path.exists():
            template = imread(wizard.template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                original_shape = template.shape[:2]
        wizard.template_gray = template
        if template is not None:
            preview = resize_to_max_edge(template, PREVIEW_TEMPLATE_EDGE)
            pixmap = _np_to_pixmap(preview)
            self.preview.setPixmap(pixmap.scaledToWidth(400, Qt.TransformationMode.SmoothTransformation))
            dims = f"{template.shape[1]}×{template.shape[0]}"
            if original_shape and template.shape[:2] != original_shape:
                dims += f" (из {original_shape[1]}×{original_shape[0]})"
            self.info_label.setText(f"Файл: {wizard.template_path}\nРазмер: {dims}")
        else:
            self.preview.setText("Эталон не найден. Сохраните его в главном окне.")
            self.preview.setPixmap(QPixmap())
            self.info_label.setText(str(wizard.template_path))

    def validatePage(self) -> bool:
        wizard: CalibrationWizard = self.wizard()  # type: ignore[assignment]
        if wizard.template_gray is None:
            QMessageBox.warning(self, "Эталон", "Сначала сохраните эталон в главном окне.")
            return False
        return True


class ScaleCalibrationPage(QWizardPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Расчёт масштаба")
        layout = QVBoxLayout(self)
        self.image_widget = CalibrationImageWidget()
        self.info_label = QLabel("Кликните по паре точек с известным расстоянием. Повторите несколько раз.")
        self.mm_step_spin = QDoubleSpinBox()
        self.mm_step_spin.setRange(1.0, 200.0)
        self.mm_step_spin.setValue(10.0)
        self.mm_step_spin.setDecimals(2)
        self.result_label = QLabel("Мм/пикс: —")
        self.reset_button = QPushButton("Сбросить измерения")
        self.reset_button.clicked.connect(self._reset_measurements)
        self.image_widget.changed.connect(self._update_result)
        layout.addWidget(self.image_widget, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.info_label)
        layout.addWidget(QLabel("Известный шаг (мм):"))
        layout.addWidget(self.mm_step_spin)
        layout.addWidget(self.result_label)
        layout.addWidget(self.reset_button)

    def initializePage(self) -> None:
        wizard: CalibrationWizard = self.wizard()  # type: ignore[assignment]
        if wizard.template_gray is not None:
            color = cv2.cvtColor(wizard.template_gray, cv2.COLOR_GRAY2BGR)
            self.image_widget.set_image(color)
        self.mm_step_spin.setValue(float(wizard.grid_step_mm))
        self._update_result()

    def _reset_measurements(self) -> None:
        self.image_widget.reset()
        self._update_result()

    def _update_result(self) -> None:
        wizard: CalibrationWizard = self.wizard()  # type: ignore[assignment]
        measurements = self.image_widget.measurements()
        if measurements:
            try:
                mm_per_pixel = mm_per_pixel_from_grid(self.mm_step_spin.value(), measurements)
            except ValueError:
                mm_per_pixel = wizard.mm_per_pixel or 0.05
        else:
            mm_per_pixel = wizard.mm_per_pixel or 0.05
        self.result_label.setText(f"Мм/пикс: {mm_per_pixel:.4f}")
        wizard.mm_per_pixel = mm_per_pixel
        wizard.grid_step_mm = self.mm_step_spin.value()

    def validatePage(self) -> bool:
        self._update_result()
        wizard: CalibrationWizard = self.wizard()  # type: ignore[assignment]
        if wizard.mm_per_pixel <= 0:
            QMessageBox.warning(self, "Масштаб", "Некорректное значение мм/пикс.")
            return False
        return True


class CalibrationWizard(QWizard):
    """Full calibration workflow: template preview and scale tuning."""

    def __init__(self, settings_manager: SettingsManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.setWindowTitle("Мастер калибровки")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(960, 720)

        calibration = settings_manager.get("calibration", {})
        self.grid_step_mm: float = float(calibration.get("grid_step_mm", 10.0))
        self.mm_per_pixel: float = float(calibration.get("mm_per_pixel", 0.05))
        self.template_path = Path(calibration.get("template_path", "app/data/template.png"))
        self.template_gray: Optional[np.ndarray] = None

        if self.template_path.exists():
            existing = imread(self.template_path, cv2.IMREAD_GRAYSCALE)
            if existing is not None:
                self.template_gray = existing

        self.addPage(TemplatePreviewPage(self))
        self.addPage(ScaleCalibrationPage(self))

    def accept(self) -> None:
        if self.template_gray is None:
            QMessageBox.warning(self, "Калибровка", "Эталон не сохранён.")
            return
        self.template_path.parent.mkdir(parents=True, exist_ok=True)
        if not imwrite(self.template_path, self.template_gray):
            QMessageBox.critical(self, "Калибровка", "Не удалось сохранить эталон.")
            return
        self._update_settings()
        super().accept()

    def _update_settings(self) -> None:
        calibration = dict(self.settings_manager.get("calibration", {}))
        calibration["grid_step_mm"] = self.grid_step_mm
        calibration["mm_per_pixel"] = self.mm_per_pixel
        calibration["template_path"] = str(self.template_path)
        calibration["mask_path"] = ""
        self.settings_manager.set("calibration", calibration)


__all__ = ["CalibrationWizard", "PREVIEW_TEMPLATE_EDGE"]
