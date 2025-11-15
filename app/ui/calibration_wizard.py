from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
    QMessageBox,
)
from app.settings_manager import SettingsManager
from app.processing.scale import mm_per_pixel_from_grid
from app.utils.image_io import imread, imwrite


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
    """Interactive label supporting measurement clicks or rectangular selection."""

    changed = pyqtSignal()

    def __init__(self, mode: str = "measure", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self._mode = mode
        self._image: Optional[np.ndarray] = None
        self._current_pair: List[Tuple[int, int]] = []
        self._measurements: List[float] = []
        self._overlay = None
        self._mask_rect: Optional[QRect] = None
        self._rubber_origin = QPoint()
        self._selecting = False
        self.setMouseTracking(True)

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.reset()

    def set_image(self, image: np.ndarray) -> None:
        self._image = image.copy()
        pixmap = _np_to_pixmap(image)
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())
        self.reset()

    def reset(self) -> None:
        self._current_pair.clear()
        self._measurements.clear()
        self._mask_rect = None
        if self._image is not None:
            self._update_overlay()
        self.changed.emit()

    def measurements(self) -> List[float]:
        return list(self._measurements)

    def mask_rect(self) -> Optional[QRect]:
        return self._mask_rect

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._image is None:
            return
        if self._mode == "measure":
            self._handle_measure_click(event)
        elif self._mode == "mask":
            if event.button() == Qt.MouseButton.LeftButton:
                self._selecting = True
                self._rubber_origin = event.position().toPoint()
            elif event.button() == Qt.MouseButton.RightButton:
                self._mask_rect = None
                self._update_overlay()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._mode == "mask" and self._selecting and self._image is not None:
            current = event.position().toPoint()
            rect = QRect(self._rubber_origin, current).normalized()
            self._mask_rect = rect
            self._update_overlay()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._mode == "mask" and event.button() == Qt.MouseButton.LeftButton:
            self._selecting = False

    def _handle_measure_click(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = int(event.position().x())
        y = int(event.position().y())
        self._current_pair.append((x, y))
        if len(self._current_pair) == 2:
            (x1, y1), (x2, y2) = self._current_pair
            distance = float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
            if distance > 1.0:
                self._measurements.append(distance)
            self._current_pair.clear()
        self._update_overlay()
        self.changed.emit()

    def _update_overlay(self) -> None:
        if self._image is None:
            return
        base = self._image.copy()
        if self._mode == "measure":
            for x, y in self._current_pair:
                cv2.drawMarker(base, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
        elif self._mode == "mask":
            if self._mask_rect is not None:
                rect = self._mask_rect
                cv2.rectangle(
                    base,
                    (rect.left(), rect.top()),
                    (rect.right(), rect.bottom()),
                    (0, 255, 255),
                    2,
                )
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
        if wizard.template_path.exists():
            template = imread(wizard.template_path, cv2.IMREAD_GRAYSCALE)
        wizard.template_gray = template
        if template is not None:
            pixmap = _np_to_pixmap(template)
            self.preview.setPixmap(pixmap.scaledToWidth(400, Qt.TransformationMode.SmoothTransformation))
            self.info_label.setText(f"Файл: {wizard.template_path}")
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
        self.image_widget = CalibrationImageWidget(mode="measure")
        self.info_label = QLabel("Кликните по паре точек с известным расстоянием. Повторите несколько раз.")
        self.mm_step_spin = QDoubleSpinBox()
        self.mm_step_spin.setRange(1.0, 200.0)
        self.mm_step_spin.setValue(10.0)
        self.mm_step_spin.setDecimals(2)
        self.result_label = QLabel("Мм/пикс: —")
        self.reset_button = QPushButton("Сбросить измерения")
        self.reset_button.clicked.connect(self._reset_measurements)
        self.image_widget.changed.connect(self._update_result)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.scroll_area.setMinimumSize(600, 400)
        self.scroll_area.setWidget(self.image_widget)
        layout.addWidget(self.scroll_area)
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


class MaskSelectionPage(QWizardPage):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setTitle("Рабочая зона")
        layout = QVBoxLayout(self)
        self.image_widget = CalibrationImageWidget(mode="mask")
        self.info_label = QLabel("Выделите прямоугольник рабочей области (правая кнопка сбрасывает выбор).")
        self.reset_button = QPushButton("Очистить выделение")
        self.reset_button.clicked.connect(self.image_widget.reset)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.scroll_area.setMinimumSize(600, 400)
        self.scroll_area.setWidget(self.image_widget)
        layout.addWidget(self.scroll_area)
        layout.addWidget(self.info_label)
        layout.addWidget(self.reset_button)

    def initializePage(self) -> None:
        wizard: CalibrationWizard = self.wizard()  # type: ignore[assignment]
        if wizard.template_gray is not None:
            color = cv2.cvtColor(wizard.template_gray, cv2.COLOR_GRAY2BGR)
            self.image_widget.set_image(color)

    def validatePage(self) -> bool:
        rect = self.image_widget.mask_rect()
        if rect is None or rect.width() < 10 or rect.height() < 10:
            QMessageBox.warning(self, "Маска", "Выделите рабочую область.")
            return False
        wizard: CalibrationWizard = self.wizard()  # type: ignore[assignment]
        wizard.mask_rect = rect
        return True


class CalibrationWizard(QWizard):
    """Full calibration workflow: template preview, scale tuning, mask definition."""

    def __init__(self, settings_manager: SettingsManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.setWindowTitle("Мастер калибровки")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)

        calibration = settings_manager.get("calibration", {})
        self.grid_step_mm: float = float(calibration.get("grid_step_mm", 10.0))
        self.mm_per_pixel: float = float(calibration.get("mm_per_pixel", 0.05))
        self.template_path = Path(calibration.get("template_path", "app/data/template.png"))
        self.mask_path = Path(calibration.get("mask_path", "app/data/mask.png"))
        self.template_gray: Optional[np.ndarray] = None
        self.mask_rect: Optional[QRect] = None

        if self.template_path.exists():
            existing = imread(self.template_path, cv2.IMREAD_GRAYSCALE)
            if existing is not None:
                self.template_gray = existing

        self.addPage(TemplatePreviewPage(self))
        self.addPage(ScaleCalibrationPage(self))
        self.addPage(MaskSelectionPage(self))

    def accept(self) -> None:
        if self.template_gray is None:
            QMessageBox.warning(self, "Калибровка", "Эталон не сохранён.")
            return
        self.template_path.parent.mkdir(parents=True, exist_ok=True)
        if not imwrite(self.template_path, self.template_gray):
            QMessageBox.critical(self, "Калибровка", "Не удалось сохранить эталон.")
            return
        if self.mask_rect is not None and self.template_gray is not None:
            mask = np.zeros_like(self.template_gray, dtype=np.uint8)
            rect = self.mask_rect
            x0 = max(0, rect.left())
            y0 = max(0, rect.top())
            x1 = min(mask.shape[1], rect.right() + 1)
            y1 = min(mask.shape[0], rect.bottom() + 1)
            mask[y0:y1, x0:x1] = 255
            self.mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not imwrite(self.mask_path, mask):
                QMessageBox.critical(self, "Калибровка", "Не удалось сохранить маску.")
                return

        self._update_settings()
        super().accept()

    def _update_settings(self) -> None:
        calibration = dict(self.settings_manager.get("calibration", {}))
        calibration["grid_step_mm"] = self.grid_step_mm
        calibration["mm_per_pixel"] = self.mm_per_pixel
        calibration["template_path"] = str(self.template_path)
        calibration["mask_path"] = str(self.mask_path)
        self.settings_manager.set("calibration", calibration)


__all__ = ["CalibrationWizard"]
