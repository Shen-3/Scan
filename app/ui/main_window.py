from __future__ import annotations

import copy
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QStatusBar,
)

from app.camera import CameraManager, CameraError
from app.export.to_csv import export_csv
from app.export.to_docx import export_docx
from app.export.to_pdf import export_pdf
from app.models import ProcessingResult, ShotPoint
from app.processing.detect_hits import DetectionParams
from app.processing.diff_threshold import DiffThresholdParams
from app.processing.overlay import render_overlay
from app.processing.pipeline import PipelineConfig, ProcessingPipeline
from app.processing.scale import ScaleModel
from app.settings_manager import SettingsManager
from app.ui.shot_view import ShotGraphicsView
from app.ui.settings_dialog import SettingsDialog
from app.ui.calibration_wizard import CalibrationWizard
from app.utils.image_io import imread, imwrite

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window wiring UI with processing pipeline."""

    def __init__(self, settings_path: Path) -> None:
        super().__init__()
        self.settings_manager = SettingsManager(settings_path)
        self.setWindowTitle("СТП анализатор")
        self.resize(1400, 800)

        self.shot_view: ShotGraphicsView = ShotGraphicsView()
        self.metrics_table: QTableWidget = self._create_metrics_table()
        self.points_list: QListWidget = QListWidget()
        self.status_label: QLabel = QLabel("Готово")
        self.camera_button = QPushButton("Снять кадр")
        self.load_image_button = QPushButton("Загрузить файл")
        self.process_button = QPushButton("Обработать")
        self.reset_button = QPushButton("Сброс")
        self.template_button = QPushButton("Загрузить эталон")
        self.export_csv_button = QPushButton("Экспорт CSV")
        self.export_pdf_button = QPushButton("Экспорт PDF")
        self.export_docx_button = QPushButton("Экспорт DOCX")
        self.capture_template_button = QPushButton("Сохранить текущий кадр как эталон")
        self.calibration_button = QPushButton("Мастер калибровки")
        self.settings_button = QPushButton("Настройки")
        self.show_problems_checkbox = QCheckBox("Показать проблемные области")
        self.split_button = QPushButton("Разделить пятно")
        self.undo_button = QToolButton()
        self.undo_button.setText("⬅ Undo")
        self.redo_button = QToolButton()
        self.redo_button.setText("Redo ➡")

        ui_settings = self.settings_manager.get("ui", {})
        self.show_problems_checkbox.setChecked(bool(ui_settings.get("show_problem_candidates", False)))

        # initialize early so status updates can be emitted before UI layout is created
        self._status_bar: Optional[QStatusBar] = None
        self._current_frame: Optional[np.ndarray] = None
        self._current_result: Optional[ProcessingResult] = None
        self._next_point_id: int = 1
        self._undo_stack: List[List[ShotPoint]] = []
        self._redo_stack: List[List[ShotPoint]] = []
        self._scale_model = self._load_scale_model()
        self._pipeline = self._create_pipeline()

        self._setup_layout()
        self._connect_signals()

    def _setup_layout(self) -> None:
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        metrics_group = QGroupBox("Метрики")
        metrics_layout = QVBoxLayout(metrics_group)
        metrics_layout.addWidget(self.metrics_table)

        points_group = QGroupBox("Попадания")
        points_layout = QVBoxLayout(points_group)
        points_layout.addWidget(self.points_list)

        buttons_group = QGroupBox("Действия")
        buttons_layout = QFormLayout(buttons_group)
        buttons_layout.addRow(self.settings_button, self.calibration_button)
        buttons_layout.addRow(self.camera_button, self.process_button)
        buttons_layout.addRow(self.load_image_button, self.reset_button)
        buttons_layout.addRow(self.template_button, self.capture_template_button)
        buttons_layout.addRow(self.export_csv_button, self.export_pdf_button)
        buttons_layout.addRow(self.export_docx_button, QLabel())

        qa_group = QGroupBox("Контроль качества")
        qa_layout = QVBoxLayout(qa_group)
        qa_layout.addWidget(self.show_problems_checkbox)
        qa_layout.addWidget(self.split_button)
        qa_layout.addWidget(self.undo_button)
        qa_layout.addWidget(self.redo_button)
        qa_layout.addStretch()
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)

        right_layout.addWidget(metrics_group)
        right_layout.addWidget(points_group)
        right_layout.addWidget(buttons_group)
        right_layout.addWidget(qa_group)
        right_layout.addStretch()
        right_layout.addWidget(self.status_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.shot_view)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([int(self.width() * 0.65), int(self.width() * 0.35)])

        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.addWidget(splitter)
        self.setCentralWidget(container)
        status_bar = self.statusBar()
        if status_bar is None:
            status_bar = QStatusBar(self)
            self.setStatusBar(status_bar)
        self._status_bar = status_bar
        self._status_bar.showMessage("Готово")

    @staticmethod
    def _create_metrics_table() -> QTableWidget:
        metrics = [
            ("Количество", "point_count"),
            ("СТП X (мм)", "mean_x_mm"),
            ("СТП Y (мм)", "mean_y_mm"),
            ("Смещ. (мм)", "displacement_mm"),
            ("Азимут (°)", "azimuth_deg"),
            ("Mean Radius (мм)", "mean_radius_mm"),
            ("Extreme Spread (мм)", "extreme_spread_mm"),
            ("R50 (мм)", "r50_mm"),
            ("σx (мм)", "std_x_mm"),
            ("σy (мм)", "std_y_mm"),
        ]
        table = QTableWidget(len(metrics), 2)
        table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        vertical_header = table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        for row, (label, _) in enumerate(metrics):
            table.setItem(row, 0, QTableWidgetItem(label))
            table.setItem(row, 1, QTableWidgetItem("---"))
        table.resizeColumnsToContents()
        table.setProperty("metric_keys", [key for _, key in metrics])
        return table

    def _connect_signals(self) -> None:
        self.camera_button.clicked.connect(self.capture_frame)
        self.process_button.clicked.connect(self.process_current_frame)
        self.reset_button.clicked.connect(self.reset_view)
        self.load_image_button.clicked.connect(self.load_image_from_file)
        self.template_button.clicked.connect(self.load_template_from_file)
        self.capture_template_button.clicked.connect(self.save_current_frame_as_template)
        self.export_csv_button.clicked.connect(lambda: self._export_result("csv"))
        self.export_pdf_button.clicked.connect(lambda: self._export_result("pdf"))
        self.export_docx_button.clicked.connect(lambda: self._export_result("docx"))
        self.shot_view.pointAdded.connect(self._handle_point_added)
        self.shot_view.pointRemoved.connect(self._handle_point_removed)
        self.shot_view.pointMoved.connect(self._handle_point_moved)
        self.show_problems_checkbox.stateChanged.connect(self._handle_show_problems_changed)
        self.split_button.clicked.connect(self._run_local_split)
        self.settings_button.clicked.connect(self._open_settings_dialog)
        self.calibration_button.clicked.connect(self._open_calibration_wizard)
        self.undo_button.clicked.connect(lambda: self._apply_history(pop_from="undo"))
        self.redo_button.clicked.connect(lambda: self._apply_history(pop_from="redo"))


    def _load_scale_model(self) -> ScaleModel:
        calibration = self.settings_manager.get("calibration", {})
        mm_per_pixel = float(calibration.get("mm_per_pixel", 0.05) or 0.05)
        reference_name = calibration.get("reference", "default")
        return ScaleModel(mm_per_pixel=mm_per_pixel, reference_name=reference_name)

    def _create_pipeline(self) -> Optional[ProcessingPipeline]:
        calibration = self.settings_manager.get("calibration", {})
        template_path = Path(calibration.get("template_path", "app/data/template.png"))
        mask_value = calibration.get("mask_path")
        mask_path = None
        if mask_value:
            candidate = Path(mask_value).expanduser()
            if str(candidate) not in {".", ""}:
                mask_path = candidate
        if not template_path.exists():
            self._set_status("Нет эталона: загрузите шаблон.")
            return None
        processing_settings = self.settings_manager.get("processing", {})
        detection_params = DetectionParams(
            min_diameter_mm=processing_settings.get("min_hole_diameter_mm", 4.5),
            max_diameter_mm=processing_settings.get("max_hole_diameter_mm", 12.0),
            min_circularity=processing_settings.get("min_circularity", DetectionParams.min_circularity),
            min_intensity_drop=processing_settings.get("min_intensity_drop", DetectionParams.min_intensity_drop),
            split_min_distance_mm=processing_settings.get("split_min_distance_mm", 5.0),
            min_diameter_relaxation=processing_settings.get("min_diameter_relaxation", 0.5),
        )
        use_adaptive = processing_settings.get("use_adaptive_threshold", True)
        morph_kernel_size = int(processing_settings.get("morph_kernel_size", 3))
        morph_iterations = int(processing_settings.get("morph_iterations", 1))
        if use_adaptive and self._scale_model.mm_per_pixel > 0:
            expected_radius_px = detection_params.min_diameter_mm / (2.0 * self._scale_model.mm_per_pixel)
            adaptive_kernel = int(round(expected_radius_px))
            adaptive_kernel = max(3, min(15, adaptive_kernel))
            adaptive_kernel |= 1  # ensure odd size for morphological ops
            morph_kernel_size = max(morph_kernel_size, adaptive_kernel)
            if morph_iterations < 1:
                morph_iterations = 1
        diff_params = DiffThresholdParams(
            use_adaptive=use_adaptive,
            gaussian_sigma=processing_settings.get("gaussian_sigma", 1.0),
            clahe_clip_limit=processing_settings.get("clahe_clip_limit", 2.0),
            adaptive_block_size=processing_settings.get("adaptive_block_size", 11),
            adaptive_c=processing_settings.get("adaptive_c", 0.0),
            morph_kernel_size=morph_kernel_size,
            morph_iterations=morph_iterations,
        )
        export_settings = self.settings_manager.get("export", {})
        output_dir = Path(export_settings.get("output_dir", "results"))
        show_r50 = self.settings_manager.get("ui", {}).get("show_r50", True)
        show_r90 = self.settings_manager.get("ui", {}).get("show_r90", False)
        config = PipelineConfig(
            diff_params=diff_params,
            detection_params=detection_params,
            mask_path=mask_path,
            template_path=template_path,
            output_dir=output_dir,
            show_r50=show_r50,
            show_r90=show_r90,
            collect_debug=True,
        )
        return ProcessingPipeline(self._scale_model, config)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        if self._status_bar is not None:
            self._status_bar.showMessage(message, 5000)

    def capture_frame(self) -> None:
        camera_id = self.settings_manager.get("active_camera_id", 0)
        processing = self.settings_manager.get("processing", {})
        target_resolution = processing.get("target_resolution", [1920, 1080])
        try:
            with CameraManager(device_index=camera_id, target_resolution=tuple(target_resolution)) as camera:
                frame = camera.capture_frame()
        except CameraError as exc:
            QMessageBox.critical(self, "Ошибка камеры", str(exc))
            logger.exception("Camera capture failed")
            return
        self._current_frame = frame.data
        self._set_status("Кадр получен. Нажмите \"Обработать\".")
        self.shot_view.set_background(self._current_frame)
        self.shot_view.set_editing_enabled(False)

    def load_image_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение мишени",
            os.getcwd(),
            "Изображения (*.png *.jpg *.jpeg *.bmp)",
        )
        if not path:
            return
        image = imread(path)
        if image is None:
            QMessageBox.warning(self, "Ошибка", "Не удалось загрузить изображение.")
            return
        self._current_frame = image
        self.shot_view.set_background(image)
        self.shot_view.set_editing_enabled(False)
        self._set_status("Изображение загружено. Нажмите \"Обработать\".")

    def load_template_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите эталон",
            os.getcwd(),
            "Изображения (*.png *.jpg *.jpeg *.bmp)",
        )
        if not path:
            return
        target_path = Path(self.settings_manager.get("calibration", {}).get("template_path", "app/data/template.png"))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        image = imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            QMessageBox.warning(self, "Ошибка", "Не удалось прочитать эталон.")
            return
        if not imwrite(target_path, image):
            QMessageBox.warning(self, "Ошибка", "Не удалось сохранить эталон.")
            return
        self._pipeline = self._create_pipeline()
        if self._pipeline:
            self._set_status("Эталон обновлён.")
        else:
            QMessageBox.warning(self, "Предупреждение", "Не удалось создать конвейер обработки.")

    def process_current_frame(self) -> None:
        if self._current_frame is None:
            QMessageBox.information(self, "Нет данных", "Сначала захватите изображение.")
            return
        if self._pipeline is None:
            QMessageBox.warning(self, "Нет конвейера", "Загрузите эталон перед обработкой.")
            return
        target_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            result = self._pipeline.process(self._current_frame, target_id=target_id)
        except Exception as exc:  # noqa: BLE001 broad except to inform user
            logger.exception("Processing failed")
            QMessageBox.critical(self, "Ошибка обработки", str(exc))
            return
        self._handle_processing_result(result)

    def _handle_processing_result(self, result: ProcessingResult) -> None:
        self._current_result = result
        self._next_point_id = max((point.id for point in result.points), default=0) + 1
        self._clear_history()
        if result.overlay_image is not None:
            background = result.overlay_image
        elif result.aligned_gray is not None:
            background = cv2.cvtColor(result.aligned_gray, cv2.COLOR_GRAY2BGR)
        elif self._current_frame is not None:
            background = self._current_frame
        else:
            background = np.zeros((1, 1, 3), dtype=np.uint8)
        self.shot_view.set_background(background)
        self.shot_view.set_scale(result.mm_per_pixel or self._scale_model.mm_per_pixel, result.origin_px or (0.0, 0.0))
        self.shot_view.load_points(result.points)
        self.shot_view.set_editing_enabled(True)
        self._refresh_metrics()
        self._refresh_points_list()
        self._push_history()
        stats = result.stats
        timing = f"Обработка: {stats.align_ms:.0f} + {stats.diff_ms:.0f} + {stats.detect_ms:.0f} + {stats.metrics_ms:.0f} мс"
        self._set_status(timing)
        if self.show_problems_checkbox.isChecked():
            self._update_overlay()

    def _refresh_metrics(self) -> None:
        keys = self.metrics_table.property("metric_keys")
        if not isinstance(keys, list):
            return
        values = self._current_result.metrics if self._current_result else None
        summary = self._current_result.to_summary_dict() if self._current_result else {}
        for row, key in enumerate(keys):
            item = self.metrics_table.item(row, 1)
            if item is None:
                continue
            if values and key in summary:
                item.setText(f"{summary[key]:.2f}")
            else:
                item.setText("---")
        self.metrics_table.resizeColumnsToContents()

    def _refresh_points_list(self) -> None:
        self.points_list.clear()
        if not self._current_result:
            return
        for point in sorted(self._current_result.points, key=lambda p: p.id):
            text = f"#{point.id}: ({point.x_mm:.1f}, {point.y_mm:.1f}) мм [{point.source}]"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, point.id)
            self.points_list.addItem(item)

    def _handle_point_added(self, x_px: float, y_px: float) -> None:
        if not self._current_result:
            return
        scene_point = QPointF(x_px, y_px)
        x_mm, y_mm = self.shot_view.scene_to_mm(scene_point)
        radius_mm = float(
            (self.settings_manager.get("processing", {}).get("min_hole_diameter_mm", 4.5)
             + self.settings_manager.get("processing", {}).get("max_hole_diameter_mm", 12.0))
            / 4.0
        )
        point = ShotPoint(
            id=self._next_point_id,
            x_mm=x_mm,
            y_mm=y_mm,
            radius_mm=radius_mm,
            confidence=0.5,
            source="manual",
        )
        self._next_point_id += 1
        self._current_result.points.append(point)
        self.shot_view.add_point_item(point)
        self._after_points_changed()

    def _handle_point_removed(self, shot_id: int) -> None:
        if not self._current_result:
            return
        self._current_result.points = [p for p in self._current_result.points if p.id != shot_id]
        self.shot_view.remove_point_item(shot_id)
        self._after_points_changed()

    def _handle_point_moved(self, shot_id: int, x_px: float, y_px: float) -> None:
        if not self._current_result:
            return
        point = next((p for p in self._current_result.points if p.id == shot_id), None)
        if not point:
            return
        scene_pos = QPointF(x_px, y_px)
        x_mm, y_mm = self.shot_view.scene_to_mm(scene_pos)
        point.x_mm = x_mm
        point.y_mm = y_mm
        point.source = "adjusted"
        self._after_points_changed(update_view=False)

    def _after_points_changed(self, update_view: bool = True, record_history: bool = True) -> None:
        if not self._current_result:
            return
        self._current_result.recompute_metrics()
        self._refresh_metrics()
        self._refresh_points_list()
        if record_history:
            self._push_history()
        if update_view:
            self._update_overlay()

    def _update_overlay(self) -> None:
        if not self._current_result or self._current_result.aligned_gray is None:
            return
        overlay = render_overlay(
            cv2.cvtColor(self._current_result.aligned_gray, cv2.COLOR_GRAY2BGR),
            self._current_result.points,
            self._current_result.metrics,
            self._current_result.mm_per_pixel or self._scale_model.mm_per_pixel,
            origin_px=self._current_result.origin_px or (0.0, 0.0),
            show_r50=self._pipeline.config.show_r50 if self._pipeline else True,
            show_r90=self._pipeline.config.show_r90 if self._pipeline else False,
            show_debug=self.show_problems_checkbox.isChecked(),
            debug_info=self._current_result.debug_info,
        )
        self._current_result.overlay_image = overlay
        self.shot_view.set_background(overlay)
        export_dir = self._pipeline.config.output_dir if self._pipeline else Path("results")
        self._ensure_overlay_saved(export_dir)

    def reset_view(self) -> None:
        self._current_frame = None
        self._current_result = None
        self._next_point_id = 1
        scene = self.shot_view.scene()
        if scene is not None:
            scene.clear()
        self.shot_view.clear_view()
        self._reset_metrics_table()
        self.points_list.clear()
        self._clear_history()
        self.show_problems_checkbox.setChecked(False)
        self._set_status("Сброс выполнен")

    def _reset_metrics_table(self) -> None:
        for row in range(self.metrics_table.rowCount()):
            item = self.metrics_table.item(row, 1)
            if item:
                item.setText("---")

    def _export_result(self, fmt: str) -> None:
        if not self._current_result:
            QMessageBox.information(self, "Нет данных", "Сначала обработайте изображение.")
            return
        export_settings = self.settings_manager.get("export", {})
        export_dir = Path(export_settings.get("output_dir", "results"))
        meta = {"author": export_settings.get("author", ""), "organization": export_settings.get("organization", "")}
        export_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_overlay_saved(export_dir)
        try:
            if fmt == "csv":
                summary_path, points_path = export_csv(self._current_result, export_dir)
                message = f"CSV сохранены:\n- {summary_path.name}\n- {points_path.name}"
            elif fmt == "pdf":
                pdf_path = export_pdf(self._current_result, export_dir, meta)
                message = f"PDF сохранён: {pdf_path.name}"
            elif fmt == "docx":
                docx_path = export_docx(self._current_result, export_dir, meta)
                message = f"DOCX сохранён: {docx_path.name}"
            else:
                raise ValueError(f"Неизвестный формат: {fmt}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Export failed")
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))
            return
        QMessageBox.information(self, "Готово", message)
        self._set_status(f"Экспортировано в {fmt.upper()}")

    def _ensure_overlay_saved(self, export_dir: Path) -> None:
        if not self._current_result or self._current_result.overlay_image is None:
            return
        path = self._current_result.overlay_path
        if not path:
            path = export_dir / f"{self._current_result.target_id}_overlay.png"
        else:
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not imwrite(path, self._current_result.overlay_image):
            QMessageBox.warning(self, "Ошибка", "Не удалось сохранить оверлей.")
            return
        self._current_result.overlay_path = str(path)

    def save_current_frame_as_template(self) -> None:
        if self._current_frame is None:
            QMessageBox.information(self, "Нет кадра", "Сначала снимите кадр с камеры.")
            return
        template = cv2.cvtColor(self._current_frame, cv2.COLOR_BGR2GRAY)
        calibration = self.settings_manager.get("calibration", {})
        template_path = Path(calibration.get("template_path", "app/data/template.png"))
        template_path.parent.mkdir(parents=True, exist_ok=True)
        if imwrite(template_path, template):
            calibration = {**calibration, "template_path": str(template_path)}
            self.settings_manager.set("calibration", calibration)
            self._pipeline = self._create_pipeline()
            self._set_status("Текущий кадр сохранён как эталон.")
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось сохранить эталон.")

    def _open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.settings_manager, self)
        if dialog.exec():
            self._scale_model = self._load_scale_model()
            self._pipeline = self._create_pipeline()

    def _open_calibration_wizard(self) -> None:
        wizard = CalibrationWizard(self.settings_manager, self)
        if wizard.exec():
            self._scale_model = self._load_scale_model()
            self._pipeline = self._create_pipeline()
            self._set_status("Калибровка обновлена.")

    def _handle_show_problems_changed(self, _: int) -> None:
        ui_settings = dict(self.settings_manager.get("ui", {}))
        ui_settings["show_problem_candidates"] = self.show_problems_checkbox.isChecked()
        self.settings_manager.set("ui", ui_settings)
        if self._current_result:
            self._update_overlay()

    def _run_local_split(self) -> None:
        if not self._current_result or not self._pipeline:
            QMessageBox.information(self, "Нет данных", "Сначала обработайте изображение.")
            return
        selection = self.shot_view.get_selected_roi()
        if selection is None:
            QMessageBox.information(self, "Нет области", "Зажмите Ctrl и выделите область на изображении.")
            return
        try:
            new_points = self._pipeline.split_roi(self._current_result, selection)
        except RuntimeError as exc:
            QMessageBox.warning(self, "Ошибка разделения", str(exc))
            return
        if not new_points:
            QMessageBox.information(self, "Результат", "Новых попаданий не найдено.")
            return
        max_existing_id = max((point.id for point in self._current_result.points), default=0)
        for idx, point in enumerate(new_points, start=1):
            point.id = max_existing_id + idx
            point.source = "split"
        self._current_result.points.extend(new_points)
        self.shot_view.load_points(self._current_result.points)
        self._after_points_changed()

    def _snapshot_points(self) -> List[ShotPoint]:
        if not self._current_result:
            return []
        return copy.deepcopy(self._current_result.points)

    def _clear_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._refresh_history_buttons()

    def _push_history(self) -> None:
        if not self._current_result:
            return
        snapshot = self._snapshot_points()
        if self._undo_stack and self._undo_stack[-1] == snapshot:
            return
        self._undo_stack.append(snapshot)
        max_depth = 20
        if len(self._undo_stack) > max_depth:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._refresh_history_buttons()

    def _apply_history(self, pop_from: str) -> None:
        if not self._current_result:
            return
        if pop_from == "undo":
            if len(self._undo_stack) <= 1:
                return
            last = self._undo_stack.pop()
            self._redo_stack.append(last)
            snapshot = self._undo_stack[-1]
        elif pop_from == "redo":
            if not self._redo_stack:
                return
            snapshot = self._redo_stack.pop()
            self._undo_stack.append(copy.deepcopy(snapshot))
        else:
            return
        self._apply_snapshot(snapshot)
        self._refresh_history_buttons()

    def _apply_snapshot(self, snapshot: List[ShotPoint]) -> None:
        if not self._current_result:
            return
        self._current_result.points = copy.deepcopy(snapshot)
        self.shot_view.load_points(self._current_result.points)
        self._after_points_changed(update_view=True, record_history=False)

    def _refresh_history_buttons(self) -> None:
        self.undo_button.setEnabled(len(self._undo_stack) > 1)
        self.redo_button.setEnabled(bool(self._redo_stack))



def run_app() -> None:
    logging.basicConfig(level=logging.INFO)
    app = QApplication([])
    window = MainWindow(Path("app/settings.json"))
    window.show()
    app.exec()


__all__ = ["MainWindow", "run_app"]
