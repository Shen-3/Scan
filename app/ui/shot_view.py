from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal, QRect, QSize
from PyQt6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap, QWheelEvent, QResizeEvent
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QRubberBand,
    QScrollBar,
    QSizePolicy,
)

from app.models import ShotPoint


def _np_to_qpixmap(image: np.ndarray) -> QPixmap:
    if image.ndim == 2:
        gray = np.ascontiguousarray(image)
        height, width = gray.shape
        qimage = QImage(gray.data, width, height, width, QImage.Format.Format_Grayscale8)
    else:
        if image.shape[2] == 3:
            rgb = np.ascontiguousarray(image[..., ::-1])
        elif image.shape[2] == 4:
            rgb = np.ascontiguousarray(image[..., [2, 1, 0, 3]])
        else:
            raise ValueError("Unsupported channel number")
        height, width = rgb.shape[:2]
        bytes_per_line = rgb.strides[0]
        if rgb.shape[2] == 3:
            qimage = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            qimage = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimage.copy())


class ShotPointItem(QGraphicsEllipseItem):
    """Graphics item for a single shot point with label."""

    def __init__(
        self,
        shot: ShotPoint,
        center_px: QPointF,
        radius_px: float,
        on_moved,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.shot = shot
        self.radius_px = max(3.0, radius_px)
        diameter = self.radius_px * 2.0
        self.setRect(-self.radius_px, -self.radius_px, diameter, diameter)
        self.setPos(center_px)
        self.setBrush(QColor(255, 0, 0, 80))
        self.setPen(QPen(QColor(255, 0, 0), 2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.label = QGraphicsSimpleTextItem(str(self.shot.id), self)
        self.label.setPos(self.radius_px * 0.6, -self.radius_px * 0.6)
        self._on_moved = on_moved

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self._on_moved:
                self._on_moved(self)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event) -> None:
        self.setBrush(QColor(255, 0, 0, 140))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.setBrush(QColor(255, 0, 0, 80))
        super().hoverLeaveEvent(event)


class ShotGraphicsView(QGraphicsView):
    """Interactive view for displaying targets and shot points."""

    pointAdded = pyqtSignal(float, float)  # px coordinates in scene space
    pointRemoved = pyqtSignal(int)
    pointMoved = pyqtSignal(int, float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene: QGraphicsScene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._background_item: Optional[QGraphicsPixmapItem] = None
        self._origin_marker: Optional[QGraphicsPathItem] = None
        self._pixmap_size: Optional[Tuple[int, int]] = None
        self._mm_per_pixel = 0.05
        self._origin_px = QPointF(0.0, 0.0)
        self._shot_items: Dict[int, ShotPointItem] = {}
        self._dragging = False
        self._last_mouse_pos = QPointF()
        self._editing_enabled = False
        self._rubber_band: Optional[QRubberBand] = None
        self._selection_origin = QPointF()
        self._selecting_roi = False
        self._selected_roi: Optional[Tuple[int, int, int, int]] = None
        self._auto_fit = True

    def set_scale(self, mm_per_pixel: float, origin_px: Tuple[float, float]) -> None:
        self._mm_per_pixel = mm_per_pixel
        self._origin_px = QPointF(*origin_px)
        self._update_origin_marker()

    def set_background(self, image: np.ndarray) -> None:
        pixmap = _np_to_qpixmap(image)
        if self._background_item is None:
            self._background_item = QGraphicsPixmapItem(pixmap)
            self._scene.addItem(self._background_item)
        else:
            self._background_item.setPixmap(pixmap)
        self._pixmap_size = (pixmap.width(), pixmap.height())
        self.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self._update_origin_marker()
        self.clear_roi()
        self._auto_fit = True
        self._fit_view()

    def _update_origin_marker(self) -> None:
        if self._pixmap_size is None:
            return
        path = QPainterPath()
        path.moveTo(self._origin_px.x() - 15, self._origin_px.y())
        path.lineTo(self._origin_px.x() + 15, self._origin_px.y())
        path.moveTo(self._origin_px.x(), self._origin_px.y() - 15)
        path.lineTo(self._origin_px.x(), self._origin_px.y() + 15)
        if self._origin_marker is None:
            self._origin_marker = QGraphicsPathItem(path)
            pen = QPen(QColor(0, 255, 0), 2)
            self._origin_marker.setPen(pen)
            self._scene.addItem(self._origin_marker)
        else:
            self._origin_marker.setPath(path)

    def load_points(self, points: Iterable[ShotPoint]) -> None:
        for item in list(self._shot_items.values()):
            self._scene.removeItem(item)
        self._shot_items.clear()
        for shot in points:
            self.add_point_item(shot)
        self.set_editing_enabled(bool(self._shot_items))

    def clear_view(self) -> None:
        self._scene.clear()
        self._shot_items.clear()
        self._background_item = None
        self._origin_marker = None
        self._pixmap_size = None
        self._editing_enabled = False
        self._selected_roi = None
        if self._rubber_band:
            self._rubber_band.hide()
        self._auto_fit = True

    def add_point_item(self, shot: ShotPoint) -> None:
        center_px = self.mm_to_scene(shot.x_mm, shot.y_mm)
        radius_px = shot.radius_mm / self._mm_per_pixel if self._mm_per_pixel else 6.0
        item = ShotPointItem(shot, center_px, radius_px, self._handle_item_moved)
        self._scene.addItem(item)
        self._shot_items[shot.id] = item
        self._editing_enabled = True
        self.set_editing_enabled(True)

    def remove_point_item(self, shot_id: int) -> None:
        item = self._shot_items.pop(shot_id, None)
        if item:
            self._scene.removeItem(item)
        if not self._shot_items:
            self._editing_enabled = False
            self.set_editing_enabled(False)

    def clear_roi(self) -> None:
        self._selected_roi = None
        if self._rubber_band:
            self._rubber_band.hide()

    def get_selected_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._selected_roi

    def mm_to_scene(self, x_mm: float, y_mm: float) -> QPointF:
        px = x_mm / self._mm_per_pixel + self._origin_px.x()
        py = y_mm / self._mm_per_pixel + self._origin_px.y()
        return QPointF(px, py)

    def scene_to_mm(self, pos: QPointF) -> Tuple[float, float]:
        x_mm = (pos.x() - self._origin_px.x()) * self._mm_per_pixel
        y_mm = (pos.y() - self._origin_px.y()) * self._mm_per_pixel
        return x_mm, y_mm

    def set_editing_enabled(self, enabled: bool) -> None:
        self._editing_enabled = enabled
        for item in self._shot_items.values():
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, enabled)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._selecting_roi = True
            if self._rubber_band is None:
                self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
            self._selection_origin = event.position()
            self._rubber_band.setGeometry(QRect(self._selection_origin.toPoint(), QSize()))
            self._rubber_band.show()
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging = True
            self._last_mouse_pos = event.position()
            self._auto_fit = False
            event.accept()
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        item = self._scene.itemAt(scene_pos, self.transform())
        if event.button() == Qt.MouseButton.LeftButton:
            if isinstance(item, ShotPointItem) and self._editing_enabled:
                super().mousePressEvent(event)
            elif self._editing_enabled:
                self.pointAdded.emit(scene_pos.x(), scene_pos.y())
            return
        if event.button() == Qt.MouseButton.RightButton:
            if isinstance(item, ShotPointItem) and self._editing_enabled:
                self.pointRemoved.emit(item.shot.id)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._selecting_roi and self._rubber_band:
            origin = self._selection_origin.toPoint()
            current = event.position().toPoint()
            rect = QRect(origin, current).normalized()
            self._rubber_band.setGeometry(rect)
            event.accept()
            return
        if self._dragging:
            delta = event.position() - self._last_mouse_pos
            self._last_mouse_pos = event.position()
            self._pan(delta)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._selecting_roi and self._rubber_band:
            self._selecting_roi = False
            self._rubber_band.hide()
            rect = self._rubber_band.geometry()
            if rect.width() > 5 and rect.height() > 5:
                top_left = self.mapToScene(rect.topLeft())
                bottom_right = self.mapToScene(rect.bottomRight())
                x0 = max(0, int(round(min(top_left.x(), bottom_right.x()))))
                y0 = max(0, int(round(min(top_left.y(), bottom_right.y()))))
                x1 = int(round(max(top_left.x(), bottom_right.x())))
                y1 = int(round(max(top_left.y(), bottom_right.y())))
                width = max(0, x1 - x0)
                height = max(0, y1 - y0)
                if width > 0 and height > 0 and self._pixmap_size:
                    width = min(width, self._pixmap_size[0] - x0)
                    height = min(height, self._pixmap_size[1] - y0)
                    self._selected_roi = (x0, y0, width, height)
                else:
                    self._selected_roi = None
            else:
                self._selected_roi = None
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8
        self._auto_fit = False
        self.scale(factor, factor)

    def _pan(self, delta) -> None:
        h_scroll: Optional[QScrollBar] = self.horizontalScrollBar()
        if h_scroll is not None:
            h_scroll.setValue(h_scroll.value() - delta.x())
        v_scroll: Optional[QScrollBar] = self.verticalScrollBar()
        if v_scroll is not None:
            v_scroll.setValue(v_scroll.value() - delta.y())

    def _handle_item_moved(self, item: ShotPointItem) -> None:
        pos = item.scenePos()
        if self._editing_enabled:
            self.pointMoved.emit(item.shot.id, pos.x(), pos.y())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._auto_fit:
            self._fit_view()

    def _fit_view(self) -> None:
        rect = self.sceneRect()
        if rect.width() > 0 and rect.height() > 0:
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
