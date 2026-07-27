"""Debugger técnico do embedding semântico e de sua reconstrução espacial."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class SemanticDNAWidget(QWidget):
    """Exibe vetores 128D, delta por dimensão e reconstrução espacial 4x4."""

    GROUP_RANGES = (
        ("EDGE", 0, 16),
        ("LUMA", 16, 32),
        ("HUE", 32, 64),
        ("SAT", 64, 96),
        ("VAL", 96, 128),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setMinimumWidth(600)

        self.is_active = False
        self.sem_loss = 0.0
        self.cosine_distance = 0.0
        self.query_emb = None
        self.ref_emb = None
        self.delta_vector = None
        self.debug = {}
        self.reconstruction_view = None

    def update_data(self, detail: dict):
        if not detail or "semantic_loss" not in detail:
            self.is_active = False
            self.update()
            return

        query = detail.get("query_emb") or []
        reference = detail.get("ref_emb") or []
        if not query or not reference:
            self.is_active = False
            self.update()
            return

        self.is_active = True
        self.sem_loss = float(detail.get("semantic_loss", 0.0))
        self.cosine_distance = float(
            detail.get("semantic_distance_cosine", self.sem_loss / 2.5)
        )
        self.query_emb = np.asarray(query, dtype=np.float32)
        self.ref_emb = np.asarray(reference, dtype=np.float32)
        self.debug = detail.get("semantic_debug") or {}

        delta = detail.get("semantic_delta")
        if delta:
            self.delta_vector = np.asarray(delta, dtype=np.float32)
        else:
            absolute = np.abs(self.query_emb - self.ref_emb)
            scale = np.abs(self.query_emb) + np.abs(self.ref_emb) + 0.04
            self.delta_vector = np.clip(absolute / scale, 0.0, 1.0)

        reconstruction = detail.get("semantic_reconstruction_view")
        self.reconstruction_view = (
            np.asarray(reconstruction).copy()
            if isinstance(reconstruction, np.ndarray) and reconstruction.size > 0
            else None
        )
        self.update()

    @staticmethod
    def _qimage_from_bgr(image_bgr: np.ndarray) -> QImage:
        contiguous = np.ascontiguousarray(image_bgr)
        rgb = cv2.cvtColor(contiguous, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        return QImage(
            rgb.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        ).copy()

    @staticmethod
    def _signal_color(value: float) -> QColor:
        value = float(np.clip(value, 0.0, 1.0))
        if value < 0.5:
            ratio = value / 0.5
            return QColor(
                15,
                int(70 + 150 * ratio),
                int(120 + 135 * ratio),
            )
        ratio = (value - 0.5) / 0.5
        return QColor(
            int(30 + 225 * ratio),
            int(220 - 35 * ratio),
            int(255 - 210 * ratio),
        )

    @staticmethod
    def _delta_color(value: float) -> QColor:
        value = float(np.clip(value, 0.0, 1.0))
        if value < 0.5:
            ratio = value / 0.5
            return QColor(int(170 * ratio), 5, 20)
        ratio = (value - 0.5) / 0.5
        return QColor(170 + int(85 * ratio), int(190 * ratio), 10)

    def _normalize_embedding_value(self, vector: np.ndarray, index: int) -> float:
        for _, start, end in self.GROUP_RANGES:
            if start <= index < end:
                combined = np.concatenate(
                    [self.ref_emb[start:end], self.query_emb[start:end]]
                )
                minimum = float(np.min(combined))
                maximum = float(np.max(combined))
                if maximum <= minimum:
                    return 0.0
                return float((vector[index] - minimum) / (maximum - minimum))
        return 0.0

    def _draw_vector(
        self,
        painter: QPainter,
        rect: QRectF,
        vector: np.ndarray,
        title: str,
        delta_mode: bool = False,
    ) -> None:
        painter.setPen(QColor("#a6a6a6") if not delta_mode else QColor("#ff7878"))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(
            QRectF(rect.x(), rect.y(), rect.width(), 13),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            title,
        )

        bar_rect = QRectF(rect.x(), rect.y() + 14, rect.width(), rect.height() - 14)
        painter.setPen(QPen(QColor("#303030"), 1))
        painter.setBrush(QColor("#070707"))
        painter.drawRect(bar_rect)

        feature_width = bar_rect.width() / max(len(vector), 1)
        painter.setPen(Qt.PenStyle.NoPen)
        for index, raw_value in enumerate(vector):
            value = (
                float(np.clip(raw_value, 0.0, 1.0))
                if delta_mode
                else self._normalize_embedding_value(vector, index)
            )
            painter.setBrush(
                self._delta_color(value) if delta_mode else self._signal_color(value)
            )
            painter.drawRect(
                QRectF(
                    bar_rect.x() + index * feature_width,
                    bar_rect.y(),
                    feature_width + 0.8,
                    bar_rect.height(),
                )
            )

        painter.setFont(QFont("Consolas", 6, QFont.Weight.Bold))
        for label, start, end in self.GROUP_RANGES:
            start_x = bar_rect.x() + start * feature_width
            end_x = bar_rect.x() + end * feature_width
            painter.setPen(QPen(QColor("#f5c518"), 1))
            painter.drawLine(
                int(start_x),
                int(bar_rect.y()),
                int(start_x),
                int(bar_rect.bottom()),
            )
            painter.setPen(QColor("#e2e2e2"))
            painter.drawText(
                QRectF(start_x, bar_rect.y() + 1, end_x - start_x, 10),
                Qt.AlignmentFlag.AlignCenter,
                label,
            )

    def _draw_reconstruction(
        self,
        painter: QPainter,
        rect: QRectF,
    ) -> None:
        painter.setPen(QColor("#f5c518"))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(
            QRectF(rect.x(), rect.y(), rect.width(), 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "RECONSTRUÇÃO ESPACIAL • GRID 4×4",
        )

        image_rect = QRectF(rect.x(), rect.y() + 16, rect.width(), rect.height() - 16)
        painter.setPen(QPen(QColor("#303030"), 1))
        painter.setBrush(QColor("#070707"))
        painter.drawRect(image_rect)

        if self.reconstruction_view is not None:
            qimage = self._qimage_from_bgr(self.reconstruction_view)
            scaled = qimage.scaled(
                max(1, int(image_rect.width() - 4)),
                max(1, int(image_rect.height() - 4)),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            image_x = image_rect.x() + (image_rect.width() - scaled.width()) / 2
            image_y = image_rect.y() + (image_rect.height() - scaled.height()) / 2
            painter.drawImage(int(image_x), int(image_y), scaled)
            target = QRectF(image_x, image_y, scaled.width(), scaled.height())
        else:
            target = image_rect.adjusted(4, 4, -4, -4)

        spatial = self.debug.get("spatial", {})
        grid = spatial.get("combined_delta_grid") or []
        if len(grid) == 4 and all(len(row) == 4 for row in grid):
            cell_width = target.width() / 4.0
            cell_height = target.height() / 4.0
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            for row in range(4):
                for column in range(4):
                    value = float(grid[row][column])
                    cell = QRectF(
                        target.x() + column * cell_width,
                        target.y() + row * cell_height,
                        cell_width,
                        cell_height,
                    )
                    painter.setPen(QPen(QColor("#c0c0c0"), 1))
                    painter.drawRect(cell)
                    painter.setPen(
                        QColor("#ffffff") if value >= 0.32 else QColor("#b0b0b0")
                    )
                    painter.drawText(
                        cell,
                        Qt.AlignmentFlag.AlignCenter,
                        f"{value:.2f}",
                    )

    @staticmethod
    def _elide(painter: QPainter, text: str, width: int) -> str:
        return painter.fontMetrics().elidedText(
            text,
            Qt.TextElideMode.ElideRight,
            max(width, 20),
        )

    def _telemetry_lines(self) -> list[str]:
        groups = self.debug.get("groups", {})
        dominant = self.debug.get("dominant_group", "unknown")
        spatial = self.debug.get("spatial", {})
        peak = spatial.get("peak_cell") or {}
        approximate_box = spatial.get("approximate_box")
        top_dimensions = self.debug.get("top_dimensions") or []

        group_parts = []
        for name, _, _ in self.GROUP_RANGES:
            full_name = {
                "EDGE": "edge_density",
                "LUMA": "brightness",
                "HUE": "hue_histogram",
                "SAT": "saturation_histogram",
                "VAL": "value_histogram",
            }[name]
            value = groups.get(full_name, {}).get("relative_divergence", 0.0)
            group_parts.append(f"{name}={value:.2f}")

        top_text = ", ".join(
            f"{item.get('label', '?')}:{item.get('signed_delta', 0.0):+.3f}"
            for item in top_dimensions[:5]
        )
        peak_text = (
            f"R{peak.get('row', 0)}C{peak.get('column', 0)}"
            f"={peak.get('value', 0.0):.2f}"
        )

        return [
            (
                f"schema={self.debug.get('schema', 'legacy')} • "
                f"cos={self.cosine_distance:.4f} • loss={self.sem_loss:.1%} • "
                f"dominante={dominant} • pico={peak_text} • box={approximate_box}"
            ),
            "grupos: " + " | ".join(group_parts),
            "top_delta: " + (top_text or "nenhuma divergência relevante"),
        ]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        painter.fillRect(0, 0, width, height, QColor("#101010"))

        painter.setPen(QColor("#f5f5f5"))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.drawText(9, 18, "DEBUG SEMÂNTICO • EMBEDDING 128D + RECONSTRUÇÃO")

        if not self.is_active or self.query_emb is None or self.ref_emb is None:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Motor semântico inativo",
            )
            painter.end()
            return

        padding = 10
        top = 30
        telemetry_height = 62
        content_height = max(120, height - top - telemetry_height - 8)
        left_width = max(300.0, (width - padding * 3) * 0.66)
        right_width = width - padding * 3 - left_width

        left_rect = QRectF(padding, top, left_width, content_height)
        right_rect = QRectF(
            padding * 2 + left_width,
            top,
            right_width,
            content_height,
        )

        block_spacing = 7
        block_height = (left_rect.height() - block_spacing * 2) / 3.0
        self._draw_vector(
            painter,
            QRectF(left_rect.x(), left_rect.y(), left_rect.width(), block_height),
            self.ref_emb,
            "1. REFERÊNCIA • VETOR SALVO/GERADO",
        )
        self._draw_vector(
            painter,
            QRectF(
                left_rect.x(),
                left_rect.y() + block_height + block_spacing,
                left_rect.width(),
                block_height,
            ),
            self.query_emb,
            "2. TESTE • VETOR DIGITALIZADO",
        )
        self._draw_vector(
            painter,
            QRectF(
                left_rect.x(),
                left_rect.y() + (block_height + block_spacing) * 2,
                left_rect.width(),
                block_height,
            ),
            self.delta_vector,
            "3. DELTA RELATIVO • DIMENSÃO POR DIMENSÃO",
            delta_mode=True,
        )
        self._draw_reconstruction(painter, right_rect)

        telemetry_y = top + content_height + 5
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        colors = (QColor("#f5c518"), QColor("#a6a6a6"), QColor("#ff7878"))
        for index, line in enumerate(self._telemetry_lines()):
            painter.setPen(colors[index])
            painter.drawText(
                padding,
                int(telemetry_y + 16 + index * 16),
                self._elide(painter, line, width - padding * 2),
            )

        painter.end()
