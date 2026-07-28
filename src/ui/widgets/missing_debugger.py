"""Debugger visual do motor exclusivo para a categoria FALTANDO."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class MissingDebuggerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setMinimumWidth(600)
        self.is_active = False
        self.is_defect = False
        self.score = 0.0
        self.tolerance = 0.42
        self.structure_loss = 0.0
        self.coverage = 0.0
        self.appearance_loss = 0.0
        self.background_exposure = 0.0
        self.retention = 1.0
        self.alignment_score = 0.0
        self.roi_width = 0
        self.roi_height = 0
        self.reason = ""
        self.reference_view = None
        self.test_view = None
        self.reconstruction_view = None

    def update_data(self, detail: dict):
        if (
            not detail
            or detail.get("missing_comparison_mode") != "missing_component"
            or not detail.get("missing_active", False)
        ):
            self.is_active = False
            self.update()
            return

        self.is_active = True
        self.is_defect = bool(detail.get("missing_is_defect", False))
        self.score = float(detail.get("missing_score", 0.0))
        self.tolerance = float(detail.get("missing_tolerance", 0.42))
        self.structure_loss = float(detail.get("missing_structure_loss", 0.0))
        self.coverage = float(detail.get("missing_coverage", 0.0))
        self.appearance_loss = float(detail.get("missing_appearance_loss", 0.0))
        self.background_exposure = float(
            detail.get("missing_background_exposure", 0.0)
        )
        self.retention = float(detail.get("missing_presence_retention", 1.0))
        self.alignment_score = float(detail.get("missing_alignment_score", 0.0))
        self.roi_width = int(detail.get("missing_roi_width", 0) or 0)
        self.roi_height = int(detail.get("missing_roi_height", 0) or 0)
        self.reason = str(detail.get("missing_reason", ""))
        self.reference_view = self._copy(detail.get("missing_reference_view"))
        self.test_view = self._copy(detail.get("missing_test_view"))
        self.reconstruction_view = self._copy(
            detail.get("missing_reconstruction_view")
        )
        self.update()

    @staticmethod
    def _copy(value):
        if isinstance(value, np.ndarray) and value.size > 0:
            return value.copy()
        return None

    @staticmethod
    def _qimage(image_bgr: np.ndarray) -> QImage:
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
    def _elide(painter: QPainter, text: str, width: int) -> str:
        return painter.fontMetrics().elidedText(
            text,
            Qt.TextElideMode.ElideRight,
            max(20, width),
        )

    def _draw_image(
        self,
        painter: QPainter,
        image,
        rect: QRectF,
        title: str,
        title_color: QColor,
    ) -> None:
        painter.setPen(QPen(QColor("#343434"), 1))
        painter.setBrush(QColor("#070707"))
        painter.drawRect(rect)
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(title_color)
        painter.drawText(
            QRectF(rect.x(), rect.y() - 14, rect.width(), 12),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            title,
        )
        if image is None or image.size == 0:
            painter.setPen(QColor("#555555"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "SEM DADOS")
            return
        qimage = self._qimage(image)
        scaled = qimage.scaled(
            max(1, int(rect.width() - 4)),
            max(1, int(rect.height() - 4)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        image_x = rect.x() + (rect.width() - scaled.width()) / 2
        image_y = rect.y() + (rect.height() - scaled.height()) / 2
        painter.drawImage(int(image_x), int(image_y), scaled)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = self.width()
        height = self.height()
        painter.fillRect(0, 0, width, height, QColor("#101010"))

        painter.setPen(QColor("#f5f5f5"))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.drawText(8, 18, "PRESENÇA DO COMPONENTE • MOTOR FALTANDO")

        if not self.is_active:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Motor FALTANDO inativo para esta categoria",
            )
            painter.end()
            return

        padding = 10
        spacing = 7
        top = 42
        footer_height = 94
        available_width = width - padding * 2 - spacing * 2
        box_width = available_width / 3.0
        box_height = max(40.0, height - top - footer_height)
        rects = [
            QRectF(padding, top, box_width, box_height),
            QRectF(padding + box_width + spacing, top, box_width, box_height),
            QRectF(padding + box_width * 2 + spacing * 2, top, box_width, box_height),
        ]
        roi_label = (
            f" • {self.roi_width}×{self.roi_height}px"
            if self.roi_width and self.roi_height
            else ""
        )
        self._draw_image(
            painter,
            self.reference_view,
            rects[0],
            "1. GABARITO • COMPONENTE ESPERADO" + roi_label,
            QColor("#4ade80"),
        )
        self._draw_image(
            painter,
            self.test_view,
            rects[1],
            "2. TESTE • PRESENÇA ENCONTRADA" + roi_label,
            QColor("#46d9ff"),
        )
        self._draw_image(
            painter,
            self.reconstruction_view,
            rects[2],
            "3. RECONSTRUÇÃO • REGIÃO AUSENTE",
            QColor("#f5c518"),
        )

        gauge_y = height - 78
        gauge_x = width * 0.10
        gauge_width = width * 0.80
        gauge_height = 8
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#303030"))
        painter.drawRoundedRect(
            QRectF(gauge_x, gauge_y, gauge_width, gauge_height),
            4,
            4,
        )
        status_color = QColor("#ff6262") if self.is_defect else QColor("#4ade80")
        painter.setBrush(status_color)
        painter.drawRoundedRect(
            QRectF(gauge_x, gauge_y, gauge_width * min(1.0, self.score), gauge_height),
            4,
            4,
        )
        cutoff_x = gauge_x + gauge_width * min(1.0, self.tolerance)
        painter.setPen(QPen(QColor("#f5c518"), 2))
        painter.drawLine(
            int(cutoff_x),
            int(gauge_y - 3),
            int(cutoff_x),
            int(gauge_y + gauge_height + 3),
        )

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#a6a6a6"))
        painter.drawText(
            padding,
            height - 55,
            "Legenda: VERDE = presença preservada • VERMELHO = região esperada ausente • AMARELO = limite esperado",
        )
        painter.setPen(status_color)
        metrics = (
            f"Score {self.score:.0%}/{self.tolerance:.0%} • estrutura ausente "
            f"{self.structure_loss:.0%} • cobertura perdida {self.coverage:.0%} • "
            f"aparência perdida {self.appearance_loss:.0%}"
        )
        painter.drawText(
            padding,
            height - 37,
            self._elide(painter, metrics, width - padding * 2),
        )
        painter.setPen(QColor("#f5c518"))
        details = (
            f"Retenção {self.retention:.0%} • fundo/padding exposto "
            f"{self.background_exposure:.0%} • alinhamento {self.alignment_score:.2f}"
        )
        painter.drawText(
            padding,
            height - 20,
            self._elide(painter, details, width - padding * 2),
        )
        painter.setPen(QColor("#d0d0d0"))
        painter.drawText(
            padding,
            height - 4,
            self._elide(painter, self.reason, width - padding * 2),
        )
        painter.end()


__all__ = ["MissingDebuggerWidget"]
