"""Debugger técnico do especialista de assinatura da face."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget


TRANSFORM_LABELS = {
    "none": "nenhuma",
    "rot180": "rotação 180°",
    "flip_horizontal": "espelhamento horizontal",
    "flip_vertical": "espelhamento vertical",
    "rot90_clockwise": "rotação 90° horária",
    "rot90_counterclockwise": "rotação 90° anti-horária",
}


class InvertedFaceDebuggerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setMinimumWidth(650)
        self.is_active = False
        self.is_defect = False
        self.score = 0.0
        self.tolerance = 0.43
        self.classification = "SEM DADOS"
        self.signature_strength = 0.0
        self.direct_similarity = 1.0
        self.feature_loss = 0.0
        self.extra_structure = 0.0
        self.topology_mismatch = 0.0
        self.orientation_mismatch = 0.0
        self.alternate_face_signal = 0.0
        self.transform_gain = 0.0
        self.best_transform = "none"
        self.best_transform_similarity = 0.0
        self.expected_angle = 0.0
        self.observed_angle = 0.0
        self.changed_coverage = 0.0
        self.roi_width = 0
        self.roi_height = 0
        self.reason = ""
        self.reference_view = None
        self.test_view = None
        self.reconstruction_view = None

    @staticmethod
    def _copy(value):
        if isinstance(value, np.ndarray) and value.size > 0:
            return value.copy()
        return None

    def update_data(self, detail: dict):
        if (
            not detail
            or detail.get("inverted_comparison_mode") != "inverted_face_signature"
            or not detail.get("inverted_active", False)
        ):
            self.is_active = False
            self.update()
            return

        self.is_active = True
        self.is_defect = bool(detail.get("inverted_is_defect", False))
        self.score = float(detail.get("inverted_score", 0.0))
        self.tolerance = float(detail.get("inverted_tolerance", 0.43))
        self.classification = str(
            detail.get("inverted_classification", "SEM CLASSIFICAÇÃO")
        )
        self.signature_strength = float(
            detail.get("inverted_signature_strength", 0.0)
        )
        self.direct_similarity = float(
            detail.get("inverted_direct_similarity", 1.0)
        )
        self.feature_loss = float(detail.get("inverted_feature_loss", 0.0))
        self.extra_structure = float(
            detail.get("inverted_extra_structure", 0.0)
        )
        self.topology_mismatch = float(
            detail.get("inverted_topology_mismatch", 0.0)
        )
        self.orientation_mismatch = float(
            detail.get("inverted_orientation_mismatch", 0.0)
        )
        self.alternate_face_signal = float(
            detail.get("inverted_alternate_face_signal", 0.0)
        )
        self.transform_gain = float(detail.get("inverted_transform_gain", 0.0))
        self.best_transform = str(detail.get("inverted_best_transform", "none"))
        self.best_transform_similarity = float(
            detail.get("inverted_best_transform_similarity", 0.0)
        )
        self.expected_angle = float(detail.get("inverted_expected_angle", 0.0))
        self.observed_angle = float(detail.get("inverted_observed_angle", 0.0))
        self.changed_coverage = float(
            detail.get("inverted_changed_coverage", 0.0)
        )
        self.roi_width = int(detail.get("inverted_roi_width", 0) or 0)
        self.roi_height = int(detail.get("inverted_roi_height", 0) or 0)
        self.reason = str(detail.get("inverted_reason", ""))
        self.reference_view = self._copy(detail.get("inverted_reference_view"))
        self.test_view = self._copy(detail.get("inverted_test_view"))
        self.reconstruction_view = self._copy(
            detail.get("inverted_reconstruction_view")
        )
        self.update()

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
        painter.drawText(8, 18, "ASSINATURA DA FACE • MOTOR INVERTIDO")

        if not self.is_active:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Motor INVERTIDO inativo para esta categoria",
            )
            painter.end()
            return

        padding = 10
        spacing = 7
        top = 44
        footer_height = 114
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
            "1. GABARITO • ASSINATURA ESPERADA" + roi_label,
            QColor("#4ade80"),
        )
        self._draw_image(
            painter,
            self.test_view,
            rects[1],
            "2. TESTE • ASSINATURA OBSERVADA" + roi_label,
            QColor("#46d9ff"),
        )
        self._draw_image(
            painter,
            self.reconstruction_view,
            rects[2],
            "3. RECONSTRUÇÃO • EVIDÊNCIA DE INVERSÃO",
            QColor("#f5c518"),
        )

        status_color = QColor("#ff6262") if self.is_defect else QColor("#4ade80")
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(status_color)
        painter.drawText(
            padding,
            height - 96,
            self._elide(
                painter,
                f"Resultado: {self.classification} • força da assinatura {self.signature_strength:.0%}",
                width - padding * 2,
            ),
        )

        gauge_y = height - 81
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
            height - 59,
            self._elide(
                painter,
                "Legenda: VERDE = coincide • AMARELO = esperado ausente • VERMELHO = estrutura extra • setas = orientação dominante",
                width - padding * 2,
            ),
        )
        painter.setPen(status_color)
        metrics = (
            f"Score {self.score:.0%}/{self.tolerance:.0%} • marca perdida {self.feature_loss:.0%} • "
            f"topologia {self.topology_mismatch:.0%} • orientação {self.orientation_mismatch:.0%} • "
            f"face alternativa {self.alternate_face_signal:.0%}"
        )
        painter.drawText(
            padding,
            height - 42,
            self._elide(painter, metrics, width - padding * 2),
        )
        transform_label = TRANSFORM_LABELS.get(
            self.best_transform,
            self.best_transform,
        )
        painter.setPen(QColor("#f5c518"))
        details = (
            f"Similaridade direta {self.direct_similarity:.0%} • ângulo esperado {self.expected_angle:.0f}° • "
            f"observado {self.observed_angle:.0f}° • melhor transformação: {transform_label} "
            f"{self.best_transform_similarity:.0%} (+{self.transform_gain:.0%}) • área {self.changed_coverage:.0%}"
        )
        painter.drawText(
            padding,
            height - 25,
            self._elide(painter, details, width - padding * 2),
        )
        painter.setPen(QColor("#d0d0d0"))
        painter.drawText(
            padding,
            height - 8,
            self._elide(painter, self.reason, width - padding * 2),
        )
        painter.end()


__all__ = ["InvertedFaceDebuggerWidget"]
