"""Debugger visual do motor de expectativa da ROI para FALTANDO."""

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
        self.setMinimumWidth(650)
        self.is_active = False
        self.is_defect = False
        self.score = 0.0
        self.tolerance = 0.40
        self.expectation_mode = "unknown"
        self.classification = "SEM DADOS"
        self.structure_loss = 0.0
        self.extra_structure = 0.0
        self.coverage = 0.0
        self.appearance_loss = 0.0
        self.background_exposure = 0.0
        self.retention = 1.0
        self.direct_similarity = 1.0
        self.best_similarity = 1.0
        self.displacement_dx = 0.0
        self.displacement_dy = 0.0
        self.displacement_pixels = 0.0
        self.reference_distinctness = 0.0
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
            or detail.get("missing_comparison_mode")
            not in {"missing_component", "roi_expectation"}
            or not detail.get("missing_active", False)
        ):
            self.is_active = False
            self.update()
            return

        self.is_active = True
        self.is_defect = bool(detail.get("missing_is_defect", False))
        self.score = float(detail.get("missing_score", 0.0))
        self.tolerance = float(detail.get("missing_tolerance", 0.40))
        self.expectation_mode = str(
            detail.get("missing_expectation_mode", "structure")
        ).lower()
        self.classification = str(
            detail.get("missing_classification", "SEM CLASSIFICAÇÃO")
        )
        self.structure_loss = float(detail.get("missing_structure_loss", 0.0))
        self.extra_structure = float(detail.get("missing_extra_structure", 0.0))
        self.coverage = float(
            detail.get(
                "missing_changed_coverage",
                detail.get("missing_coverage", 0.0),
            )
        )
        self.appearance_loss = float(detail.get("missing_appearance_loss", 0.0))
        self.background_exposure = float(
            detail.get("missing_background_exposure", 0.0)
        )
        self.retention = float(detail.get("missing_presence_retention", 1.0))
        self.direct_similarity = float(detail.get("missing_direct_similarity", 1.0))
        self.best_similarity = float(detail.get("missing_best_similarity", 1.0))
        self.displacement_dx = float(detail.get("missing_displacement_dx", 0.0))
        self.displacement_dy = float(detail.get("missing_displacement_dy", 0.0))
        self.displacement_pixels = float(
            detail.get("missing_displacement_pixels", 0.0)
        )
        self.reference_distinctness = float(
            detail.get("missing_reference_distinctness", 0.0)
        )
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

    def _expectation_label(self) -> str:
        if self.expectation_mode == "background":
            return "FUNDO LIVRE"
        if self.expectation_mode == "structure":
            return "ESTRUTURA ESPERADA"
        return "NÃO DEFINIDA"

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = self.width()
        height = self.height()
        painter.fillRect(0, 0, width, height, QColor("#101010"))

        painter.setPen(QColor("#f5f5f5"))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.drawText(8, 18, "EXPECTATIVA DA ROI • MOTOR FALTANDO")

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
        top = 44
        footer_height = 112
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
        expectation = self._expectation_label()
        self._draw_image(
            painter,
            self.reference_view,
            rects[0],
            f"1. GABARITO • {expectation}" + roi_label,
            QColor("#4ade80"),
        )
        self._draw_image(
            painter,
            self.test_view,
            rects[1],
            "2. TESTE • CONTEÚDO RECEBIDO NA ROI" + roi_label,
            QColor("#46d9ff"),
        )
        self._draw_image(
            painter,
            self.reconstruction_view,
            rects[2],
            "3. RECONSTRUÇÃO • QUEBRA DA EXPECTATIVA",
            QColor("#f5c518"),
        )

        status_color = QColor("#ff6262") if self.is_defect else QColor("#4ade80")
        classification_color = QColor("#ff6262") if self.is_defect else QColor("#4ade80")
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(classification_color)
        painter.drawText(
            padding,
            height - 94,
            self._elide(
                painter,
                f"Expectativa: {expectation} • Resultado: {self.classification}",
                width - padding * 2,
            ),
        )

        gauge_y = height - 80
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
            height - 58,
            self._elide(
                painter,
                "Legenda: VERDE = conforme • VERMELHO = divergência • AMARELO = limite esperado • CIANO = direção/localização",
                width - padding * 2,
            ),
        )
        painter.setPen(status_color)
        metrics = (
            f"Score {self.score:.0%}/{self.tolerance:.0%} • divergência da ROI "
            f"{self.coverage:.0%} • estrutura ausente {self.structure_loss:.0%} • "
            f"estrutura extra {self.extra_structure:.0%} • aparência {self.appearance_loss:.0%}"
        )
        painter.drawText(
            padding,
            height - 41,
            self._elide(painter, metrics, width - padding * 2),
        )
        painter.setPen(QColor("#f5c518"))
        details = (
            f"Similaridade na posição {self.direct_similarity:.0%} • melhor próxima "
            f"{self.best_similarity:.0%} • deslocamento X:{self.displacement_dx:+.1f}px "
            f"Y:{self.displacement_dy:+.1f}px ({self.displacement_pixels:.1f}px) • "
            f"fundo exposto {self.background_exposure:.0%}"
        )
        painter.drawText(
            padding,
            height - 24,
            self._elide(painter, details, width - padding * 2),
        )
        painter.setPen(QColor("#d0d0d0"))
        painter.drawText(
            padding,
            height - 7,
            self._elide(painter, self.reason, width - padding * 2),
        )
        painter.end()


__all__ = ["MissingDebuggerWidget"]
