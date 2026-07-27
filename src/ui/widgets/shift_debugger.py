"""Debugger visual dedicado ao fluxo, expansão e vazamento de adesivo."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class ShiftDebuggerWidget(QWidget):
    """Mantém o nome histórico do widget, mas exibe telemetria de adesivo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setMinimumWidth(600)

        self.is_active = False
        self.is_defect = False
        self.reason = ""
        self.adhesive_score = 0.0
        self.tolerance = 0.32
        self.excess_coverage = 0.0
        self.padding_overlap = 0.0
        self.area_growth_ratio = 0.0
        self.spread_growth_ratio = 0.0
        self.lower_leakage_ratio = 0.0
        self.reference_area_pct = 0.0
        self.test_area_pct = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.shift_pixels = 0.0
        self.shift_pct = 0.0
        self.direction = "ESTÁVEL"
        self.alignment_score = 0.0
        self.alignment_shift = (0.0, 0.0)
        self.roi_width = 0
        self.roi_height = 0
        self.reference_view = None
        self.test_view = None
        self.flow_view = None

    def update_data(self, detail: dict):
        if (
            not detail
            or detail.get("adhesive_comparison_mode") != "adhesive_flow"
            or not detail.get("shift_active", False)
        ):
            self.is_active = False
            self.update()
            return

        self.is_active = True
        self.is_defect = bool(detail.get("adhesive_is_defect", False))
        self.reason = str(detail.get("adhesive_reason", ""))
        self.adhesive_score = float(detail.get("adhesive_score", 0.0))
        self.tolerance = float(detail.get("adhesive_tolerance", 0.32))
        self.excess_coverage = float(detail.get("excess_coverage", 0.0))
        self.padding_overlap = float(detail.get("padding_overlap", 0.0))
        self.area_growth_ratio = float(detail.get("area_growth_ratio", 0.0))
        self.spread_growth_ratio = float(detail.get("spread_growth_ratio", 0.0))
        self.lower_leakage_ratio = float(detail.get("lower_leakage_ratio", 0.0))
        self.reference_area_pct = float(detail.get("reference_area_pct", 0.0))
        self.test_area_pct = float(detail.get("test_area_pct", 0.0))
        self.dx = float(detail.get("adhesive_dx", 0.0))
        self.dy = float(detail.get("adhesive_dy", 0.0))
        self.shift_pixels = float(detail.get("adhesive_shift_pixels", 0.0))
        self.shift_pct = float(detail.get("adhesive_shift_pct", 0.0))
        self.direction = str(detail.get("adhesive_direction", "ESTÁVEL"))
        self.alignment_score = float(detail.get("adhesive_alignment_score", 0.0))
        self.alignment_shift = tuple(
            detail.get("adhesive_alignment_shift", (0.0, 0.0))
        )
        self.roi_width = int(detail.get("adhesive_roi_width", 0) or 0)
        self.roi_height = int(detail.get("adhesive_roi_height", 0) or 0)

        self.reference_view = self._copy_image(
            detail.get("adhesive_reference_view")
        )
        self.test_view = self._copy_image(detail.get("adhesive_test_view"))
        self.flow_view = self._copy_image(detail.get("adhesive_flow_view"))
        self.update()

    @staticmethod
    def _copy_image(value):
        if isinstance(value, np.ndarray) and value.size > 0:
            return value.copy()
        return None

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

    def _draw_image(
        self,
        painter: QPainter,
        image_bgr: np.ndarray | None,
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

        if image_bgr is None or image_bgr.size == 0:
            painter.setPen(QColor("#555555"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "SEM DADOS")
            return

        qimage = self._qimage_from_bgr(image_bgr)
        scaled = qimage.scaled(
            max(1, int(rect.width() - 4)),
            max(1, int(rect.height() - 4)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        image_x = rect.x() + (rect.width() - scaled.width()) / 2
        image_y = rect.y() + (rect.height() - scaled.height()) / 2
        painter.drawImage(int(image_x), int(image_y), scaled)

    @staticmethod
    def _elide(painter: QPainter, text: str, width: int) -> str:
        return painter.fontMetrics().elidedText(
            text,
            Qt.TextElideMode.ElideRight,
            max(20, width),
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        painter.fillRect(0, 0, width, height, QColor("#101010"))

        painter.setPen(QColor("#f5f5f5"))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.drawText(8, 18, "FLUXO DE ADESIVO • EXPANSÃO, PADDING E VAZAMENTO")

        if not self.is_active:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Motor de adesivo inativo para esta categoria",
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

        reference_rect = QRectF(padding, top, box_width, box_height)
        test_rect = QRectF(
            padding + box_width + spacing,
            top,
            box_width,
            box_height,
        )
        flow_rect = QRectF(
            padding + box_width * 2 + spacing * 2,
            top,
            box_width,
            box_height,
        )

        roi_label = ""
        if self.roi_width > 0 and self.roi_height > 0:
            roi_label = f" • {self.roi_width}×{self.roi_height}px"

        self._draw_image(
            painter,
            self.reference_view,
            reference_rect,
            "1. GABARITO • MASSA ESPERADA" + roi_label,
            QColor("#4ade80"),
        )
        self._draw_image(
            painter,
            self.test_view,
            test_rect,
            "2. TESTE • MASSA ENCONTRADA" + roi_label,
            QColor("#46d9ff"),
        )
        self._draw_image(
            painter,
            self.flow_view,
            flow_rect,
            "3. RECONSTRUÇÃO • EXCESSO E FLUXO",
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

        gauge_max = max(0.65, self.tolerance * 1.8, self.adhesive_score * 1.08)
        gauge_max = min(1.0, gauge_max)
        fill_ratio = min(1.0, self.adhesive_score / max(gauge_max, 1e-6))
        status_color = QColor("#ff6262") if self.is_defect else QColor("#4ade80")
        painter.setBrush(status_color)
        painter.drawRoundedRect(
            QRectF(gauge_x, gauge_y, gauge_width * fill_ratio, gauge_height),
            4,
            4,
        )

        cutoff_x = gauge_x + gauge_width * min(
            1.0,
            self.tolerance / max(gauge_max, 1e-6),
        )
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
            (
                "Legenda: VERDE = adesivo estável • VERMELHO = excesso • "
                "AMARELO = excesso sobre padding • seta = fluxo da massa"
            ),
        )

        painter.setPen(status_color)
        metrics = (
            f"Score {self.adhesive_score:.0%}/{self.tolerance:.0%} • "
            f"Excesso {self.excess_coverage:.1%} • "
            f"Padding {self.padding_overlap:.1%} • "
            f"Área {self.reference_area_pct:.1%}→{self.test_area_pct:.1%} • "
            f"Expansão {self.area_growth_ratio:.0%} • "
            f"Espalhamento {self.spread_growth_ratio:.0%}"
        )
        painter.drawText(
            padding,
            height - 37,
            self._elide(painter, metrics, width - padding * 2),
        )

        painter.setPen(QColor("#f5c518"))
        flow = (
            f"Fluxo {self.direction} • centro X:{self.dx:+.1f}px Y:{self.dy:+.1f}px "
            f"({self.shift_pixels:.1f}px) • vazamento inferior "
            f"{self.lower_leakage_ratio:.0%} • alinhamento "
            f"{self.alignment_score:.2f}"
        )
        painter.drawText(
            padding,
            height - 20,
            self._elide(painter, flow, width - padding * 2),
        )

        painter.setPen(QColor("#d0d0d0"))
        painter.drawText(
            padding,
            height - 4,
            self._elide(painter, self.reason, width - padding * 2),
        )
        painter.end()
