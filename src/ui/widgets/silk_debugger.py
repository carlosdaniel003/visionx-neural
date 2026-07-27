import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt6.QtCore import Qt, QRectF, QPointF


class SilkDebuggerWidget(QWidget):
    """Explica a comparação estrutural da mesma ROI por alinhamento e XOR."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMinimumWidth(150)

        self.is_active = False
        self.silk_error_pct = 0.0
        self.tolerance = 0.08
        self.extra_pct = 0.0
        self.missing_pct = 0.0
        self.matched_pct = 1.0
        self.is_defect = False
        self.reason = ""
        self.dx = 0.0
        self.dy = 0.0
        self.alignment_score = 0.0
        self.roi_width = 0
        self.roi_height = 0
        self.reference_view = None
        self.test_view = None
        self.difference_view = None

    @staticmethod
    def _mask_to_bgr(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None or mask.size == 0:
            return None
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def update_data(self, detail: dict):
        if not detail or "silk_error_pct" not in detail:
            self.is_active = False
            self.update()
            return

        self.is_active = True
        self.silk_error_pct = float(detail.get("silk_error_pct", 0.0))
        self.tolerance = float(detail.get("tolerance", 0.08))
        self.extra_pct = float(detail.get("extra_pct", 0.0))
        self.missing_pct = float(detail.get("missing_pct", 0.0))
        self.matched_pct = float(detail.get("matched_pct", 1.0))
        self.is_defect = bool(detail.get("is_defect", False))
        self.reason = str(detail.get("reason", ""))
        self.dx = float(detail.get("dx", 0.0))
        self.dy = float(detail.get("dy", 0.0))
        self.alignment_score = float(detail.get("alignment_score", 0.0))
        self.roi_width = int(detail.get("roi_width", 0) or 0)
        self.roi_height = int(detail.get("roi_height", 0) or 0)

        self.reference_view = detail.get("reference_view")
        self.test_view = detail.get("test_view")
        self.difference_view = detail.get("difference_view")

        if self.reference_view is None:
            self.reference_view = self._mask_to_bgr(detail.get("mask_gab"))
        if self.test_view is None:
            self.test_view = self._mask_to_bgr(detail.get("mask_test"))
        if self.difference_view is None:
            diff = detail.get("diff_mask")
            if diff is not None and diff.size > 0:
                fallback = np.zeros((*diff.shape, 3), dtype=np.uint8)
                fallback[diff > 0] = (45, 65, 255)
                self.difference_view = fallback

        self.update()

    @staticmethod
    def _qimage_from_bgr(image_bgr: np.ndarray) -> QImage:
        if not image_bgr.flags["C_CONTIGUOUS"]:
            image_bgr = np.ascontiguousarray(image_bgr)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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
        painter.setPen(QPen(QColor("#353535"), 1))
        painter.setBrush(QColor("#070707"))
        painter.drawRect(rect)

        painter.setPen(title_color)
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(
            rect.adjusted(0, -12, 0, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        painter.fillRect(0, 0, width, height, QColor("#101010"))

        painter.setPen(QColor("#f5f5f5"))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(
            7,
            15,
            "Comparador Estrutural do Epicentro • XOR",
        )

        if not self.is_active:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Comparador estrutural inativo",
            )
            painter.end()
            return

        padding = 10
        spacing = 7
        available_width = width - padding * 2 - spacing * 2
        box_width = available_width / 3.0
        y_start = 40
        y_end = height - 82
        box_height = max(24.0, y_end - y_start)

        reference_rect = QRectF(padding, y_start, box_width, box_height)
        test_rect = QRectF(
            padding + box_width + spacing,
            y_start,
            box_width,
            box_height,
        )
        difference_rect = QRectF(
            padding + box_width * 2 + spacing * 2,
            y_start,
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
            "1. GABARITO • CONTORNOS ESPERADOS" + roi_label,
            QColor("#f5c518"),
        )
        self._draw_image(
            painter,
            self.test_view,
            test_rect,
            "2. TESTE ALINHADO • CONTORNOS ENCONTRADOS" + roi_label,
            QColor("#46d9ff"),
        )
        self._draw_image(
            painter,
            self.difference_view,
            difference_rect,
            "3. RECONSTRUÇÃO • EXTRA / AUSENTE",
            QColor("#ff7878"),
        )

        gauge_max = max(0.25, self.tolerance * 2.5, self.silk_error_pct * 1.10)
        gauge_max = min(1.0, gauge_max)
        bar_x = width * 0.10
        bar_width = width * 0.80
        bar_y = height - 66
        bar_height = 7

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#303030"))
        painter.drawRoundedRect(
            QRectF(bar_x, bar_y, bar_width, bar_height),
            3,
            3,
        )

        fill_ratio = min(1.0, self.silk_error_pct / max(gauge_max, 1e-6))
        status_color = QColor("#ff6262") if self.is_defect else QColor("#4ade80")
        painter.setBrush(status_color)
        painter.drawRoundedRect(
            QRectF(bar_x, bar_y, bar_width * fill_ratio, bar_height),
            3,
            3,
        )

        tolerance_x = bar_x + bar_width * min(
            1.0,
            self.tolerance / max(gauge_max, 1e-6),
        )
        painter.setPen(QPen(QColor("#f5c518"), 2))
        painter.drawLine(
            QPointF(tolerance_x, bar_y - 3),
            QPointF(tolerance_x, bar_y + bar_height + 3),
        )

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#a6a6a6"))
        painter.drawText(
            padding,
            height - 45,
            "Legenda: VERDE = coincide • VERMELHO = estrutura extra • "
            "AMARELO = estrutura ausente",
        )

        painter.setPen(status_color)
        painter.drawText(
            padding,
            height - 27,
            f"Divergência {self.silk_error_pct:.1%} • "
            f"Extra {self.extra_pct:.1%} • "
            f"Ausente {self.missing_pct:.1%} • "
            f"Coincidência {self.matched_pct:.1%} • "
            f"Tolerância {self.tolerance:.1%}",
        )

        painter.setPen(QColor("#f5c518"))
        painter.drawText(
            padding,
            height - 9,
            f"Alinhamento do teste: X {self.dx:.1f}px • Y {self.dy:.1f}px • "
            f"confiança {self.alignment_score:.2f} • {self.reason}",
        )
        painter.end()
