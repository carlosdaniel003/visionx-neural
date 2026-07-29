import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt6.QtCore import Qt, QRectF, QPointF


class SilkDebuggerWidget(QWidget):
    """Explica a comparação estrutural sem trocar o enquadramento do epicentro."""

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
    def _copy(value):
        if isinstance(value, np.ndarray) and value.size > 0:
            return value.copy()
        return None

    @staticmethod
    def _first_array(detail: dict, *keys):
        for key in keys:
            value = detail.get(key)
            if isinstance(value, np.ndarray) and value.size > 0:
                return value
        return None

    @staticmethod
    def _mask_to_bgr(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None or mask.size == 0:
            return None
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _fit_mask(mask, shape):
        if not isinstance(mask, np.ndarray) or mask.size == 0:
            return None
        height, width = shape[:2]
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return mask.astype(np.uint8)

    @staticmethod
    def _paint(image, mask, color, alpha):
        if mask is None:
            return image
        output = image.astype(np.float32).copy()
        selected = mask > 0
        if np.any(selected):
            paint = np.asarray(color, dtype=np.float32)
            output[selected] = output[selected] * (1.0 - alpha) + paint * alpha
        return np.clip(output, 0, 255).astype(np.uint8)

    @classmethod
    def _reference_overlay(cls, canonical_reference, detail):
        if canonical_reference is None:
            return cls._copy(detail.get("reference_view"))
        output = canonical_reference.copy()
        expected = cls._fit_mask(detail.get("mask_gab"), output.shape)
        if expected is not None:
            expected = cv2.dilate(expected, np.ones((2, 2), np.uint8))
            output = cls._paint(output, expected, (0, 210, 255), 0.78)
        return output

    @classmethod
    def _raw_reconstruction(cls, canonical_test, detail):
        """Reconstrói extra/ausente usando sempre a ROI bruta como fundo."""
        if canonical_test is None:
            return cls._copy(detail.get("difference_view"))

        output = (canonical_test.astype(np.float32) * 0.72).astype(np.uint8)
        matched_source = cls._first_array(
            detail,
            "match_mask_raw_coordinates",
            "match_mask",
        )
        extra_source = cls._first_array(
            detail,
            "extra_mask_raw_coordinates",
            "extra_mask",
        )
        matched = cls._fit_mask(matched_source, output.shape)
        extra = cls._fit_mask(extra_source, output.shape)
        missing = cls._fit_mask(detail.get("missing_mask"), output.shape)

        if matched is not None:
            matched = cv2.dilate(
                matched,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            output = cls._paint(output, matched, (70, 190, 90), 0.38)
        if missing is not None:
            missing = cv2.dilate(
                missing,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            output = cls._paint(output, missing, (0, 220, 255), 0.92)
        if extra is not None:
            extra = cv2.dilate(
                extra,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            output = cls._paint(output, extra, (45, 65, 255), 0.94)

        for mask, color in (
            (extra, (30, 30, 255)),
            (missing, (0, 220, 255)),
        ):
            if mask is None:
                continue
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                if cv2.contourArea(contour) >= 3:
                    cv2.drawContours(
                        output,
                        [contour],
                        -1,
                        color,
                        1,
                        lineType=cv2.LINE_AA,
                    )
        return output

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

        # crop_gab/crop_test são exatamente as matrizes exibidas no Laboratório
        # de Textura. O widget as usa diretamente, sem novo recorte.
        canonical_reference = self._copy(detail.get("crop_gab"))
        canonical_test = self._copy(detail.get("crop_test"))
        if canonical_reference is None:
            canonical_reference = self._copy(detail.get("canonical_roi_reference"))
        if canonical_test is None:
            canonical_test = self._copy(detail.get("canonical_roi_test"))
        if canonical_reference is None:
            canonical_reference = self._copy(detail.get("roi_reference_raw"))
        if canonical_test is None:
            canonical_test = self._copy(detail.get("roi_test_raw"))

        self.reference_view = self._reference_overlay(canonical_reference, detail)
        self.test_view = canonical_test
        self.difference_view = self._raw_reconstruction(canonical_test, detail)

        if self.reference_view is None:
            self.reference_view = self._copy(detail.get("reference_view"))
        if self.test_view is None:
            self.test_view = self._copy(detail.get("test_view"))
        if self.difference_view is None:
            self.difference_view = self._copy(detail.get("difference_view"))

        if self.reference_view is None:
            self.reference_view = self._mask_to_bgr(detail.get("mask_gab"))
        if self.test_view is None:
            self.test_view = self._mask_to_bgr(detail.get("mask_test"))
        if self.difference_view is None:
            diff = detail.get("diff_mask")
            if isinstance(diff, np.ndarray) and diff.size > 0:
                fallback = np.zeros((*diff.shape, 3), dtype=np.uint8)
                fallback[diff > 0] = (45, 65, 255)
                self.difference_view = fallback

        source = canonical_test if canonical_test is not None else self.test_view
        if isinstance(source, np.ndarray) and source.size > 0:
            self.roi_height, self.roi_width = source.shape[:2]
        else:
            self.roi_width = int(detail.get("roi_width", 0) or 0)
            self.roi_height = int(detail.get("roi_height", 0) or 0)
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
        painter.drawText(7, 15, "Comparador Estrutural do Epicentro • XOR")

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
            "2. TESTE • MESMA ROI DO LABORATÓRIO" + roi_label,
            QColor("#46d9ff"),
        )
        self._draw_image(
            painter,
            self.difference_view,
            difference_rect,
            "3. RECONSTRUÇÃO SOBRE A MESMA ROI",
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
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_width, bar_height), 3, 3)

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
            f"Alinhamento usado somente no cálculo: X {self.dx:.1f}px • Y {self.dy:.1f}px • "
            f"confiança {self.alignment_score:.2f} • {self.reason}",
        )
        painter.end()
