"""Debugger visual da expectativa do patch para a categoria FALTANDO."""

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
        self.tolerance = 0.36
        self.patch_type = "unknown"
        self.classification = "SEM DADOS"
        self.coverage = 0.0
        self.residual_mean = 0.0
        self.residual_p90 = 0.0
        self.residual_peak = 0.0
        self.structure_loss = 0.0
        self.extra_structure = 0.0
        self.edge_mismatch = 0.0
        self.direct_similarity = 1.0
        self.best_similarity = 0.0
        self.displacement_dx = 0.0
        self.displacement_dy = 0.0
        self.displacement_pixels = 0.0
        self.background_signal = 0.0
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

    @staticmethod
    def _fit_gray(value, shape):
        if not isinstance(value, np.ndarray) or value.size == 0:
            return None
        height, width = shape[:2]
        image = value.copy()
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape[:2] != (height, width):
            image = cv2.resize(
                image,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
        return image

    @classmethod
    def _rebuild_on_canonical_test(cls, canonical_test, detail):
        """Aplica o mapa do motor sobre a mesma ROI exibida no Laboratório."""
        if canonical_test is None:
            return cls._copy(detail.get("missing_reconstruction_view"))

        residual = cls._fit_gray(detail.get("missing_residual_map"), canonical_test.shape)
        anomaly_mask = cls._fit_gray(detail.get("roi_anomaly_mask"), canonical_test.shape)
        if anomaly_mask is None:
            anomaly_mask = cls._fit_gray(
                detail.get("component_missing_mask"),
                canonical_test.shape,
            )

        reconstruction = (canonical_test.astype(np.float32) * 0.78).astype(np.uint8)
        if residual is None:
            residual = np.zeros(canonical_test.shape[:2], dtype=np.float32)
        else:
            residual = residual.astype(np.float32)
            if float(np.max(residual)) > 1.0:
                residual /= 255.0
            residual = np.clip(residual, 0.0, 1.0)

        if anomaly_mask is None:
            anomaly_mask = (residual >= 0.31).astype(np.uint8) * 255
        else:
            anomaly_mask = (anomaly_mask > 0).astype(np.uint8) * 255

        heat_source = np.clip(residual * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heat_source, cv2.COLORMAP_TURBO)
        selected = anomaly_mask > 0
        if np.any(selected):
            reconstruction[selected] = (
                reconstruction[selected].astype(np.float32) * 0.20
                + heatmap[selected].astype(np.float32) * 0.80
            ).astype(np.uint8)

        contours, _ = cv2.findContours(
            anomaly_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            reconstruction,
            contours,
            -1,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        if np.any(selected):
            weighted = residual * selected.astype(np.float32)
            _, _, _, maximum_location = cv2.minMaxLoc(weighted.astype(np.float32))
            cv2.drawMarker(
                reconstruction,
                maximum_location,
                (255, 230, 40),
                cv2.MARKER_CROSS,
                10,
                1,
                cv2.LINE_AA,
            )
        return reconstruction

    def update_data(self, detail: dict):
        valid_modes = {
            "missing_component",
            "roi_expectation",
            "roi_patch_expectation",
        }
        if (
            not detail
            or detail.get("missing_comparison_mode") not in valid_modes
            or not detail.get("missing_active", False)
        ):
            self.is_active = False
            self.update()
            return

        self.is_active = True
        self.is_defect = bool(detail.get("missing_is_defect", False))
        self.score = float(detail.get("missing_score", 0.0))
        self.tolerance = float(detail.get("missing_tolerance", 0.36))
        self.patch_type = str(detail.get("missing_patch_type", "unknown"))
        self.classification = str(
            detail.get("missing_classification", "SEM CLASSIFICAÇÃO")
        )
        self.coverage = float(
            detail.get(
                "missing_changed_coverage",
                detail.get("missing_coverage", 0.0),
            )
        )
        self.residual_mean = float(detail.get("missing_residual_mean", 0.0))
        self.residual_p90 = float(detail.get("missing_residual_p90", 0.0))
        self.residual_peak = float(detail.get("missing_residual_peak", 0.0))
        self.structure_loss = float(detail.get("missing_structure_loss", 0.0))
        self.extra_structure = float(detail.get("missing_extra_structure", 0.0))
        self.edge_mismatch = float(
            detail.get(
                "missing_edge_mismatch",
                max(self.structure_loss, self.extra_structure),
            )
        )
        self.direct_similarity = float(detail.get("missing_direct_similarity", 1.0))
        self.best_similarity = float(detail.get("missing_best_similarity", 0.0))
        self.displacement_dx = float(detail.get("missing_displacement_dx", 0.0))
        self.displacement_dy = float(detail.get("missing_displacement_dy", 0.0))
        self.displacement_pixels = float(
            detail.get("missing_displacement_pixels", 0.0)
        )
        self.background_signal = float(
            detail.get("missing_background_exposure", 0.0)
        )
        self.reason = str(detail.get("missing_reason", ""))

        # Mesma fonte do Laboratório de Textura: nenhuma nova extração da ROI.
        canonical_reference = self._copy(detail.get("crop_gab"))
        canonical_test = self._copy(detail.get("crop_test"))
        if canonical_reference is None:
            canonical_reference = self._copy(detail.get("canonical_roi_reference"))
        if canonical_test is None:
            canonical_test = self._copy(detail.get("canonical_roi_test"))
        if canonical_reference is None:
            canonical_reference = self._copy(detail.get("missing_reference_view"))
        if canonical_test is None:
            canonical_test = self._copy(detail.get("missing_test_input_raw"))
        if canonical_test is None:
            canonical_test = self._copy(detail.get("missing_test_view"))

        self.reference_view = canonical_reference
        self.test_view = canonical_test
        self.reconstruction_view = self._rebuild_on_canonical_test(
            canonical_test,
            detail,
        )

        source = canonical_test if canonical_test is not None else self.test_view
        if isinstance(source, np.ndarray) and source.size > 0:
            self.roi_height, self.roi_width = source.shape[:2]
        else:
            self.roi_width = int(detail.get("missing_roi_width", 0) or 0)
            self.roi_height = int(detail.get("missing_roi_height", 0) or 0)
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

    def _patch_label(self) -> str:
        if self.patch_type == "homogeneous":
            return "PATCH HOMOGÊNEO"
        if self.patch_type == "structured":
            return "PATCH ESTRUTURADO"
        return "PATCH VISUAL"

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = self.width()
        height = self.height()
        painter.fillRect(0, 0, width, height, QColor("#101010"))

        painter.setPen(QColor("#f5f5f5"))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.drawText(8, 18, "EXPECTATIVA DO PATCH • MOTOR FALTANDO")

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
        self._draw_image(
            painter,
            self.reference_view,
            rects[0],
            "1. GABARITO • MESMA ROI DO LABORATÓRIO" + roi_label,
            QColor("#4ade80"),
        )
        self._draw_image(
            painter,
            self.test_view,
            rects[1],
            "2. TESTE • MESMA ROI DO LABORATÓRIO" + roi_label,
            QColor("#46d9ff"),
        )
        self._draw_image(
            painter,
            self.reconstruction_view,
            rects[2],
            "3. DIVERGÊNCIA SOBRE A MESMA ROI",
            QColor("#f5c518"),
        )

        status_color = QColor("#ff6262") if self.is_defect else QColor("#4ade80")
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(status_color)
        painter.drawText(
            padding,
            height - 94,
            self._elide(
                painter,
                f"Expectativa: {self._patch_label()} • Resultado: {self.classification}",
                width - padding * 2,
            ),
        )

        gauge_y = height - 80
        gauge_x = width * 0.10
        gauge_width = width * 0.80
        gauge_height = 8
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#303030"))
        painter.drawRoundedRect(QRectF(gauge_x, gauge_y, gauge_width, gauge_height), 4, 4)
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
                "Legenda: imagem normal = compatível • mapa colorido = divergência • amarelo = contorno • cruz = pico",
                width - padding * 2,
            ),
        )
        painter.setPen(status_color)
        metrics = (
            f"Score {self.score:.0%}/{self.tolerance:.0%} • área divergente "
            f"{self.coverage:.0%} • intensidade média {self.residual_mean:.0%} • "
            f"P90 {self.residual_p90:.0%} • bordas incompatíveis {self.edge_mismatch:.0%}"
        )
        painter.drawText(
            padding,
            height - 41,
            self._elide(painter, metrics, width - padding * 2),
        )
        painter.setPen(QColor("#f5c518"))
        details = (
            f"Similaridade na posição {self.direct_similarity:.0%} • melhor próxima "
            f"{self.best_similarity:.0%} • possível deslocamento X:{self.displacement_dx:+.1f}px "
            f"Y:{self.displacement_dy:+.1f}px ({self.displacement_pixels:.1f}px) • "
            f"sinal de fundo {self.background_signal:.0%}"
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
