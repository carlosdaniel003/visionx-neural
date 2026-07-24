import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QImage
from PyQt6.QtCore import Qt, QRectF


class SSIMDebuggerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMinimumWidth(150)

        self.is_active = False
        self.local_score = 0.0
        self.ctx_score = 0.0
        self.ctx_reason = ""
        self.heat_map_raw = None
        self.crop_gab = None
        self.crop_test = None
        self.focus_source = ""
        self.focus_box = None
        self.focus_width = 0
        self.focus_height = 0
        self.heat_map_mode = "generic_ssim"
        self.adhesive_coverage = 0.0
        self.adhesive_evidence = 0.0
        self.adhesive_centroid = None
        self.alignment_shift = (0.0, 0.0)
        self.alignment_score = 0.0

    def update_data(self, detail: dict):
        """Recebe os detalhes da análise do SSIMExpert."""
        if not detail or "heat_map_raw" not in detail:
            self.is_active = False
            self.update()
            return

        self.is_active = True
        self.local_score = detail.get("local_score", 0.0)
        self.ctx_score = detail.get("ctx_score", 0.0)
        self.ctx_reason = detail.get("ctx_reason", "")
        self.heat_map_raw = detail.get("heat_map_raw")
        self.crop_gab = detail.get("crop_gab")
        self.crop_test = detail.get("crop_test")
        self.focus_source = detail.get("focus_source", "")
        self.focus_box = detail.get("focus_box")
        self.focus_width = int(detail.get("focus_width", 0) or 0)
        self.focus_height = int(detail.get("focus_height", 0) or 0)
        self.heat_map_mode = detail.get("heat_map_mode", "generic_ssim")
        self.adhesive_coverage = float(detail.get("adhesive_coverage", 0.0) or 0.0)
        self.adhesive_evidence = float(detail.get("adhesive_evidence", 0.0) or 0.0)
        self.adhesive_centroid = detail.get("adhesive_centroid")
        self.alignment_shift = detail.get("alignment_shift", (0.0, 0.0))
        self.alignment_score = float(detail.get("alignment_score", 0.0) or 0.0)
        self.update()

    @staticmethod
    def _qimage_from_bgr(img_bgr: np.ndarray) -> QImage:
        if not img_bgr.flags["C_CONTIGUOUS"]:
            img_bgr = np.ascontiguousarray(img_bgr)
        rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb_img.shape[:2]
        return QImage(
            rgb_img.data,
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        ).copy()

    def _draw_image_box(
        self,
        painter: QPainter,
        img_bgr: np.ndarray,
        rect: QRectF,
        title: str,
    ) -> None:
        painter.setPen(QPen(QColor("#3a3a3a"), 1))
        painter.setBrush(QColor("#070707"))
        painter.drawRect(rect)

        painter.setPen(QColor("#b7b7b7"))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(
            rect.adjusted(0, -12, 0, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            title,
        )

        if img_bgr is None or img_bgr.size == 0:
            painter.setPen(QColor("#555555"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "SEM FOTO")
            return

        qimg = self._qimage_from_bgr(img_bgr)
        scaled_img = qimg.scaled(
            max(1, int(rect.width() - 4)),
            max(1, int(rect.height() - 4)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        image_x = rect.x() + (rect.width() - scaled_img.width()) / 2
        image_y = rect.y() + (rect.height() - scaled_img.height()) / 2
        painter.drawImage(int(image_x), int(image_y), scaled_img)

    def _draw_overlay_heatmap(
        self,
        painter: QPainter,
        heat_arr: np.ndarray,
        bg_img: np.ndarray,
        rect: QRectF,
        title: str,
    ) -> None:
        painter.setPen(QPen(QColor("#3a3a3a"), 1))
        painter.setBrush(QColor("#070707"))
        painter.drawRect(rect)

        painter.setPen(QColor("#f5c518"))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(
            rect.adjusted(0, -12, 0, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
            title,
        )

        if (
            heat_arr is None
            or heat_arr.size == 0
            or bg_img is None
            or bg_img.size == 0
        ):
            painter.setPen(QColor("#555555"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "FALHA NA FUSÃO")
            return

        bg_height, bg_width = bg_img.shape[:2]
        if heat_arr.shape[:2] != (bg_height, bg_width):
            heat_arr = cv2.resize(
                heat_arr,
                (bg_width, bg_height),
                interpolation=cv2.INTER_LINEAR,
            )

        heat_uint8 = np.clip(heat_arr, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)

        # A foto real permanece inalterada onde não existe evidência. O mapa
        # colore somente o adesivo candidato em vez de tingir toda a ROI.
        heat_strength = heat_uint8.astype(np.float32) / 255.0
        alpha = np.power(heat_strength, 0.85) * 0.88
        alpha[heat_uint8 < 18] = 0.0
        alpha_3 = alpha[:, :, None]
        blended = (
            bg_img.astype(np.float32) * (1.0 - alpha_3)
            + heatmap_bgr.astype(np.float32) * alpha_3
        )
        blended_bgr = np.clip(blended, 0, 255).astype(np.uint8)

        if self.heat_map_mode == "adhesive_excess":
            strong_mask = (heat_uint8 >= 96).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                strong_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                if cv2.contourArea(contour) >= 3:
                    cv2.drawContours(
                        blended_bgr,
                        [contour],
                        -1,
                        (0, 220, 255),
                        1,
                        lineType=cv2.LINE_AA,
                    )

            if self.adhesive_centroid and len(self.adhesive_centroid) == 2:
                center_x = int(round(self.adhesive_centroid[0]))
                center_y = int(round(self.adhesive_centroid[1]))
                if 0 <= center_x < bg_width and 0 <= center_y < bg_height:
                    cv2.drawMarker(
                        blended_bgr,
                        (center_x, center_y),
                        (0, 255, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=max(7, min(bg_width, bg_height) // 8),
                        thickness=1,
                        line_type=cv2.LINE_AA,
                    )

        qimg = self._qimage_from_bgr(blended_bgr)
        scaled_img = qimg.scaled(
            max(1, int(rect.width() - 4)),
            max(1, int(rect.height() - 4)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        image_x = rect.x() + (rect.width() - scaled_img.width()) / 2
        image_y = rect.y() + (rect.height() - scaled_img.height()) / 2
        painter.drawImage(int(image_x), int(image_y), scaled_img)

    def _focus_description(self) -> str:
        source_names = {
            "epicenter_extractor": "Epicentro exibido no painel",
            "aoi_intersection": "Interseção da AOI",
            "raw_anomaly": "Fallback de anomalia",
        }
        source = source_names.get(
            self.focus_source,
            self.focus_source or "Fonte desconhecida",
        )
        dimensions = ""
        if self.focus_width > 0 and self.focus_height > 0:
            dimensions = f" • ROI {self.focus_width}×{self.focus_height}px"
        coordinates = ""
        if self.focus_box and len(self.focus_box) == 4:
            x, y, width, height = self.focus_box
            coordinates = f" • X:{x} Y:{y} W:{width} H:{height}"
        return source + dimensions + coordinates

    def _map_description(self) -> str:
        if self.heat_map_mode != "adhesive_excess":
            return "Mapa: diferença estrutural genérica"

        shift_x, shift_y = self.alignment_shift
        return (
            "Mapa: adesivo excedente"
            f" • Cobertura {self.adhesive_coverage:.1%}"
            f" • Evidência {self.adhesive_evidence:.2f}"
            f" • Alinhamento X:{shift_x:.1f} Y:{shift_y:.1f}"
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        painter.fillRect(0, 0, width, height, QColor("#101010"))

        painter.setPen(QColor("#f5f5f5"))
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.drawText(7, 15, "Laboratório de Textura • SSIM")

        if not self.is_active:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Motor SSIM Inativo",
            )
            painter.end()
            return

        padding = 10
        spacing = 7
        available_width = width - padding * 2 - spacing * 2
        box_width = available_width / 3.0
        y_start = 40
        y_end = height - 76
        box_height = max(20.0, y_end - y_start)

        rect_gab = QRectF(padding, y_start, box_width, box_height)
        rect_test = QRectF(
            padding + box_width + spacing,
            y_start,
            box_width,
            box_height,
        )
        rect_diff = QRectF(
            padding + box_width * 2 + spacing * 2,
            y_start,
            box_width,
            box_height,
        )

        size_label = ""
        if self.focus_width and self.focus_height:
            size_label = f" • {self.focus_width}×{self.focus_height}px"
        self._draw_image_box(
            painter,
            self.crop_gab,
            rect_gab,
            "1. GABARITO • ROI EXATA" + size_label,
        )
        self._draw_image_box(
            painter,
            self.crop_test,
            rect_test,
            "2. TESTE • MESMA ROI" + size_label,
        )

        heat_title = (
            "3. ADESIVO EXCEDENTE • MAPA SELETIVO"
            if self.heat_map_mode == "adhesive_excess"
            else "3. DIFERENÇA SSIM • MESMA ROI"
        )
        self._draw_overlay_heatmap(
            painter,
            self.heat_map_raw,
            self.crop_test,
            rect_diff,
            heat_title,
        )

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#f5c518"))
        painter.drawText(padding, height - 52, self._focus_description())
        painter.drawText(padding, height - 34, self._map_description())

        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#aaaaaa"))
        painter.drawText(padding, height - 12, f"Contexto IA: {self.ctx_reason}")

        is_critical = self.local_score > 0.45
        status_color = QColor("#ff6262") if is_critical else QColor("#4ade80")
        status_text = f"Dano físico: {self.local_score:.0%}"
        text_width = painter.fontMetrics().horizontalAdvance(status_text)
        painter.setPen(status_color)
        painter.drawText(
            int(width - padding - text_width),
            height - 12,
            status_text,
        )
        painter.end()
