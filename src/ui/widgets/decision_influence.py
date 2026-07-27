"""Visualização responsiva da influência dos motores na decisão final."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from src.ui.decision_model import influence_rows


class DecisionInfluenceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(210)
        self.setMinimumWidth(320)
        self.trace = {}
        self.rows = []

    def update_data(self, analysis: dict | None):
        detail = (analysis or {}).get("detail", {})
        trace = detail.get("decision_trace", {})
        self.trace = trace if isinstance(trace, dict) else {}
        self.rows = influence_rows(self.trace)
        self.update()

    @staticmethod
    def _status_color(row: dict) -> QColor:
        if not row["active"]:
            return QColor("#555555")
        if row["selected"]:
            return QColor("#f5c518")
        if row["triggered"]:
            return QColor("#ff6262")
        return QColor("#4ade80")

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

        if not self.rows:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Rastreamento da decisão indisponível",
            )
            painter.end()
            return

        padding = 8
        top = 5
        footer_height = 42
        row_area = max(100, height - footer_height - top)
        row_height = row_area / max(len(self.rows), 1)

        label_width = min(165, max(95, int(width * 0.28)))
        value_width = min(160, max(112, int(width * 0.24)))
        bar_x = padding + label_width
        bar_width = max(45, width - bar_x - value_width - padding)

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))

        for index, row in enumerate(self.rows):
            y = top + index * row_height
            center_y = y + row_height * 0.50
            color = self._status_color(row)

            painter.setPen(color)
            label = self._elide(painter, row["label"], label_width - 8)
            painter.drawText(
                QRectF(padding, y, label_width - 5, row_height),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

            bar_h = min(12.0, max(7.0, row_height * 0.30))
            bar_y = center_y - bar_h / 2
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#2d2d2d"))
            painter.drawRoundedRect(
                QRectF(bar_x, bar_y, bar_width, bar_h),
                3,
                3,
            )

            score = max(0.0, min(1.0, row["raw_score"]))
            painter.setBrush(color)
            painter.drawRoundedRect(
                QRectF(bar_x, bar_y, bar_width * score, bar_h),
                3,
                3,
            )

            threshold = max(0.0, min(1.0, row["threshold"]))
            threshold_x = bar_x + bar_width * threshold
            painter.setPen(QPen(QColor("#f5f5f5"), 1))
            painter.drawLine(
                int(threshold_x),
                int(bar_y - 2),
                int(threshold_x),
                int(bar_y + bar_h + 2),
            )

            if row["selected"]:
                status = "DOMINANTE"
            elif not row["active"]:
                status = "INATIVO"
            elif row["triggered"]:
                status = "ACIMA"
            else:
                status = "ABAIXO"

            value_text = (
                f"{score:.0%}/{threshold:.0%} • {status} • "
                f"impacto {row['final_influence']:.0%}"
            )
            painter.setPen(color)
            painter.drawText(
                QRectF(
                    bar_x + bar_width + 7,
                    y,
                    value_width - 7,
                    row_height,
                ),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._elide(painter, value_text, value_width - 10),
            )

        cutoff = float(self.trace.get("cutoff", 0.45))
        physical = float(self.trace.get("physical_score", 0.0))
        final_score = float(self.trace.get("final_score", 0.0))
        weights = self.trace.get("weights", {})

        footer_y = height - footer_height + 3
        painter.setPen(QColor("#a6a6a6"))
        footer_1 = (
            f"Físico {physical:.0%} • Final {final_score:.0%} • "
            f"Corte {cutoff:.0%}"
        )
        painter.drawText(
            padding,
            int(footer_y + 13),
            self._elide(painter, footer_1, width - padding * 2),
        )

        painter.setPen(QColor("#f5c518"))
        footer_2 = (
            f"Regra {self.trace.get('fusion_rule', 'physical_only')} • "
            f"peso físico {float(weights.get('physical', 1.0)):.0%} • "
            f"peso KNN {float(weights.get('knn', 0.0)):.0%}"
        )
        painter.drawText(
            padding,
            int(footer_y + 30),
            self._elide(painter, footer_2, width - padding * 2),
        )
        painter.end()
