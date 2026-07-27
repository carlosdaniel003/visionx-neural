"""Visualização responsiva da influência dos motores na decisão final."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from src.ui.decision_model import influence_rows


class DecisionInfluenceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
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

    @staticmethod
    def _status_text(row: dict) -> str:
        if row["selected"]:
            return "DOMINANTE"
        if row.get("participates", False):
            return "PARTICIPA"
        if not row["active"]:
            return "INATIVO"
        if row["triggered"]:
            return "EVIDÊNCIA"
        return "ABAIXO"

    @staticmethod
    def _row_value_text(row: dict, score: float, threshold: float) -> str:
        weight = float(row.get("fusion_weight", 0.0))
        contribution = float(row.get("score_contribution", 0.0))
        status = DecisionInfluenceWidget._status_text(row)

        if row.get("id") == "knn":
            effect = float(row.get("effect_vs_physical", 0.0))
            effect_text = f"{effect * 100:+.0f} pp"
            return (
                f"voto {score:.0%} NG • peso {weight:.0%} • "
                f"efeito {effect_text}"
            )

        if weight > 0.0:
            return (
                f"{score:.0%}/{threshold:.0%} • peso {weight:.0%} • "
                f"parcela {contribution:.0%}"
            )

        return f"{score:.0%}/{threshold:.0%} • {status} • peso direto 0%"

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
        footer_height = 48
        row_area = max(105, height - footer_height - top)
        row_height = row_area / max(len(self.rows), 1)

        label_width = min(160, max(92, int(width * 0.25)))
        value_width = min(245, max(145, int(width * 0.34)))
        bar_x = padding + label_width
        bar_width = max(42, width - bar_x - value_width - padding)

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))

        for index, row in enumerate(self.rows):
            y = top + index * row_height
            center_y = y + row_height * 0.46
            color = self._status_color(row)

            painter.setPen(color)
            label = self._elide(painter, row["label"], label_width - 8)
            painter.drawText(
                QRectF(padding, y, label_width - 5, row_height),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

            # Barra principal: evidência do motor ou voto NG do KNN.
            bar_h = min(11.0, max(7.0, row_height * 0.27))
            bar_y = center_y - bar_h / 2
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#2d2d2d"))
            painter.drawRoundedRect(
                QRectF(bar_x, bar_y, bar_width, bar_h),
                3,
                3,
            )

            score = max(0.0, min(1.0, row["raw_score"]))
            if score > 0.0:
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

            # Barra fina amarela: peso efetivo usado na fórmula de fusão.
            weight = max(0.0, min(1.0, float(row.get("fusion_weight", 0.0))))
            weight_y = bar_y + bar_h + 3
            weight_h = 3.0
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#232323"))
            painter.drawRoundedRect(
                QRectF(bar_x, weight_y, bar_width, weight_h),
                1.5,
                1.5,
            )
            if weight > 0.0:
                painter.setBrush(QColor("#f5c518"))
                painter.drawRoundedRect(
                    QRectF(bar_x, weight_y, bar_width * weight, weight_h),
                    1.5,
                    1.5,
                )

            value_text = self._row_value_text(row, score, threshold)
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
        physical_weight = float(weights.get("physical", 1.0))
        knn_weight = float(weights.get("knn", 0.0))
        memory = self.trace.get("memory", {})
        knn_vote = float(memory.get("vote_defect", 0.5))

        footer_y = height - footer_height + 2
        painter.setPen(QColor("#d0d0d0"))
        formula = (
            f"Fusão: físico {physical:.0%}×{physical_weight:.0%} + "
            f"KNN {knn_vote:.0%}×{knn_weight:.0%} = {final_score:.0%}"
        )
        painter.drawText(
            padding,
            int(footer_y + 14),
            self._elide(painter, formula, width - padding * 2),
        )

        painter.setPen(QColor("#f5c518"))
        footer_2 = (
            f"Corte {cutoff:.0%} • regra {self.trace.get('fusion_rule', 'physical_only')} • "
            "barra maior = evidência; barra amarela fina = peso"
        )
        painter.drawText(
            padding,
            int(footer_y + 32),
            self._elide(painter, footer_2, width - padding * 2),
        )
        painter.end()
