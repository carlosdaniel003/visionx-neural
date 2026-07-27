"""Visualização da memória KNN e de seu peso na fusão final."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class KNNSpectrumWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)

        self.is_active = False
        self.has_memory = False
        self.vote = 0.5
        self.best_sim = 0.0
        self.n_neighbors = 0
        self.role = "SEM MEMÓRIA"
        self.memory_weight = 0.0
        self.physical_weight = 1.0
        self.fusion_rule = "physical_only"

    def update_data(self, detail: dict):
        if not detail or (
            "vote_defect" not in detail
            and "db_vote" not in detail
            and "decision_trace" not in detail
        ):
            self.is_active = False
            self.update()
            return

        self.is_active = True
        trace = detail.get("decision_trace", {})
        memory = trace.get("memory", {}) if isinstance(trace, dict) else {}
        weights = trace.get("weights", {}) if isinstance(trace, dict) else {}

        self.has_memory = bool(
            memory.get(
                "has_memory",
                detail.get("has_memory", detail.get("db_has_memory", False)),
            )
        )
        self.vote = float(
            memory.get(
                "vote_defect",
                detail.get("vote_defect", detail.get("db_vote", 0.5)),
            )
        )
        self.best_sim = float(
            memory.get(
                "best_similarity",
                detail.get("best_similarity", detail.get("db_best_sim", 0.0)),
            )
        )
        self.n_neighbors = int(
            memory.get(
                "n_neighbors",
                detail.get("n_neighbors", detail.get("db_neighbors", 0)),
            )
        )
        self.role = str(memory.get("role", "MEMÓRIA AUXILIAR"))
        self.memory_weight = float(weights.get("knn", 0.0))
        self.physical_weight = float(weights.get("physical", 1.0))
        self.fusion_rule = str(trace.get("fusion_rule", "physical_only"))
        self.update()

    @staticmethod
    def _elide(painter: QPainter, text: str, width: int) -> str:
        return painter.fontMetrics().elidedText(
            text,
            Qt.TextElideMode.ElideRight,
            max(20, width),
        )

    @staticmethod
    def _draw_progress(
        painter: QPainter,
        rect: QRectF,
        value: float,
        fill: QColor,
    ) -> None:
        value = max(0.0, min(1.0, float(value)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#303030"))
        painter.drawRoundedRect(rect, 4, 4)
        painter.setBrush(fill)
        painter.drawRoundedRect(
            QRectF(rect.x(), rect.y(), rect.width() * value, rect.height()),
            4,
            4,
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        painter.fillRect(0, 0, width, height, QColor("#101010"))

        if not self.is_active:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Motor KNN inativo")
            painter.end()
            return

        if not self.has_memory:
            painter.setPen(QColor("#555555"))
            painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Dataset sem vizinhos compatíveis",
            )
            painter.end()
            return

        padding = 12
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        role_color = (
            QColor("#ff6262")
            if self.fusion_rule in {"memory_veto", "memory_override"}
            else QColor("#f5c518")
        )
        painter.setPen(role_color)
        painter.drawText(
            padding,
            14,
            self._elide(
                painter,
                f"{self.role} • peso final KNN {self.memory_weight:.0%}",
                width - padding * 2,
            ),
        )

        bar_width = max(60, width - padding * 2)
        bar_height = 10

        vote_y = 31
        painter.setPen(QColor("#a6a6a6"))
        painter.drawText(padding, vote_y, "VOTO DE DEFEITO")
        vote_rect = QRectF(padding, vote_y + 6, bar_width, bar_height)
        gradient = QLinearGradient(
            vote_rect.x(),
            vote_rect.y(),
            vote_rect.right(),
            vote_rect.y(),
        )
        gradient.setColorAt(0.0, QColor("#4ade80"))
        gradient.setColorAt(0.5, QColor("#6b6b6b"))
        gradient.setColorAt(1.0, QColor("#ff6262"))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(vote_rect, 4, 4)
        marker_x = vote_rect.x() + vote_rect.width() * self.vote
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(
            QPointF(marker_x, vote_rect.y() - 2),
            QPointF(marker_x, vote_rect.bottom() + 2),
        )

        similarity_y = vote_y + 38
        painter.setPen(QColor("#a6a6a6"))
        painter.drawText(padding, similarity_y, "SIMILARIDADE DO MELHOR VIZINHO")
        similarity_rect = QRectF(
            padding,
            similarity_y + 6,
            bar_width,
            bar_height,
        )
        self._draw_progress(
            painter,
            similarity_rect,
            self.best_sim,
            QColor("#f5c518"),
        )

        painter.setPen(QColor("#d0d0d0"))
        info = (
            f"Voto {self.vote:.0%} NG • similaridade {self.best_sim:.0%} • "
            f"{self.n_neighbors} vizinho(s) • físico {self.physical_weight:.0%}"
        )
        painter.drawText(
            padding,
            height - 8,
            self._elide(painter, info, width - padding * 2),
        )
        painter.end()
