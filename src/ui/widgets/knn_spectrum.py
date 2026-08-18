"""Painel responsivo da memória visual do VisionX."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF, QSize
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from src.ui.memory_status_model import memory_status_from_detail


class KNNSpectrumWidget(QWidget):
    """Expõe dual-scale, contraste OK/NG e compactação por protótipos."""

    WIDE_BREAKPOINT = 520

    def __init__(self, parent=None):
        super().__init__(parent)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMinimumWidth(300)
        self.model = memory_status_from_detail({})

        # Mantidos para compatibilidade com extensões visuais já instaladas.
        self.is_active = False
        self.has_memory = False
        self.memory_available = False
        self.memory_conflict = False
        self.best_label = "-"
        self.best_sim = 0.0
        self.vote = 0.5
        self.n_neighbors = 0
        self.role = "SEM MEMÓRIA"
        self.memory_mode = "anomaly"
        self.memory_scope = "none"
        self.quantity_influence = False

    def sizeHint(self) -> QSize:
        return QSize(620, 245)

    def minimumSizeHint(self) -> QSize:
        return QSize(300, 225)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return 245 if width >= self.WIDE_BREAKPOINT else 338

    def update_data(self, detail: dict):
        self.model = memory_status_from_detail(detail)
        self.is_active = bool(self.model["active"])
        self.has_memory = bool(self.model["has_memory"])
        self.memory_available = bool(self.model["memory_available"])
        self.memory_conflict = bool(self.model["conflict"])
        self.best_label = self.model["leading_hypothesis"] or "-"
        self.best_sim = float(self.model["combined_similarity"])
        self.vote = float(self.model["memory_score"])
        self.n_neighbors = int(self.model["n_neighbors"])
        self.role = str(self.model["role"])
        self.memory_scope = str(self.model["scope"]).lower()
        self.quantity_influence = bool(self.model["quantity_influence"])
        self.updateGeometry()
        self.update()

    def _mode_label(self) -> str:
        return "ANOMALIA"

    @staticmethod
    def _pct(value, decimals: int = 1) -> str:
        if value is None:
            return "--"
        return f"{float(value) * 100:.{decimals}f}%"

    @staticmethod
    def _elide(painter: QPainter, text: str, width: float) -> str:
        return painter.fontMetrics().elidedText(
            str(text),
            Qt.TextElideMode.ElideRight,
            max(20, int(width)),
        )

    @staticmethod
    def _section(painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(QColor("#2d333b"), 1))
        painter.setBrush(QColor("#15191e"))
        painter.drawRoundedRect(rect, 7, 7)

    @staticmethod
    def _bar(
        painter: QPainter,
        rect: QRectF,
        value: float | None,
        color: QColor,
    ) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2a3037"))
        painter.drawRoundedRect(rect, 3, 3)
        if value is None:
            return
        bounded = max(0.0, min(1.0, float(value)))
        if bounded <= 0:
            return
        painter.setBrush(color)
        painter.drawRoundedRect(
            QRectF(rect.x(), rect.y(), rect.width() * bounded, rect.height()),
            3,
            3,
        )

    def _draw_header(self, painter: QPainter, rect: QRectF) -> None:
        model = self.model
        if model["conflict"]:
            color = QColor("#ffb454")
            title = "CONFLITO DE MEMÓRIA • REVISÃO HUMANA OBRIGATÓRIA"
        elif model["has_memory"]:
            label = model["leading_hypothesis"] or "-"
            color = QColor("#ff6262") if label == "NG" else QColor("#4ade80")
            title = f"HIPÓTESE {label} • MEMÓRIA VISUAL"
        elif model["memory_available"]:
            color = QColor("#f5c518")
            title = "MEMÓRIA DISPONÍVEL • CORRESPONDÊNCIA INSUFICIENTE"
        else:
            color = QColor("#6e7681")
            title = "SEM MEMÓRIA COMPATÍVEL"

        painter.setPen(QPen(color, 1))
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 24))
        painter.drawRoundedRect(rect, 7, 7)
        painter.setPen(color)
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        category = model["category"]
        suffix = f" • {category}" if category else ""
        painter.drawText(
            rect.adjusted(10, 0, -10, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self._elide(painter, title + suffix, rect.width() - 20),
        )

    def _draw_scales(self, painter: QPainter, rect: QRectF) -> None:
        model = self.model
        self._section(painter, rect)
        x = rect.x() + 10
        width = rect.width() - 20

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#a6a6a6"))
        painter.drawText(x, rect.y() + 16, "ESCALAS DA CORRESPONDÊNCIA LÍDER")

        rows = [
            ("Epicentro", model["epicenter_similarity"], QColor("#58a6ff")),
            ("Contexto", model["context_similarity"], QColor("#bc8cff")),
            ("Combinado", model["combined_similarity"], QColor("#f5c518")),
        ]
        y = rect.y() + 31
        for label, value, color in rows:
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QColor("#c9d1d9"))
            painter.drawText(x, y + 8, label)
            painter.setPen(color if value is not None else QColor("#6e7681"))
            painter.drawText(
                QRectF(x, y, width, 11),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self._pct(value),
            )
            self._bar(painter, QRectF(x, y + 12, width, 6), value, color)
            y += 26

        painter.setFont(QFont("Consolas", 6))
        painter.setPen(QColor("#7d8590"))
        if model["dual_scale"]:
            note = (
                f"pesos: epicentro {model['epicenter_weight']:.0%} • "
                f"contexto {model['context_weight']:.0%}"
            )
        else:
            note = "memória legada: somente epicentro disponível"
        painter.drawText(
            QRectF(x, rect.bottom() - 17, width, 12),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._elide(painter, note, width),
        )

    def _draw_hypotheses(self, painter: QPainter, rect: QRectF) -> None:
        model = self.model
        self._section(painter, rect)
        x = rect.x() + 10
        width = rect.width() - 20

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#a6a6a6"))
        painter.drawText(x, rect.y() + 16, "CONTRASTE DE HIPÓTESES")

        rows = [
            ("Defeito NG", model["best_ng_similarity"], QColor("#ff6262")),
            ("Falha falsa OK", model["best_ok_similarity"], QColor("#4ade80")),
        ]
        y = rect.y() + 32
        for label, value, color in rows:
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(color)
            painter.drawText(x, y + 8, label)
            painter.drawText(
                QRectF(x, y, width, 11),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self._pct(value),
            )
            self._bar(painter, QRectF(x, y + 12, width, 7), value, color)
            y += 29

        margin = model["hypothesis_margin"]
        threshold = model["conflict_margin_threshold"]
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffb454") if model["conflict"] else QColor("#c9d1d9"))
        if margin is None:
            margin_text = "Margem: -- • apenas uma hipótese disponível"
        else:
            margin_text = (
                f"Margem: {self._pct(margin)} • limite de conflito: "
                f"{self._pct(threshold)}"
            )
        painter.drawText(
            QRectF(x, rect.bottom() - 24, width, 13),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._elide(painter, margin_text, width),
        )

        result = (
            "CONFLITO • operador 0=OK / 1=NG"
            if model["conflict"]
            else f"Resultado da memória: {model['leading_hypothesis'] or '-'}"
        )
        painter.drawText(
            QRectF(x, rect.bottom() - 12, width, 11),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._elide(painter, result, width),
        )

    def _draw_stats(self, painter: QPainter, rect: QRectF) -> None:
        model = self.model
        self._section(painter, rect)
        x = rect.x() + 10
        width = rect.width() - 20

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#a6a6a6"))
        painter.drawText(x, rect.y() + 15, "PERSISTÊNCIA DA MEMÓRIA")

        if model["prototype_stats_available"]:
            stats = (
                f"OK: {model['ok_prototypes']} protótipo(s) / "
                f"{model['ok_observations']} ocorrência(s)  •  "
                f"NG: {model['protected_ng']} protegido(s)"
            )
        else:
            stats = "Protótipos: telemetria ainda não disponível para este registro"

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#d0d7de"))
        painter.drawText(
            QRectF(x, rect.y() + 22, width, 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._elide(painter, stats, width),
        )

        quantity = (
            "QUANTIDADE NÃO INFLUENCIA O JULGAMENTO"
            if not model["quantity_influence"]
            else "ATENÇÃO: QUANTIDADE ESTÁ INFLUENCIANDO"
        )
        painter.setFont(QFont("Consolas", 6, QFont.Weight.Bold))
        painter.setPen(
            QColor("#4ade80")
            if not model["quantity_influence"]
            else QColor("#ff6262")
        )
        painter.drawText(
            QRectF(x, rect.bottom() - 17, width, 12),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._elide(painter, quantity, width),
        )

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101010"))

        if not self.is_active:
            painter.setPen(QColor("#6e7681"))
            painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Memória visual aguardando inspeção",
            )
            painter.end()
            return

        width = max(1, self.width())
        padding = 8
        header = QRectF(padding, 7, width - padding * 2, 31)
        self._draw_header(painter, header)

        if width >= self.WIDE_BREAKPOINT:
            gap = 8
            content_y = 46
            section_h = 127
            column_w = (width - padding * 2 - gap) / 2
            scales = QRectF(padding, content_y, column_w, section_h)
            hypotheses = QRectF(
                padding + column_w + gap,
                content_y,
                column_w,
                section_h,
            )
            stats = QRectF(padding, content_y + section_h + 8, width - padding * 2, 57)
        else:
            content_y = 46
            scales = QRectF(padding, content_y, width - padding * 2, 126)
            hypotheses = QRectF(padding, content_y + 134, width - padding * 2, 112)
            stats = QRectF(padding, content_y + 254, width - padding * 2, 66)

        self._draw_scales(painter, scales)
        self._draw_hypotheses(painter, hypotheses)
        self._draw_stats(painter, stats)
        painter.end()
