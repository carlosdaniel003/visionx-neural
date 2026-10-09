"""Visualização responsiva da influência dos motores na decisão final."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from src.ui.decision_model import influence_rows
from src.ui.neural_influence_model import neural_influence_rows


class DecisionInfluenceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(285)
        self.setMinimumWidth(320)
        self.trace = {}
        self.rows = []

    def update_data(self, analysis: dict | None):
        detail = (analysis or {}).get("detail", {})
        trace = detail.get("decision_trace", {})
        self.trace = trace if isinstance(trace, dict) else {}
        self.rows = neural_influence_rows(analysis) or influence_rows(self.trace)
        if self.rows and self.rows[0].get("telemetry_row"):
            self.setToolTip(
                "Verde: falha falsa/OK; vermelho: defeito/NG. "
                "A barra representa os scores brutos da CNN por iluminação, "
                "não acurácia nem probabilidade calibrada. "
                "KNN exato recupera somente rótulos humanos confirmados."
            )
        else:
            self.setToolTip("Influência dos especialistas ativos na decisão.")
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
            if row.get("multilight_local_origin", False):
                return "ORIGEM LOCAL"
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
            match = float(row.get("evidence_score", 0.0))
            if score < 0.5:
                vote_label = "OK"
            elif score > 0.5:
                vote_label = "NG"
            else:
                vote_label = "INCONCLUSIVO"
            return (
                f"match {match:.0%} • voto {vote_label} • "
                f"peso {weight:.0%} • efeito {effect_text}"
            )

        if row.get("telemetry_row"):
            return str(row.get("display_text", status))

        if weight > 0.0:
            return (
                f"{score:.0%}/{threshold:.0%} • peso {weight:.0%} • "
                f"parcela {contribution:.0%}"
            )

        return f"{score:.0%}/{threshold:.0%} • {status} • peso direto 0%"

    def _paint_neural(self, painter: QPainter, width: int, height: int) -> None:
        """Cards SIDE/TOP/MID, verde=OK, vermelho=NG; sem pesos inventados."""
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#f5c518"))
        painter.drawText(
            QRectF(12, 5, width - 24, 23),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "COMO CADA MOTOR JULGOU",
        )
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#9aa6b2"))
        painter.drawText(
            QRectF(12, 28, width - 24, 20),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "Verde: sinal OK  |  Vermelho: sinal NG  |  Amarelo: revisão",
        )

        n = len(self.rows)
        top, bottom, gap = 55, 35, 7
        row_height = max(40.0, (height - top - bottom - gap * (n - 1)) / max(n, 1))
        for index, row in enumerate(self.rows):
            y = top + index * (row_height + gap)
            h = min(row_height, max(35.0, height - bottom - y))
            rect = QRectF(8, y, width - 16, h)
            score = row.get("ng_score_uncalibrated")
            known_label = str(row.get("known_memory_label", "")).upper()
            verdict = str(row.get("summary", "")).upper()
            if known_label in {"OK", "NG"}:
                state = known_label
                status_text = f"MEMÓRIA HUMANA • {known_label} • PAR EXATO"
            elif "FALHA FALSA" in verdict:
                state = "OK"
                status_text = "OK • FALHA FALSA"
            elif "DEFEITO REAL" in verdict:
                state = "NG"
                status_text = "NG • DEFEITO REAL"
            else:
                state = "REVISÃO"
                status_text = "REVISÃO • EVIDÊNCIA INSUFICIENTE"

            base_color = QColor(
                "#4ade80" if state == "OK" else
                "#ff6262" if state == "NG" else "#f5c518"
            )
            painter.setPen(QPen(base_color, 1))
            painter.setBrush(QColor("#151e22"))
            painter.drawRoundedRect(rect, 7, 7)

            label_space = max(75.0, rect.width() * .40)
            painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#dae5ee"))
            painter.drawText(
                QRectF(rect.x() + 10, y + 3, label_space - 8, 19),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                self._elide(painter, row.get("label", ""), int(label_space - 10)),
            )
            painter.setPen(base_color)
            painter.drawText(
                QRectF(rect.x() + label_space, y + 3, rect.width() - label_space - 9, 19),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                self._elide(painter, status_text, int(rect.width() - label_space - 10)),
            )
            if score is not None:
                # O comprimento representa o score bruto da CNN, não certeza.
                bar_rect = QRectF(rect.x() + 10, y + 28, rect.width() - 20, 10)
                ok_width = bar_rect.width() * (1.0 - float(score))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#4ade80"))
                if ok_width > 0:
                    painter.drawRect(QRectF(bar_rect.x(), bar_rect.y(), ok_width, 10))
                painter.setBrush(QColor("#ff6262"))
                if bar_rect.width() - ok_width > 0:
                    painter.drawRect(QRectF(bar_rect.x() + ok_width, bar_rect.y(),
                                            bar_rect.width() - ok_width, 10))
                painter.setPen(QColor("#acb8c4"))
                painter.setFont(QFont("Consolas", 7))
                painter.drawText(
                    QRectF(rect.x() + 10, y + 39, rect.width() - 20, 15),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    f"Score NG bruto {score * 100:.4f}%  •  OK complementar "
                    f"{(1-score)*100:.4f}%",
                )
            else:
                painter.setFont(QFont("Consolas", 7))
                painter.setPen(QColor("#9aa6b2"))
                painter.drawText(
                    QRectF(rect.x() + 10, y + 24, rect.width() - 20, 21),
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    "KNN: rótulo humano exato • sem score de probabilidade",
                )

        painter.setPen(QColor("#98a5b1"))
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(
            QRectF(10, max(0, height - 29), width - 20, 25),
            Qt.AlignmentFlag.AlignCenter,
            "Scores CNN não calibrados. Votos multilight não são soma de pesos.",
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        painter.fillRect(0, 0, width, height, QColor("#101010"))

        if self.rows and self.rows[0].get("telemetry_row"):
            self._paint_neural(painter, width, height)
            painter.end()
            return

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

            # Barra principal: evidência física; no KNN, similaridade visual.
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
            evidence_score = max(
                0.0,
                min(1.0, float(row.get("evidence_score", score))),
            )
            if evidence_score > 0.0:
                painter.setBrush(color)
                painter.drawRoundedRect(
                    QRectF(
                        bar_x,
                        bar_y,
                        bar_width * evidence_score,
                        bar_h,
                    ),
                    3,
                    3,
                )

            threshold = max(
                0.0,
                min(
                    1.0,
                    float(
                        row.get(
                            "evidence_threshold",
                            row["threshold"],
                        )
                    ),
                ),
            )
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
        knn_match = float(memory.get("best_similarity", 0.0) or 0.0)
        if knn_vote < 0.5:
            knn_vote_label = "OK"
        elif knn_vote > 0.5:
            knn_vote_label = "NG"
        else:
            knn_vote_label = "INCONCLUSIVO"

        footer_y = height - footer_height + 2
        painter.setPen(QColor("#d0d0d0"))
        if self.rows and self.rows[0].get("telemetry_row"):
            if len(self.rows) > 1:
                formula = "CNN/KNN por iluminação • votos independentes • sem soma linear"
            else:
                formula = "Motor único • " + self.rows[0]["label"]
        elif str(self.trace.get("dominant_engine", "")) == "multilight":
            origin = str(self.trace.get("multilight_dominant_mode", "-"))
            formula = (
                f"Fusão SIDE/TOP/MID • origem: {origin} • "
                f"veredito: {self.trace.get('verdict', '-')} • score {final_score:.0%}"
            )
        else:
            formula = (
                f"Fusão: físico {physical:.0%}×{physical_weight:.0%} + "
                f"KNN {knn_vote_label} ({knn_vote:.0%} NG)×{knn_weight:.0%} "
                f"= {final_score:.0%} • match {knn_match:.0%}"
            )
        painter.drawText(
            padding,
            int(footer_y + 14),
            self._elide(painter, formula, width - padding * 2),
        )

        painter.setPen(QColor("#f5c518"))
        footer_2 = (
            f"Score CNN não calibrado • regra {self.trace.get('fusion_rule', '-')} • " 
            "verde/rosa = força do voto, não probabilidade de acerto"
            if self.rows and self.rows[0].get("telemetry_row") else
            f"Corte {cutoff:.0%} • regra {self.trace.get('fusion_rule', 'physical_only')} • "
            "barra maior = evidência (KNN = match); barra amarela fina = peso"
        )
        painter.drawText(
            padding,
            int(footer_y + 32),
            self._elide(painter, footer_2, width - padding * 2),
        )
        painter.end()
