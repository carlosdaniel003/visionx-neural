"""Painel responsivo da memória visual do VisionX."""

from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF, QSize
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from src.ui.memory_status_model import memory_status_from_detail
from src.ui.neural_telemetry_model import memory_panel_text, memory_seen_state, LIGHTS


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
        self.recognition_route = str((detail or {}).get("recognition_route", "") or "")
        self.route_header, self.route_explanation = memory_panel_text({"detail": detail})
        self.seen_state = memory_seen_state({"detail": detail})
        self.route_labels = {
            light: str((detail or {}).get("cnn_v2_light_diagnostics", {}).get(light, {}).get("human_memory_label", "") or "")
            for light in LIGHTS
        }
        self.setToolTip(self.route_header + "\n" + self.route_explanation)
        self.is_active = bool(self.model["active"])
        self.has_memory = bool(self.model["has_memory"])
        self.memory_available = bool(self.model["memory_available"])
        self.stored_memory_available = bool(self.model["stored_memory_available"])
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
        if model["hard_missing_override"]:
            color = QColor("#ff6262")
            title = (
                "AUSÊNCIA FÍSICA FORTE • "
                f"KNN AUDITORIA • MATCH {self._pct(model['combined_similarity'])}"
            )
        elif model["conflict"]:
            color = QColor("#ffb454")
            title = (
                "CONFLITO DE MEMÓRIA • "
                f"MATCH {self._pct(model['combined_similarity'])} • "
                "REVISÃO HUMANA"
            )
        elif model["has_memory"]:
            label = model["leading_hypothesis"] or "-"
            color = QColor("#ff6262") if label == "NG" else QColor("#4ade80")
            title = (
                f"JÁ VISTO • HIPÓTESE {label} • "
                f"MATCH {self._pct(model['combined_similarity'])}"
            )
        elif model["memory_available"] or model["visual_match_available"]:
            color = QColor("#f5c518")
            title = (
                "MEMÓRIA ENCONTRADA • "
                f"MATCH {self._pct(model['combined_similarity'])} • "
                "ABAIXO DO LIMIAR"
            )
        elif model["stored_memory_available"]:
            color = QColor("#f5c518")
            count = model["category_candidate_count"]
            title = (
                f"MEMÓRIA CARREGADA • {count} REGISTRO(S) • "
                "SEM MATCH VISUAL VÁLIDO"
            )
        else:
            color = QColor("#6e7681")
            title = "PRIMEIRA OCORRÊNCIA • SEM MEMÓRIA DA CATEGORIA"

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
        painter.drawText(int(x), int(rect.y() + 16), "ESCALAS DA CORRESPONDÊNCIA LÍDER")

        rows = [
            ("Epicentro", model["epicenter_similarity"], QColor("#58a6ff")),
            ("Contexto", model["context_similarity"], QColor("#bc8cff")),
            ("Melhor match", model["combined_similarity"], QColor("#f5c518")),
        ]
        y = rect.y() + 31
        for label, value, color in rows:
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QColor("#c9d1d9"))
            painter.drawText(int(x), int(y + 8), label)
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
        painter.drawText(int(x), int(rect.y() + 16), "CONTRASTE DE HIPÓTESES")

        rows = [
            ("Defeito NG", model["best_ng_similarity"], QColor("#ff6262")),
            ("Falha falsa OK", model["best_ok_similarity"], QColor("#4ade80")),
        ]
        y = rect.y() + 32
        for label, value, color in rows:
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(color)
            painter.drawText(int(x), int(y + 8), label)
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

        if model["hard_missing_override"]:
            result = (
                "KNN somente auditoria • ausência física tem prioridade"
            )
        elif model["conflict"]:
            result = "CONFLITO • operador 0=OK / 1=NG"
        else:
            result = f"Resultado da memória: {model['leading_hypothesis'] or '-'}"
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
        painter.drawText(int(x), int(rect.y() + 15), "PERSISTÊNCIA DA MEMÓRIA")

        if model["category_candidate_count"] > 0:
            prefix = (
                f"Categoria: {model['category_candidate_count']} registro(s) em memória"
            )
        elif model["stored_memory_available"]:
            prefix = "Categoria: memória carregada"
        else:
            prefix = "Categoria: nenhuma memória armazenada"

        if model["prototype_stats_available"]:
            stats = (
                f"{prefix}  •  OK: {model['ok_prototypes']} protótipo(s) / "
                f"{model['ok_observations']} ocorrência(s)  •  "
                f"NG: {model['protected_ng']} protegido(s)"
            )
        else:
            stats = f"{prefix}  •  protótipos sem telemetria"

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

    def _draw_recognition_dashboard(self, painter: QPainter) -> None:
        """Área KNN inteira preenchida por indicadores reais de SIDE/TOP/MID.

        As cores representam a existência do par humano EXATO, nunca
        probabilidade de acerto ou similaridade da CNN.
        """
        info = getattr(self, "seen_state", {})
        state = info.get("status")
        known = info.get("recognized_lights", [])
        multilight = bool(info.get("has_multilight"))
        w, h = float(self.width()), float(self.height())

        # Cabeçalho amplo, bem contrastado, sem fundo vazio.
        color = QColor("#4ade80" if state == "JA_VI" else
                       "#f5c518" if state == "NUNCA_VI" else "#8da2b5")
        painter.setPen(QPen(color, 2))
        painter.setBrush(QColor("#14231d" if state == "JA_VI" else
                                "#252012" if state == "NUNCA_VI" else "#182028"))
        painter.drawRoundedRect(QRectF(8, 8, w-16, 66), 9, 9)
        painter.setPen(color)
        painter.setFont(QFont("Consolas", 17, QFont.Weight.Bold))
        painter.drawText(
            QRectF(18, 12, w-36, 30),
            Qt.AlignmentFlag.AlignCenter,
            info.get("label") or "CONSULTA NÃO CONFIRMADA",
        )
        painter.setPen(QColor("#d6e5ed"))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        caption = (
            f"{len(known)}/3 ILUMINAÇÕES RECONHECIDAS NA MEMÓRIA KNN"
            if multilight and state else
            "PAR HUMANO EXATO RECUPERADO" if state == "JA_VI" else
            "NENHUM PAR EXATO RECUPERADO" if state == "NUNCA_VI" else
            "NÃO HÁ ROTA KNN VERIFICADA"
        )
        painter.drawText(QRectF(18, 43, w-36, 21),
                         Qt.AlignmentFlag.AlignCenter, caption)

        light_routes = info.get("routes_by_light", {})
        cards = (
            [(light, light_routes.get(light, ""), self.route_labels.get(light, ""))
             for light in LIGHTS]
            if multilight else
            [("INSPEÇÃO", self.recognition_route,
              self.route_labels.get("INSPEÇÃO", self.best_label))]
        )
        vertical = w < 660
        gap = 10
        left, right, top, bottom = 8.0, 8.0, 88.0, 32.0
        usable_h = max(75.0, h - top - bottom)
        if vertical:
            box_h = max(42.0, (usable_h - gap*(len(cards)-1))/len(cards))
            box_w = w - left - right
        else:
            box_h = usable_h
            box_w = (w-left-right-gap*(len(cards)-1))/len(cards)

        for i, (light, route, human_label) in enumerate(cards):
            if vertical:
                x,y = left, top + i*(box_h + gap)
            else:
                x,y = left + i*(box_w + gap), top
            if route == "KNOWN_KNN":
                accent, fill = "#4ade80", "#152c20"
                primary = "JÁ VI"
                secondary = "KNN • PAR EXATO" + (
                    f" • {human_label.upper()}" if human_label.upper() in {"OK","NG"} else "")
            elif route in {"NEW_CNN", "NEW_EXPERTS"}:
                accent, fill = "#f5c518", "#2a2418"
                primary = "NUNCA VI"
                secondary = "CNN FALTANDO v2" if route == "NEW_CNN" else "OUTRO ESPECIALISTA"
                secondary += " • SEM MATCH EXATO"
            else:
                accent, fill = "#8093a3", "#19232a"
                primary, secondary = "N/D", "SEM CONSULTA CONFIRMADA"
            rect = QRectF(x,y,box_w,box_h)
            painter.setPen(QPen(QColor(accent), 1.5))
            painter.setBrush(QColor(fill))
            painter.drawRoundedRect(rect, 8, 8)

            # Marcador preenchido: status visual independente de texto.
            painter.setBrush(QColor(accent))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(x+15,y+14,13,13))
            painter.setPen(QColor("#e4edf4"))
            painter.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
            painter.drawText(QRectF(x+36,y+7,box_w-48,28),
                             Qt.AlignmentFlag.AlignVCenter,
                             light)
            painter.setPen(QColor(accent))
            painter.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
            painter.drawText(QRectF(x+12,y+36,box_w-24,28),
                             Qt.AlignmentFlag.AlignCenter, primary)
            painter.setPen(QColor("#cfdae4"))
            painter.setFont(QFont("Consolas", 8))
            text_w = box_w - 24
            painter.drawText(
                QRectF(x+12, y+67 if box_h>=105 else y+60,
                       text_w, max(17, box_h-75)),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
                | Qt.TextFlag.TextWordWrap,
                secondary if text_w >= 230 else self._elide(painter, secondary, text_w)
            )

        painter.setPen(QColor("#9caeb8"))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(QRectF(10, h-29, w-20, 23),
                         Qt.AlignmentFlag.AlignCenter,
                         "Já vi = ≥1 luz conhecida • Nunca vi = nenhuma luz conhecida • pares exatos")

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101010"))

        if getattr(self, "recognition_route", "") or getattr(self, "seen_state", {}).get("has_multilight"):
            self._draw_recognition_dashboard(painter)
            painter.end()
            return

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
