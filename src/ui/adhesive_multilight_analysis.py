"""Painel responsivo de especialistas SIDE/TOP/MID.

O nome do módulo é mantido por compatibilidade histórica. SIDE, TOP e MID
recebem análises independentes do pipeline multilight de qualquer categoria AOI.
A camada é visual e não altera a decisão fundida da peça.
"""

from __future__ import annotations

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.ui.widgets.radar_chart import RadarChartWidget
from src.ui.widgets.neural_specialist import NeuralSpecialistWidget, VerifiedMemorySpecialistWidget
from src.ui.widgets.semantic_dna import SemanticDNAWidget
from src.ui.widgets.shift_debugger import ShiftDebuggerWidget
from src.ui.widgets.silk_debugger import SilkDebuggerWidget
from src.ui.widgets.ssim_debugger import SSIMDebuggerWidget


LIGHTING_ORDER = ("SIDE", "TOP", "MID")
LIGHTING_SYMBOL = {
    "SIDE": "↓",
    "TOP": "←",
    "MID": "→",
}
LARGE_MONITOR_COLUMNS_BREAKPOINT = 1500
EXPERT_MIN_WIDTH = 340
EXPERT_MIN_HEIGHT = 285

EXPERT_DEFINITIONS = (
    ("ssim_expert.py", "SSIM • TEXTURA E CALOR", SSIMDebuggerWidget),
    ("silk_expert.py", "XOR • TINTA E EPICENTRO", SilkDebuggerWidget),
    ("semantic_expert.py", "DNA • ASSINATURA SEMÂNTICA", SemanticDNAWidget),
    ("shift_expert.py", "SHIFT • DESLOCAMENTO", ShiftDebuggerWidget),
    ("faltando_cnn_v2.py", "CNN FALTANDO v2 • REDE NEURAL", NeuralSpecialistWidget),
    ("knn_expert.py", "KNN • MEMÓRIA HUMANA", VerifiedMemorySpecialistWidget),
)


class _LightingExpertLane(QFrame):
    """Conjunto de especialistas pertencente a uma única iluminação."""

    def __init__(self, mode: str):
        super().__init__()
        self.mode = str(mode).strip().upper()
        self.setObjectName("imageCard")
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)

        title = QLabel(
            f"ANÁLISE {self.mode}  {LIGHTING_SYMBOL.get(self.mode, '')}"
        )
        title.setObjectName("eyebrowLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        self.status_label = QLabel(
            f"Aguardando análise da iluminação {self.mode}"
        )
        self.status_label.setObjectName("sectionHint")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(48)
        root.addWidget(self.status_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setMinimumWidth(0)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.scroll_content = QWidget()
        self.scroll_content.setMinimumWidth(EXPERT_MIN_WIDTH + 4)
        self.scroll_content.setMinimumHeight(EXPERT_MIN_HEIGHT + 8)
        self.scroll_layout = QHBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll_layout.setSpacing(10)

        self.frames = {}
        for engine_name, label, widget_cls in EXPERT_DEFINITIONS:
            widget = widget_cls()
            widget.setObjectName("debugCard")
            widget.setToolTip(label)
            widget.setAccessibleName(
                f"{self.mode} • {label}"
            )
            widget.setMinimumWidth(EXPERT_MIN_WIDTH)
            widget.setMinimumHeight(EXPERT_MIN_HEIGHT)
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            widget.setVisible(False)
            self.frames[engine_name] = widget
            self.scroll_layout.addWidget(widget)

        self.radar = RadarChartWidget()
        self.radar.setObjectName("debugCard")
        self.radar.setToolTip("FUSÃO • SCORE FINAL")
        self.radar.setAccessibleName(
            f"{self.mode} • FUSÃO • SCORE FINAL"
        )
        self.radar.setMinimumWidth(EXPERT_MIN_WIDTH)
        self.radar.setMinimumHeight(EXPERT_MIN_HEIGHT)
        self.radar.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.radar.setVisible(False)
        self.scroll_layout.addWidget(self.radar)
        self.scroll_layout.addStretch()

        self.scroll.setWidget(self.scroll_content)
        self.scroll.setMinimumHeight(EXPERT_MIN_HEIGHT + 28)
        self.scroll.setVisible(False)
        root.addWidget(self.scroll, stretch=1)

    def _hide_all_frames(self) -> None:
        for widget in self.frames.values():
            widget.setVisible(False)
        self.radar.setVisible(False)

    def clear_analysis(self) -> None:
        self._hide_all_frames()
        self.scroll.setVisible(False)
        self.status_label.setText(
            f"Aguardando análise da iluminação {self.mode}"
        )
        self.status_label.setVisible(True)

    def set_analysis(self, analysis: dict | None) -> bool:
        if not isinstance(analysis, dict):
            self.clear_analysis()
            return False

        detail = analysis.get("detail", {})
        active_engines = list(analysis.get("active_engines", []) or [])
        self._hide_all_frames()

        visible_count = 0
        for engine_name, widget in self.frames.items():
            if engine_name not in active_engines:
                continue
            if engine_name in {"faltando_cnn_v2.py", "knn_expert.py"}:
                widget.update_data(detail, analysis)
            else:
                widget.update_data(detail)
            widget.setVisible(True)
            visible_count += 1

        if not active_engines:
            self.radar.update_data(detail)
            self.radar.setVisible(True)
            visible_count += 1

        # Mantém a mesma semântica do painel original: só aparecem os motores
        # declarados como ativos; o radar é fallback quando não há especialistas.
        self.scroll_content.setMinimumWidth(
            max(
                EXPERT_MIN_WIDTH + 4,
                visible_count * EXPERT_MIN_WIDTH
                + max(0, visible_count - 1) * 10
                + 4,
            )
        )

        if visible_count <= 0:
            self.status_label.setText(
                f"Análise {self.mode} recebida, sem especialista visual ativo"
            )
            self.status_label.setVisible(True)
            self.scroll.setVisible(False)
            return True

        self.status_label.setText(
            f"Análise {self.mode} disponível • "
            f"{visible_count} painel(is) de especialista"
        )
        self.status_label.setVisible(True)
        self.scroll.setVisible(True)
        return True


class AdhesiveMultiLightAnalysisView(QWidget):
    """SIDE/TOP/MID responsivos na área ANÁLISE DOS ESPECIALISTAS."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("adhesiveMultiLightAnalysisView")
        self.setMinimumWidth(0)
        self._columns = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(7)

        hint = QLabel(
            "MULTILIGHT • análise independente dos especialistas em "
            "SIDE / TOP / MID • sem fusão de resultado final nesta etapa."
        )
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(8)

        self.lanes = {
            mode: _LightingExpertLane(mode)
            for mode in LIGHTING_ORDER
        }
        root.addLayout(self.grid)

        # Um único scroll horizontal mestre controla SIDE/TOP/MID em conjunto.
        # Isso evita três barras pequenas e mantém os mesmos especialistas
        # alinhados visualmente entre as iluminações.
        self.horizontal_scroll = QScrollBar(Qt.Orientation.Horizontal)
        self.horizontal_scroll.setObjectName(
            "adhesiveExpertHorizontalScroll"
        )
        self.horizontal_scroll.setRange(0, 0)
        self.horizontal_scroll.setEnabled(False)
        self.horizontal_scroll.valueChanged.connect(
            self._sync_lanes_from_master
        )
        root.addWidget(self.horizontal_scroll)

        self._reflow(LARGE_MONITOR_COLUMNS_BREAKPOINT + 1)
        QTimer.singleShot(0, self._sync_master_scroll_range)

    @staticmethod
    def columns_for_width(width: int) -> int:
        return (
            3
            if int(width) >= LARGE_MONITOR_COLUMNS_BREAKPOINT
            else 1
        )

    def _reflow(self, width: int) -> None:
        columns = self.columns_for_width(width)
        if columns == self._columns and self.grid.count() == len(self.lanes):
            return

        for lane in self.lanes.values():
            self.grid.removeWidget(lane)

        for index, mode in enumerate(LIGHTING_ORDER):
            row = index // columns
            column = index % columns
            self.grid.addWidget(self.lanes[mode], row, column)

        for column in range(3):
            self.grid.setColumnStretch(
                column,
                1 if column < columns else 0,
            )
        self._columns = columns

    def _sync_master_scroll_range(self) -> None:
        maxima = []
        page_steps = []
        for lane in self.lanes.values():
            bar = lane.scroll.horizontalScrollBar()
            maxima.append(int(bar.maximum()))
            page_steps.append(int(bar.pageStep()))

        maximum = max(maxima, default=0)
        self.horizontal_scroll.blockSignals(True)
        self.horizontal_scroll.setRange(0, maximum)
        self.horizontal_scroll.setPageStep(
            max(1, min(page_steps, default=1))
        )
        self.horizontal_scroll.setEnabled(maximum > 0)
        if maximum <= 0:
            self.horizontal_scroll.setValue(0)
        elif self.horizontal_scroll.value() > maximum:
            self.horizontal_scroll.setValue(maximum)
        self.horizontal_scroll.blockSignals(False)
        self._sync_lanes_from_master(self.horizontal_scroll.value())

    def _sync_lanes_from_master(self, value: int) -> None:
        master_max = int(self.horizontal_scroll.maximum())
        ratio = (
            float(value) / float(master_max)
            if master_max > 0
            else 0.0
        )
        for lane in self.lanes.values():
            bar = lane.scroll.horizontalScrollBar()
            lane_max = int(bar.maximum())
            bar.setValue(
                int(round(ratio * lane_max))
                if lane_max > 0
                else 0
            )

    def horizontal_scroll_bars(self):
        """Contrato usado pelo Modo Produção para apresentação automática."""
        self._sync_master_scroll_range()
        return [self.horizontal_scroll]

    def set_analysis(self, mode: str, analysis: dict | None) -> bool:
        normalized = str(mode or "").strip().upper()
        lane = self.lanes.get(normalized)
        if lane is None:
            return False
        result = lane.set_analysis(analysis)
        QTimer.singleShot(0, self._sync_master_scroll_range)
        return result

    def clear_analysis(self, mode: str) -> bool:
        normalized = str(mode or "").strip().upper()
        lane = self.lanes.get(normalized)
        if lane is None:
            return False
        lane.clear_analysis()
        return True

    def clear_all(self) -> None:
        for lane in self.lanes.values():
            lane.clear_analysis()
        self.horizontal_scroll.setRange(0, 0)
        self.horizontal_scroll.setValue(0)
        self.horizontal_scroll.setEnabled(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow(event.size().width())
        QTimer.singleShot(0, self._sync_master_scroll_range)


__all__ = [
    "AdhesiveMultiLightAnalysisView",
    "EXPERT_MIN_HEIGHT",
    "EXPERT_MIN_WIDTH",
    "LARGE_MONITOR_COLUMNS_BREAKPOINT",
    "LIGHTING_ORDER",
]
