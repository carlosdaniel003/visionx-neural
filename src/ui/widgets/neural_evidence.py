"""Painel de evidência visual AOI: seis imagens em faixa horizontal única.

SIDE/TOP/MID: MAIOR (cinza, diferenças, blocos) + MENOR (mesmas visões).
Não são ativações internas ou Grad-CAM da CNN FALTANDO v2.
Uma única QScrollArea horizontal, sem qualquer rolagem vertical interna.
"""
from __future__ import annotations

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from src.ui.neural_evidence_model import EPICENTERS, VIEWS, epicenter_evidence

CARD_WIDTH = 252
CARD_HEIGHT = 245
STRIP_GAP = 9
STRIP_PADDING = 3
PANEL_HEIGHT = 342
YELLOW = "#f5c518"


class _FittedEvidenceImage(QLabel):
    """Preserva fonte sem zoom infinito e escala novamente após resize."""

    def __init__(self):
        super().__init__("Aguardando imagem")
        self._source = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(0, 0)
        self.setStyleSheet(
            "QLabel {background:#080808; color:#888; border:1px solid #454029;"
            "border-radius:4px;}"
        )

    def set_bgr(self, bgr):
        if bgr is None:
            self._source = QPixmap()
            self.clear()
            self.setText("SEM IMAGEM")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
        self._source = QPixmap.fromImage(image)
        self._redraw()

    def _redraw(self):
        if self._source.isNull():
            return
        size = self.contentsRect().size()
        if size.width() > 0 and size.height() > 0:
            QLabel.setPixmap(
                self, self._source.scaled(
                    size, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()


class _DiagnosticTile(QFrame):
    """Cartão preto/amarelo de altura fixa, sem textos explicativos longos."""

    def __init__(self, epicenter: str, mode: str):
        super().__init__()
        self.setObjectName("neuralEvidenceTile")
        self.epicenter = epicenter
        self.mode = mode
        self.setFixedSize(CARD_WIDTH, CARD_HEIGHT)
        self.setStyleSheet(
            "QFrame#neuralEvidenceTile {background:#101010;"
            "border:1px solid #d3a900; border-radius:9px;}"
            "QLabel {background:transparent; border:none;}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(9, 8, 9, 8)
        root.setSpacing(5)
        label = "MAIOR" if epicenter == "major" else "MENOR"
        self.heading = QLabel(f"EPICENTRO {label} • {mode.upper()}")
        self.heading.setStyleSheet(
            "color:#f5c518;font-size:11px;font-weight:800"
        )
        root.addWidget(self.heading)
        self.image = _FittedEvidenceImage()
        root.addWidget(self.image, 1)
        self.metric = QLabel("Aguardando recorte")
        self.metric.setStyleSheet(
            "color:#d5d5d5; font:10px Consolas;"
        )
        self.metric.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self.metric)

    def render_data(self, frame, info, description):
        self.image.set_bgr(frame)
        self.metric.setText(info)
        self.setToolTip(description)
        self.image.setToolTip(description)


class _EpicenterSection(QWidget):
    """Grupo lógico sem empilhar imagens, mantido para compatibilidade de API."""

    def __init__(self, key: str):
        super().__init__()
        self.key = key
        self.tiles = tuple(_DiagnosticTile(key, mode) for mode in VIEWS)
        self.metrics = QLabel("Aguardando recorte")
        self._columns = 3

    def reflow(self, columns: int):
        # O layout é sempre uma linha horizontal. Nada empilha os cartões.
        self._columns = 3

    def render_evidence(self, data):
        if data is None:
            self.metrics.setText("EPICENTRO INDISPONÍVEL")
            for tile in self.tiles:
                tile.render_data(None, "SEM RECORTE", (
                    "O epicentro não foi encontrado nesta iluminação."
                ))
            return
        w, h = data["dimensions"]
        mean = data["difference_mean"]
        contrast = data["contrast_std"]
        self.metrics.setText(
            f"{w}×{h}px • Δ {mean:.2f}/255 • σ {contrast:.2f}"
        )
        notes = (
            "Cinza: transformação dos pixels do teste. A CNN usa RGB.",
            "Calor: diferença entre gabarito e teste; NÃO é atenção CNN.",
            "Blocos: médias locais dos pixels; NÃO é reconstrução CNN.",
        )
        summaries = (
            f"{w}×{h} px • CINZA",
            f"Δ média {mean:.2f}/255",
            f"Contraste σ {contrast:.2f}",
        )
        for tile, frame, label, tip in zip(
            self.tiles, data["images"], summaries, notes
        ):
            tile.render_data(frame, label, tip)


class NeuralEvidencePanel(QFrame):
    """Faixa única de seis cards e scrollbar horizontal interna."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("neuralEvidencePanel")
        self.setMinimumWidth(0)
        self.setFixedHeight(PANEL_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.setStyleSheet(
            "QFrame#neuralEvidencePanel {background:#080808;"
            "border:1px solid #f5c518;border-radius:9px;}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 7, 10, 7)
        root.setSpacing(5)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        self.heading = QLabel("PAINEL NEURAL EXPLICÁVEL")
        self.heading.setStyleSheet(
            "color:#f5c518;font-size:12px;font-weight:900"
        )
        toolbar.addWidget(self.heading, 1)

        self.left_button = QPushButton("◀")
        self.right_button = QPushButton("▶")
        for button in (self.left_button, self.right_button):
            button.setFixedSize(31, 27)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setStyleSheet(
                "QPushButton {background:#121212;color:#f5c518;"
                "border:1px solid #f5c518;border-radius:5px;font-weight:900;}"
                "QPushButton:disabled {color:#666;border-color:#505050;}"
                "QPushButton:hover:enabled {background:#352d0b;}"
            )
            toolbar.addWidget(button)
        root.addLayout(toolbar)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("neuralEvidenceHorizontalScroll")
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(False)
        self.scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll.setFixedHeight(CARD_HEIGHT + 22)
        self.scroll.setMinimumWidth(0)
        self.scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.scroll.setStyleSheet(
            "QScrollArea {background:#080808;border:none;}"
            "QScrollBar:horizontal {background:#191919;height:12px;"
            "border-radius:4px;}"
            "QScrollBar::handle:horizontal {background:#e0b500;"
            "min-width:38px;border-radius:5px;}"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal"
            " {width:0px;}"
        )
        self.strip = QWidget()
        self.strip.setObjectName("neuralEvidenceStrip")
        self.strip.setStyleSheet("QWidget#neuralEvidenceStrip {background:#080808;}")
        strip_layout = QHBoxLayout(self.strip)
        strip_layout.setContentsMargins(
            STRIP_PADDING, 0, STRIP_PADDING, 0
        )
        strip_layout.setSpacing(STRIP_GAP)
        self.sections = {
            "major": _EpicenterSection("major"),
            "minor": _EpicenterSection("minor"),
        }
        self.tiles = tuple(
            tile
            for section in self.sections.values()
            for tile in section.tiles
        )
        for i, tile in enumerate(self.tiles):
            if i == len(VIEWS):
                # Separador discreto entre EPICENTRO MAIOR / MENOR.
                sep = QFrame()
                sep.setObjectName("epicenterDivider")
                sep.setFixedWidth(2)
                sep.setFixedHeight(CARD_HEIGHT-16)
                sep.setStyleSheet("background:#a48719;border:none;")
                strip_layout.addWidget(sep)
            strip_layout.addWidget(tile)
        total_width = (
            len(self.tiles)*CARD_WIDTH
            + (len(self.tiles))*STRIP_GAP
            + 2*STRIP_PADDING + 2
        )
        self.strip.setFixedSize(total_width, CARD_HEIGHT+2)
        self.scroll.setWidget(self.strip)
        root.addWidget(self.scroll)

        self.footer = QLabel(
            "PIXELS AOI • CINZA / DIFERENÇAS / BLOCOS"
        )
        self.footer.setStyleSheet(
            "color:#aaa; font:9px Consolas;"
        )
        root.addWidget(self.footer)

        self.left_button.clicked.connect(lambda: self.scroll_by(-1))
        self.right_button.clicked.connect(lambda: self.scroll_by(1))
        self.scroll.horizontalScrollBar().valueChanged.connect(
            lambda _value: self._refresh_arrows()
        )
        self._last_payload = None
        self._last_analysis = None
        self._columns = 6
        self._refresh_arrows()
        QTimer.singleShot(0, self._refresh_arrows)

    @staticmethod
    def columns_for_width(width: int) -> int:
        # Nunca reordenar em linhas. A janela mostra tantos cards quanto cabem.
        return max(1, min(6, int(max(0,width) / (CARD_WIDTH+STRIP_GAP))))

    def _reflow(self, width):
        # Responsividade por viewport/scroll: altura sempre fixa, sem empilhar.
        self._columns = 6
        self._refresh_arrows()

    def _refresh_arrows(self):
        bar = self.scroll.horizontalScrollBar()
        self.left_button.setEnabled(bar.value() > bar.minimum())
        self.right_button.setEnabled(bar.value() < bar.maximum())

    def scroll_by(self, direction: int):
        bar = self.scroll.horizontalScrollBar()
        step = (CARD_WIDTH + STRIP_GAP) * 2
        bar.setValue(bar.value() + step*(1 if direction > 0 else -1))

    def update_data(self, detail: dict, analysis: dict | None = None):
        self._last_analysis = analysis
        d = detail if isinstance(detail, dict) else {}
        active = bool(d.get("cnn_v2_active"))
        route = str(d.get("recognition_route", "") or "")
        suffix = ("CNN FALTANDO v2" if active else
                  "KNN EXATO" if route == "KNOWN_KNN" else "AOI")
        self.heading.setText(f"PAINEL NEURAL EXPLICÁVEL • {suffix}")
        self.footer.setText(
            "VISUALIZAÇÃO DE PIXELS • NÃO É ATENÇÃO CNN"
            if active else
            "VISUALIZAÇÃO DE PIXELS • CNN NÃO EXECUTADA NESTA LUZ"
        )

    def set_visual_payload(self, payload: dict | None):
        self._last_payload = payload
        result = epicenter_evidence(payload)
        for key in EPICENTERS:
            self.sections[key].render_evidence(result.get(key))
        self.scroll.horizontalScrollBar().setValue(0)
        self._refresh_arrows()

    def clear_data(self):
        self._last_payload = None
        self._last_analysis = None
        self.heading.setText("PAINEL NEURAL EXPLICÁVEL")
        self.footer.setText("AGUARDANDO SIDE / TOP / MID")
        for section in self.sections.values():
            section.render_evidence(None)
        self.scroll.horizontalScrollBar().setValue(0)
        self._refresh_arrows()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._refresh_arrows)
