"""Painel horizontal com 6 MAPAS DERIVADOS DA CNN FALTANDO v2.

Sondas dos epicentros AOI em worker Qt; jamais filtros da imagem original.
A decisão operacional KNN/CNN não é alterada.
"""
from __future__ import annotations

import cv2
from PyQt6.QtCore import Qt, QTimer, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

from src.ui.neural_evidence_model import EPICENTERS

VIEWS = ("DIF. LATENTE", "GRAD-CAM", "ATIVAÇÃO CNN")


class _Signals(QObject):
    done = pyqtSignal(int, object, str)


class _NeuralProbeTask(QRunnable):
    """Executa torch fora da thread Qt. Objetos QPixmap só na GUI."""

    def __init__(self, epoch: int, crops: dict):
        super().__init__()
        self.epoch = epoch
        self.crops = crops
        self.signals = _Signals()
        self.setAutoDelete(True)

    def run(self):
        try:
            from src.core.neural.faltando_explainability import explain_epicenters
            maps = explain_epicenters(self.crops)
            self.signals.done.emit(self.epoch, maps, "")
        except Exception as exc:
            self.signals.done.emit(
                self.epoch, None,
                "CNN indisponível: " + type(exc).__name__ + " • " + str(exc)[:170]
            )


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
        if not isinstance(data, dict) or data.get("neural") is not True:
            self.render_status("MAPA CNN INDISPONÍVEL", "Sem resultado neural validado")
            return
        self.metrics.setText(
            f"{data['dimensions'][0]}×{data['dimensions'][1]} • "
            f"{data['layer']} • classe local {data['target_class']}"
        )
        notes = (
            "Diferença das features convolucionais entre (ref,teste) e (ref,ref). "
            "Projeção da CNN, relativa a esta ROI.",
            "Grad-CAM REAL do logit local " + data['target_class'] +
            ". Sonda ROI, não voto operacional da peça.",
            "Energia RMS das features do encoder. "
            "Projeção interna sem decoder nem reconstrução literal RGB.",
        )
        vals = data.get("raw_feature_means", [None, None, None])
        for index, (tile, frame, note) in enumerate(
            zip(self.tiles, data["images"], notes)
        ):
            value = vals[index] if index < len(vals) else None
            text = (f"energia {value:.4f}" if value is not None
                    else "sem métrica")
            tile.render_data(frame, text, note)

    def render_status(self, status: str, detail: str):
        self.metrics.setText(status)
        for tile in self.tiles:
            tile.render_data(None, status, detail)


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
            "SONDAS DA CNN • NÃO ALTERAM A DECISÃO KNN / CNN"
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
        self._epoch = 0
        self._requested_epoch = -1
        self._jobs = {}
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
            "CNN operacional + sondas explicativas das ROIs"
            if active else
            "DECISÃO KNN • CNN em paralelo APENAS para mapas"
        )
        # Apenas APÓS o resultado operacional: nunca disputar CPU durante
        # a decisão do motor CNN ou da KNN da iluminação.
        self._start_probe_after_decision()

    def _on_probe_finished(self, epoch: int, maps: object, error: str):
        self._jobs.pop(epoch, None)
        if epoch != self._epoch:
            return  # Resultado de peça/luz anterior: NUNCA exibir.
        if error or not isinstance(maps, dict):
            self._show_unavailable(error or "Resposta neural inválida")
        else:
            for key in EPICENTERS:
                self.sections[key].render_evidence(maps.get(key))
            self.footer.setText(
                "MAPAS DA CNN VERIFICADA • SONDAS ROI • "
                "NÃO SÃO RECONSTRUÇÕES RGB NEM VOTO OPERACIONAL"
            )
        self._refresh_arrows()

    def _show_unavailable(self, reason: str):
        for section in self.sections.values():
            section.render_status("CNN INDISPONÍVEL", reason)
        self.footer.setText(reason)
        self.setToolTip(reason)

    def _start_probe_after_decision(self):
        payload = self._last_payload
        if not isinstance(payload, dict) or self._requested_epoch == self._epoch:
            return
        self._requested_epoch = self._epoch
        if payload.get("_cnn_explain_allowed") is False:
            self._show_unavailable(
                "Categoria fora do escopo CNN FALTANDO v2 • sem sonda"
            )
            return
        crops = {
            key: payload.get(key).copy()
            for key in ("large_reference", "large", "small_reference", "small")
            if getattr(payload.get(key), "ndim", 0) == 3
        }
        if len(crops) < 2:
            self._show_unavailable("Pares gabarito/teste AOI indisponíveis")
            return
        for section in self.sections.values():
            section.render_status("PROCESSANDO CNN...", "Capturando ativações reais")
        task = _NeuralProbeTask(self._epoch, crops)
        task.signals.done.connect(self._on_probe_finished)
        self._jobs[self._epoch] = task
        QThreadPool.globalInstance().start(task, -1)

    def set_visual_payload(self, payload: dict | None):
        if payload is self._last_payload and isinstance(payload, dict):
            return
        self._epoch += 1
        self._last_payload = payload
        self._requested_epoch = -1
        self._last_analysis = None
        self.scroll.horizontalScrollBar().setValue(0)
        for section in self.sections.values():
            section.render_status(
                "AGUARDANDO JULGAMENTO",
                "Sondas CNN serão executadas após resultado da iluminação"
            )
        self._refresh_arrows()

    def clear_data(self):
        self._epoch += 1  # cancela logicamente qualquer conclusão pendente
        self._requested_epoch = -1
        self._last_payload = None
        self._last_analysis = None
        self.heading.setText("PAINEL NEURAL EXPLICÁVEL")
        self.footer.setText("AGUARDANDO SIDE / TOP / MID")
        for section in self.sections.values():
            section.render_status("AGUARDANDO CNN", "Sem inferência auxiliar")
        self.scroll.horizontalScrollBar().setValue(0)
        self._refresh_arrows()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._refresh_arrows)
