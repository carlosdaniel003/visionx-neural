"""Responsive AOI neural-evidence board, one card per illumination.

The pictures are deterministic pixel diagnostics, NOT CNN activations.
The CNN's actual forward path is RGB full frame + central 70% (checkpoint
metadata controls the fraction); the AOI epicenters are independent.
"""
from __future__ import annotations

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from src.ui.neural_evidence_model import EPICENTERS, VIEWS, epicenter_evidence


class _FittedEvidenceImage(QLabel):
    def __init__(self):
        super().__init__("Sem recorte")
        self._source = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(0, 94)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "background:#0b1118; color:#8d9ca8; border:1px solid #293b4a;"
            "border-radius:5px;"
        )

    def set_bgr(self, bgr) -> None:
        if bgr is None:
            self._source = QPixmap()
            self.clear()
            self.setText("Recorte indisponível")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h,w,_ = rgb.shape
        self._source = QPixmap.fromImage(QImage(
            rgb.data, w, h, 3*w, QImage.Format.Format_RGB888,
        ).copy())
        self._redraw()

    def _redraw(self):
        if self._source.isNull():
            return
        size = self.contentsRect().size()
        if size.width() > 0 and size.height() > 0:
            QLabel.setPixmap(self, self._source.scaled(
                size, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()


class _DiagnosticTile(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("neuralEvidenceTile")
        self.setMinimumWidth(0)
        self.setStyleSheet(
            "QFrame#neuralEvidenceTile {background:#121c26;"
            "border:1px solid #31404b; border-radius:6px;}"
            "QLabel {background:transparent; border:none;}"
        )
        root=QVBoxLayout(self)
        root.setContentsMargins(6, 5, 6, 5)
        root.setSpacing(4)
        self.heading=QLabel(title)
        self.heading.setStyleSheet("color:#f5c518;font-size:11px;font-weight:800")
        root.addWidget(self.heading)
        self.image=_FittedEvidenceImage()
        root.addWidget(self.image,1)
        self.metric=QLabel("Aguardando recorte")
        self.metric.setWordWrap(True)
        self.metric.setStyleSheet("color:#c0cfdb;font:9px Consolas")
        root.addWidget(self.metric)

    def render_data(self, frame, metrics):
        self.image.set_bgr(frame)
        self.metric.setText(metrics)
        self.setToolTip(metrics)


class _EpicenterSection(QWidget):
    def __init__(self, label: str):
        super().__init__()
        self.setMinimumWidth(0)
        outer=QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0)
        outer.setSpacing(4)
        self.title=QLabel(label)
        self.title.setStyleSheet(
            "color:#f5c518;font-weight:800;font-size:12px"
        )
        outer.addWidget(self.title)
        self.metrics=QLabel("Aguardando recorte do epicentro")
        self.metrics.setWordWrap(True)
        self.metrics.setStyleSheet("color:#cbd7e2;font:10px Consolas")
        outer.addWidget(self.metrics)
        self.grid=QGridLayout()
        self.grid.setContentsMargins(0,0,0,0)
        self.grid.setSpacing(6)
        self.tiles=tuple(_DiagnosticTile(t) for t in VIEWS)
        outer.addLayout(self.grid)
        self._columns=0
        self.reflow(3)

    def reflow(self, columns: int):
        columns=max(1,min(3,int(columns)))
        if columns==self._columns:
            return
        for widget in self.tiles:
            self.grid.removeWidget(widget)
        for i,widget in enumerate(self.tiles):
            self.grid.addWidget(widget, i//columns, i%columns)
        for column in range(3):
            self.grid.setColumnStretch(column, 1 if column<columns else 0)
        self._columns=columns

    def render_evidence(self,data):
        if not data:
            self.metrics.setText(
                "Recorte AOI indisponível: não é possível validar visualmente esta região."
            )
            for tile in self.tiles:
                tile.render_data(None,"Sem pixels; nenhum score inferido.")
            return
        width,height=data["dimensions"]
        self.metrics.setText(
            f"Recorte AOI {width}×{height} • diferença média {data['difference_mean']:.2f}/255"
            f" • contraste σ {data['contrast_std']:.2f}"
        )
        explanatory=(
            "Tons de cinza da imagem do teste (CNN recebe RGB).",
            "Diferença absoluta teste–gabarito; NÃO é atenção da CNN.",
            "Médias locais em blocos; NÃO é reconstrução interna da CNN.",
        )
        for tile,frame,tip in zip(self.tiles,data["images"],explanatory):
            tile.render_data(frame,tip)


class NeuralEvidencePanel(QFrame):
    """Adaptive 3/2/1 columns with the two actual AOI regions per light."""

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName("neuralEvidencePanel")
        self.setMinimumWidth(320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Preferred)
        self.setStyleSheet(
            "QFrame#neuralEvidencePanel {background:#101820;"
            "border:1px solid #536778;border-radius:9px;}"
        )
        root=QVBoxLayout(self)
        root.setContentsMargins(10,8,10,8)
        root.setSpacing(9)
        self.heading=QLabel("PAINEL NEURAL EXPLICÁVEL")
        self.heading.setStyleSheet(
            "color:#f5c518;font-weight:900;font-size:13px"
        )
        root.addWidget(self.heading)
        self.subtitle=QLabel(
            "Diagnóstico dos pixels da AOI • não é Grad-CAM nem atenção da rede"
        )
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet("color:#9fb2c1;font:10px Consolas")
        root.addWidget(self.subtitle)
        self.sections={
            "major":_EpicenterSection("EPICENTRO MAIOR • CONTEXTO AOI"),
            "minor":_EpicenterSection("EPICENTRO MENOR • FOCO AOI"),
        }
        for section in self.sections.values():
            root.addWidget(section)
        self.footer=QLabel(
            "CNN v2 utiliza entradas RGB da área completa e crop central; "
            "estes epicentros são inspeção pixel-a-pixel auxiliar."
        )
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet("color:#93a7b7;font:9px Consolas")
        root.addWidget(self.footer)
        self._last_payload=None
        self._last_analysis=None
        self._columns=0
        self._reflow(900)

    @staticmethod
    def columns_for_width(width: int) -> int:
        return 3 if width>=830 else 2 if width>=560 else 1

    def _reflow(self,width):
        columns=self.columns_for_width(int(width))
        if columns==self._columns:
            return
        for item in self.sections.values():
            item.reflow(columns)
        # Parent lane scroll is vertically scrollable: avoid clipping on 1366×768.
        self.setMinimumHeight({3:485,2:820,1:1190}[columns])
        self._columns=columns

    def update_data(self, detail:dict, analysis:dict|None=None):
        self._last_analysis=analysis
        active=bool((detail or {}).get("cnn_v2_active"))
        route=str((detail or {}).get("recognition_route","") or "")
        self.heading.setText("PAINEL NEURAL EXPLICÁVEL • " + (
            "CNN FALTANDO v2" if active else
            "KNN • VISUALIZAÇÃO DOS PIXELS" if route=="KNOWN_KNN"
            else "ENTRADA AOI"
        ))
        self.subtitle.setText(
            "CNN executou nesta luz • mapas de pixel NÃO demonstram atenção neural."
            if active else
            "CNN não executou nesta luz • memória KNN ou especialista distinto."
        )

    def set_visual_payload(self,payload:dict|None):
        self._last_payload=payload
        result=epicenter_evidence(payload)
        for key in EPICENTERS:
            self.sections[key].render_evidence(result.get(key))

    def clear_data(self):
        self._last_payload=None
        self._last_analysis=None
        self.heading.setText("PAINEL NEURAL EXPLICÁVEL")
        self.subtitle.setText("Aguardando dados SIDE/TOP/MID")
        for section in self.sections.values():
            section.render_evidence(None)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self._reflow(event.size().width())
