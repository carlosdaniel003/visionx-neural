"""Card de especialistas atuais: CNN de ausência e KNN humano verificado.

O card é telemetria pura: não envia rótulos, não lê KNN nem modifica scores.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QSizePolicy

from src.ui.neural_telemetry_model import cnn_panel_text, memory_panel_text, neural_summary


class NeuralSpecialistWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("neuralSpecialistCard")
        self.setMinimumWidth(320)
        self.setMinimumHeight(265)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(
            "QFrame#neuralSpecialistCard {background:#111820;"
            "border:1px solid #2e3a44; border-radius:8px;}"
            "QLabel {border:none; background:transparent; color:#c9d1d9;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(5)
        self.title = QLabel("CNN FALTANDO v2")
        self.title.setStyleSheet("font-size:13px; font-weight:800; color:#f5c518;")
        layout.addWidget(self.title)
        self.lines = []
        for _ in range(10):
            label = QLabel()
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            label.setStyleSheet("font-family:Consolas; font-size:10px;")
            layout.addWidget(label)
            self.lines.append(label)
        layout.addStretch(1)

    def update_data(self, detail: dict, analysis: dict | None = None):
        payload = analysis if isinstance(analysis, dict) else {"detail": detail}
        view = cnn_panel_text(payload)
        self.title.setText(view["header"])
        for label, line in zip(self.lines, list(view["lines"]) + [""] * len(self.lines)):
            label.setText(line)
            label.setVisible(bool(line))
        self.setToolTip("\n".join(view["lines"]))
        self.update()


class VerifiedMemorySpecialistWidget(NeuralSpecialistWidget):
    def update_data(self, detail: dict, analysis: dict | None = None):
        payload = analysis if isinstance(analysis, dict) else {"detail": detail}
        label, description = memory_panel_text(payload)
        self.title.setText(label)
        items = [
            description,
            "A memória KNN reconhece apenas pares previamente confirmados",
            "por operador; não mede proximidade nesta recuperação exata.",
        ]
        for widget, text in zip(self.lines, items + [""] * len(self.lines)):
            widget.setText(text)
            widget.setVisible(bool(text))
        self.setToolTip("\n".join(items))
        self.update()
