"""Painel visual de "Treinamento" / validação histórica não mutante.

Executa validação em PROCESSO SEPARADO com prioridade baixa para não
congelar a interface nem tocar nos motores ao vivo. Nenhuma chamada de
0/1, DatasetManager, treinamento de pesos ou escrita de memória.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from uuid import uuid4

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Qt
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QSizePolicy, QVBoxLayout,
)

from src.config.settings import BASE_DIR


STYLES = """
QFrame#archiveTrainingFrame {
  background: #101010; border: 1px solid #373737;
  border-radius: 8px;
}
QLabel#archiveTrainingTitle {
  font-size: 12px; font-weight: 900; color: #f5c518;
}
QLabel#archiveTrainingHint {
  font-size: 11px; font-weight: 500; color: #999999;
}
QLabel#archiveTrainingStatus {
  font-size: 12px; font-weight: 800; color: #e5e5e5;
}
QLabel#archiveTrainingMetric {
  font-size: 12px; font-weight: 700; color: #cfcfcf;
  background-color: #171717; border: 1px solid #303030;
  border-radius: 6px; padding: 8px;
}
QLabel#archiveTrainingDetails {
  font-size: 11px; font-weight: 600; color: #a9a9a9;
}
QPushButton#archiveTrainingStart {
  min-height: 38px; font-weight: 900; font-size: 12px;
  border: 1px solid #f5c518; border-radius: 7px;
  background-color: #f5c518; color: #111111; padding: 7px 16px;
}
QPushButton#archiveTrainingStart:hover {
  background-color: #ffe070;
}
QPushButton#archiveTrainingStart:disabled {
  background-color: #333333; color: #777777; border-color: #484848;
}
QPushButton#archiveTrainingCancel {
  min-height: 38px; font-weight: 800; font-size: 12px;
  border: 1px solid #cc5858; border-radius: 7px;
  background-color: #301818; color: #ffd0d0; padding: 7px 16px;
}
QPushButton#archiveTrainingCancel:disabled {
  background-color: #202020; color: #666666; border-color: #444444;
}
QProgressBar#archiveTrainingProgress {
  background-color: #242424; border: 1px solid #444444;
  color: #f5c518; border-radius: 6px; font-size: 11px;
  font-weight: 800; text-align: center; min-height: 18px;
}
QProgressBar#archiveTrainingProgress::chunk {
  background-color: #f5c518; border-radius: 5px;
}
"""


def format_model_line(name: str, info: dict) -> str:
    label = "KNN MEMÓRIA" if name == "KNN_MEMORIA" else "CNN FALTANDO V2"
    segments = []
    for group in ("OK", "NG"):
        metrics = (info.get("by_label") or {}).get(group) or {}
        num = int(metrics.get("passed") or 0)
        total = int(metrics.get("total") or 0)
        rate = metrics.get("accuracy")
        pct = f"{100*rate:.1f}%" if isinstance(rate, (int, float)) else "N/D"
        segments.append(f"{group} {num}/{total} ({pct})")
    misses = int(info.get("incorrect") or 0)
    coverage = int(info.get("no_coverage") or 0)
    invalid = int(info.get("invalid") or 0)
    skipped = int(info.get("out_of_scope") or 0)
    tail = f" • Erros {misses} • Sem cobertura {coverage} • Inválidos {invalid}"
    if skipped:
        tail += f" • Fora da CNN {skipped}"
    return f"{label}  |  {'  •  '.join(segments)}{tail}"


class ArchiveTrainingPanel(QObject):
    """Um único processo por instância, com término controlado pelo botão."""

    def __init__(self, panel, *, root: Path | None = None):
        super().__init__(panel)
        self.panel = panel
        self.root = Path(root or BASE_DIR).resolve()
        self.process: QProcess | None = None
        self._stdout = ""
        self._stderr = ""
        self._last_finished = None
        self._cancel_requested = False
        self._cancel_path: Path | None = None
        self._count = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        self._passes = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        self._errors = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        self._build()
        panel.destroyed.connect(self.shutdown)

    def _build(self):
        frame = QFrame()
        frame.setObjectName("archiveTrainingFrame")
        frame.setStyleSheet(STYLES)
        frame.setMinimumWidth(0)
        frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        container = QVBoxLayout(frame)
        container.setContentsMargins(12, 10, 12, 12)
        container.setSpacing(7)

        title_row = QHBoxLayout()
        title = QLabel("TREINAMENTO • VALIDAÇÃO DO ACERVO")
        title.setObjectName("archiveTrainingTitle")
        title_row.addWidget(title, stretch=1)
        self.start_button = QPushButton("Iniciar treinamento")
        self.start_button.setObjectName("archiveTrainingStart")
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.setObjectName("archiveTrainingCancel")
        self.cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button.setEnabled(False)
        title_row.addWidget(self.start_button)
        title_row.addWidget(self.cancel_button)
        container.addLayout(title_row)

        hint = QLabel(
            "Executa somente replay KNN + CNN FALTANDO V2 de ok_archive/ng_archive. "
            "Não treina pesos, não altera memória e não interrompe a inspeção."
        )
        hint.setObjectName("archiveTrainingHint")
        hint.setWordWrap(True)
        container.addWidget(hint)

        self.status = QLabel("Pronto para validar o histórico OK e NG.")
        self.status.setObjectName("archiveTrainingStatus")
        self.status.setWordWrap(True)
        container.addWidget(self.status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("archiveTrainingProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0% • 0/0 verificações")
        container.addWidget(self.progress_bar)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(8)
        metrics.setVerticalSpacing(6)
        self.knn_metric = QLabel("KNN MEMÓRIA | ainda não avaliado")
        self.cnn_metric = QLabel("CNN FALTANDO V2 | ainda não avaliada")
        for widget in (self.knn_metric, self.cnn_metric):
            widget.setObjectName("archiveTrainingMetric")
            widget.setWordWrap(True)
            widget.setMinimumWidth(0)
        metrics.addWidget(self.knn_metric, 0, 0)
        metrics.addWidget(self.cnn_metric, 1, 0)
        container.addLayout(metrics)

        self.details = QLabel("Relatórios: reports/startup_regression/")
        self.details.setObjectName("archiveTrainingDetails")
        self.details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.details.setWordWrap(True)
        container.addWidget(self.details)

        self.start_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)

        layout = self.panel.content_layout
        target = getattr(self.panel, "network_debug_frame", None)
        position = layout.indexOf(target) if target is not None else -1
        if position < 0:
            layout.addWidget(frame)
        else:
            layout.insertWidget(position, frame)
        self.panel.archive_training_frame = frame
        self.panel.btn_training = self.start_button
        self.panel.btn_cancel_training = self.cancel_button
        self.panel.training_progress_bar = self.progress_bar

    def _active(self) -> bool:
        return (self.process is not None
                and self.process.state() != QProcess.ProcessState.NotRunning)

    def start(self):
        if self._active():
            return
        self._stdout = ""
        self._stderr = ""
        self._last_finished = None
        self._cancel_requested = False
        self._count = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        self._passes = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        self._errors = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        reports = self.root / "reports" / "startup_regression"
        reports.mkdir(parents=True, exist_ok=True)
        self._cancel_path = reports / (
            "training_validation_" + uuid4().hex + ".cancel"
        )
        self._cancel_path.unlink(missing_ok=True)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText("Inventariando imagens OK/NG...")
        self.details.setText("Processo de diagnóstico isolado, prioridade baixa.")
        self.knn_metric.setText("KNN MEMÓRIA | aguardando avaliação")
        self.cnn_metric.setText("CNN FALTANDO V2 | aguardando avaliação")
        self.progress_bar.setRange(0, 0)  # inventário inicial indeterminado

        proc = QProcess(self)
        self.process = proc
        env = QProcessEnvironment.systemEnvironment()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.insert(key, "1")
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(self.root))
        proc.setProgram(sys.executable)
        proc.setArguments([
            "-u", "-m",
            "src.services.startup_regression.interactive_training_validation",
            "--root", str(self.root),
            "--cancel-file", str(self._cancel_path),
        ])
        proc.readyReadStandardOutput.connect(self._read_stdout)
        proc.readyReadStandardError.connect(self._read_stderr)
        proc.finished.connect(self._finished)
        proc.errorOccurred.connect(self._process_error)
        proc.start()

    def _read_stdout(self):
        if self.process is None:
            return
        self._stdout += bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while "\n" in self._stdout:
            line, self._stdout = self._stdout.split("\n", 1)
            try:
                packet = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(packet, dict):
                self._handle(packet)

    def _read_stderr(self):
        if self.process is None:
            return
        self._stderr = (
            self._stderr + bytes(self.process.readAllStandardError()).decode(
                "utf-8", errors="replace"
            )
        )[-3000:]

    def _handle(self, packet: dict):
        event = packet.get("event")
        if event == "phase":
            self.status.setText(str(packet.get("phase") or "Preparando"))
        elif event == "started":
            total = max(1, int(packet.get("total") or 0))
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(
                f"%p% • %v/{total} verificações"
            )
            self.status.setText(
                f"{packet.get('images')} imagens encontradas • "
                f"{total} avaliações elegíveis."
            )
        elif event == "progress":
            done, total = int(packet.get("done", 0)), max(
                1, int(packet.get("total", 1))
            )
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(total, done))
            model = packet.get("model")
            if model in self._count:
                self._count[model] += 1
                if packet.get("status") == "PASSOU":
                    self._passes[model] += 1
                else:
                    self._errors[model] += 1
                target = (self.knn_metric if model == "KNN_MEMORIA"
                          else self.cnn_metric)
                target.setText(
                    f"{'KNN MEMÓRIA' if model == 'KNN_MEMORIA' else 'CNN FALTANDO V2'}"
                    f" | {self._passes[model]}/{self._count[model]} corretas"
                    f" • {self._errors[model]} pendências/erros"
                )
            name = Path(str(packet.get("path", ""))).name
            display_model = "KNN" if model == "KNN_MEMORIA" else "CNN"
            if not self._cancel_requested:
                self.status.setText(f"{done}/{total} • {display_model} • {name}")
        elif event == "finished":
            self._last_finished = packet
        elif event == "error":
            self.status.setText(
                "Falha no diagnóstico: " + str(packet.get("message", ""))[:350]
            )

    def cancel(self):
        if not self._active() or self._cancel_requested:
            return
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        self.status.setText("Cancelamento solicitado • encerrando análise...")
        if self._cancel_path is not None:
            try:
                self._cancel_path.write_text("cancel\n", encoding="ascii")
            except OSError:
                pass
        # Após timeout também cancela se OCR/PyTorch estiver preso.
        active_process = self.process
        QTimer.singleShot(1800, lambda: self._force_stop(active_process))

    def _force_stop(self, active_process: QProcess | None):
        if (active_process is None or self.process is not active_process
                or active_process.state() == QProcess.ProcessState.NotRunning):
            return
        active_process.terminate()
        QTimer.singleShot(900, lambda: self._kill(active_process))

    def _kill(self, active_process: QProcess | None):
        if (active_process is not None
                and self.process is active_process
                and active_process.state() != QProcess.ProcessState.NotRunning):
            active_process.kill()

    def _process_error(self, _error):
        if self.process is None:
            return
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            if not self._cancel_requested and self._last_finished is None:
                self.status.setText("Não foi possível iniciar a validação.")

    def _finished(self, code: int, _status):
        self._read_stdout()
        self._read_stderr()
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        result = self._last_finished
        if result is not None:
            for model, target in (("KNN_MEMORIA", self.knn_metric),
                                  ("CNN_FALTANDO_V2", self.cnn_metric)):
                info = (result.get("models") or {}).get(model, {})
                if info:
                    target.setText(format_model_line(model, info))
            self.details.setText(
                "JSON: " + str(result.get("json_path") or "-") + "\n"
                "TXT: " + str(result.get("txt_path") or "-")
            )
            if result.get("canceled") or self._cancel_requested:
                self.status.setText("Cancelado • relatório PARCIAL salvo.")
            else:
                met = all(
                    (result.get("models", {}).get(model, {})
                     .get("target_98pct_per_label") is True)
                    for model in ("KNN_MEMORIA", "CNN_FALTANDO_V2")
                )
                self.status.setText(
                    "Validação completa • meta OK/NG >=98% atingida."
                    if met else
                    "Validação completa • algumas imagens NÃO passaram."
                )
        elif self._cancel_requested:
            self.status.setText(
                "Validação cancelada. Processo encerrado; relatório parcial "
                "pode não existir se OCR/inferência exigiram término forçado."
            )
        else:
            self.status.setText(
                "Validação encerrada com erro "
                f"(código {code}). Veja o terminal/log. " +
                self._stderr[-400:]
            )
        if self._cancel_path is not None:
            self._cancel_path.unlink(missing_ok=True)
        self.process = None

    def shutdown(self, *_args):
        if self.process is not None and self._active():
            self.process.kill()


def install_training_validation_panel(panel, *, root=None):
    existing = getattr(panel, "_archive_training_panel", None)
    if existing is not None:
        return existing
    panel._archive_training_panel = ArchiveTrainingPanel(panel, root=root)
    return panel._archive_training_panel


__all__ = ["install_training_validation_panel", "ArchiveTrainingPanel",
           "format_model_line"]
