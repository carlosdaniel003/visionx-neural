"""Painel visual de "Treinamento" / validação histórica não mutante.

Executa validação em PROCESSO SEPARADO com prioridade baixa para não
congelar a interface nem tocar nos motores ao vivo. Nenhuma chamada de
0/1, DatasetManager, treinamento de pesos ou escrita de memória.
"""
from __future__ import annotations

from html import escape
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Qt
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QSizePolicy, QVBoxLayout, QApplication, QMessageBox,
    QCheckBox,
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
QLabel#archiveTrainingLegendOK {
  color: #4ade80; background: #14271a;
  border: 1px solid #285c38; border-radius: 5px;
  padding: 4px 10px; font-size: 11px; font-weight: 900;
}
QLabel#archiveTrainingLegendNG {
  color: #ff7777; background: #2b171b;
  border: 1px solid #78343b; border-radius: 5px;
  padding: 4px 10px; font-size: 11px; font-weight: 900;
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
QPushButton#archiveTrainingSimilarity {
  min-height: 38px; font-weight: 800; font-size: 12px;
  border: 1px solid #a78623; border-radius: 7px;
  background-color: #252113; color: #f5d067; padding: 7px 12px;
}
QPushButton#archiveTrainingSimilarity:hover {
  background-color: #36301b;
}
QPushButton#archiveTrainingSimilarity:disabled {
  background-color: #222222; color: #777777; border-color: #444444;
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
        color = "#4ade80" if group == "OK" else "#ff7777"
        segments.append(
            f'<span style="color:{color};font-weight:900;">'
            f'{group} {num}/{total} ({pct})</span>'
        )
    misses = int(info.get("incorrect") or 0)
    coverage = int(info.get("no_coverage") or 0)
    invalid = int(info.get("invalid") or 0)
    skipped = int(info.get("out_of_scope") or 0)
    tail = f" • Erros {misses} • Sem cobertura {coverage} • Inválidos {invalid}"
    if skipped:
        tail += f" • Fora da CNN {skipped}"
    return (
        f'<span style="color:#e8e8e8;font-weight:800;">{label}</span>'
        f' &nbsp;|&nbsp; {" &nbsp;•&nbsp; ".join(segments)}'
        f'<span style="color:#aaaaaa;">{escape(tail)}</span>'
    )


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
        self._run_type = "exact"
        self._cancel_requested = False
        self._cancel_path: Path | None = None
        self._count = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        self._passes = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        self._errors = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        self._by_label = self._empty_label_counts()
        self._build()
        panel.destroyed.connect(self.shutdown)

    @staticmethod
    def _empty_label_counts():
        return {
            model: {label: {"passed": 0, "checked": 0}
                    for label in ("OK", "NG")}
            for model in ("KNN_MEMORIA", "CNN_FALTANDO_V2")
        }

    def _live_metric_html(self, model: str) -> str:
        name = "KNN MEMÓRIA" if model == "KNN_MEMORIA" else "CNN FALTANDO V2"
        ok = self._by_label[model]["OK"]
        ng = self._by_label[model]["NG"]
        return (
            f'<span style="color:#e8e8e8;font-weight:800;">{name}</span>'
            f' &nbsp;|&nbsp; <span style="color:#4ade80;font-weight:900;">'
            f'OK {ok["passed"]}/{ok["checked"]}</span>'
            f' &nbsp;•&nbsp; <span style="color:#ff7777;font-weight:900;">'
            f'NG {ng["passed"]}/{ng["checked"]}</span>'
            f' &nbsp;•&nbsp; <span style="color:#aaaaaa;">'
            f'{self._errors[model]} pendências/erros</span>'
        )

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
        self.similarity_button = QPushButton("Testar KNN por similaridade")
        self.similarity_button.setObjectName("archiveTrainingSimilarity")
        self.similarity_button.setToolTip(
            "Compara pares AOI gabarito/captura por diferenças visuais, "
            "usando caixa maior e epicentro menor sem filtrar por datas."
        )
        container.addLayout(title_row)
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.similarity_button)
        action_row.addWidget(self.cancel_button)
        container.addLayout(action_row)
        self.geometry_experiment_box = QCheckBox(
            "Comparar geometria de bordas (experimental, somente acervo)"
        )
        self.geometry_experiment_box.setToolTip(
            "Executa duas KNNs offline sobre as MESMAS imagens: "
            "visual v3 intacta e candidato geométrico A/B. "
            "Não habilita produção nem muda memória."
        )
        self.geometry_experiment_box.setChecked(False)
        container.addWidget(self.geometry_experiment_box)

        hint = QLabel(
            "Memória: correspondência exata KNN + CNN. Similaridade: KNN "
            "por diferença entre gabarito e captura, com caixa maior e "
            "epicentro menor. Usa somente acervo OK/NG, sem filtrar datas "
            "e sem comandar o XP."
        )
        hint.setObjectName("archiveTrainingHint")
        hint.setWordWrap(True)
        container.addWidget(hint)

        legend = QHBoxLayout()
        legend.setSpacing(7)
        ok_legend = QLabel("● OK")
        ok_legend.setObjectName("archiveTrainingLegendOK")
        ng_legend = QLabel("● NG")
        ng_legend.setObjectName("archiveTrainingLegendNG")
        legend.addWidget(ok_legend)
        legend.addWidget(ng_legend)
        legend.addStretch(1)
        container.addLayout(legend)

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
        self.similarity_metric = QLabel(
            "KNN SIMILARIDADE | ainda não avaliada (somente diagnóstico)"
        )
        self.shadow_candidate_metric = QLabel(
            "KNN CANDIDATO ACERVO | ainda não avaliado (revisão habilitada)"
        )
        for widget in (self.knn_metric, self.cnn_metric, self.similarity_metric,
                       self.shadow_candidate_metric):
            widget.setObjectName("archiveTrainingMetric")
            widget.setTextFormat(Qt.TextFormat.RichText)
            widget.setWordWrap(True)
            widget.setMinimumWidth(0)
        metrics.addWidget(self.knn_metric, 0, 0)
        metrics.addWidget(self.cnn_metric, 1, 0)
        metrics.addWidget(self.similarity_metric, 2, 0)
        metrics.addWidget(self.shadow_candidate_metric, 3, 0)
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
        self.similarity_button.clicked.connect(self.start_similarity)

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
        self._run_type = "exact"
        self._stdout = ""
        self._stderr = ""
        self._last_finished = None
        self._cancel_requested = False
        self._count = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        self._passes = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        self._errors = {"KNN_MEMORIA": 0, "CNN_FALTANDO_V2": 0}
        self._by_label = self._empty_label_counts()
        reports = self.root / "reports" / "startup_regression"
        reports.mkdir(parents=True, exist_ok=True)
        self._cancel_path = reports / (
            "training_validation_" + uuid4().hex + ".cancel"
        )
        self._cancel_path.unlink(missing_ok=True)
        self.start_button.setEnabled(False)
        self.similarity_button.setEnabled(False)
        self.geometry_experiment_box.setEnabled(False)
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
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(self.root))
        proc.setProgram(sys.executable)
        proc.setArguments([
            "-u", "-m",
            "src.services.startup_regression.interactive_training_validation",
            "--root", str(self.root),
            "--cancel-file", str(self._cancel_path),
        ] + (
            ["--geometry-experiment"]
            if self.geometry_experiment_box.isChecked() else []
        ))
        proc.readyReadStandardOutput.connect(self._read_stdout)
        proc.readyReadStandardError.connect(self._read_stderr)
        proc.finished.connect(self._finished)
        proc.errorOccurred.connect(self._process_error)
        proc.start()

    def start_similarity(self):
        """Segundo teste: somente avaliação de vizinhos KNN, sem aprendizado."""
        if self._active():
            return
        self._run_type = "similarity"
        self._stdout = ""
        self._stderr = ""
        self._last_finished = None
        self._cancel_requested = False
        reports = self.root / "reports" / "startup_regression"
        reports.mkdir(parents=True, exist_ok=True)
        self._cancel_path = reports / (
            "knn_similarity_" + uuid4().hex + ".cancel"
        )
        self._cancel_path.unlink(missing_ok=True)
        self.start_button.setEnabled(False)
        self.similarity_button.setEnabled(False)
        self.geometry_experiment_box.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText("KNN similaridade: inventariando OK e NG...")
        self.similarity_metric.setText(
            "KNN SIMILARIDADE | processando imagens, sem alterar memória"
        )
        self.shadow_candidate_metric.setText(
            "KNN CANDIDATO ACERVO | avaliando sugestões e revisões"
        )
        self.details.setText(
            "Comparação visual gabarito × captura: imagem global, região "
            "maior e epicentro menor quando marcado pela AOI. "
            "Sem seleção por data. Aguarde relatório por classe e luz."
        )
        self.progress_bar.setRange(0, 0)

        proc = QProcess(self)
        self.process = proc
        env = QProcessEnvironment.systemEnvironment()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.insert(key, "1")
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(self.root))
        proc.setProgram(sys.executable)
        proc.setArguments([
            "-u", "-m",
            "src.services.startup_regression.knn_visual_panel_worker",
            "--root", str(self.root),
            "--cancel-file", str(self._cancel_path),
        ])
        proc.readyReadStandardOutput.connect(self._read_stdout)
        proc.readyReadStandardError.connect(self._read_stderr)
        proc.finished.connect(self._finished)
        proc.errorOccurred.connect(self._process_error)
        proc.start()

    def _handle_similarity(self, packet: dict):
        event = packet.get("event")
        if event == "phase":
            self.status.setText(str(packet.get("phase") or "Comparando KNN"))
        elif event == "started":
            total = max(1, int(packet.get("total") or 0))
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(f"%p% • %v/{total} PNGs")
        elif event == "progress":
            done = int(packet.get("done") or 0)
            total = max(1, int(packet.get("total") or 1))
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(min(done, total))
            if not self._cancel_requested:
                self.status.setText(
                    f"Extraindo assinaturas KNN: {done}/{total} • "
                    + escape(Path(str(packet.get("path") or "")).name)
                )
        elif event == "finished":
            self._last_finished = packet
        elif event == "error":
            self.status.setText(
                "Falha na similaridade: "
                + str(packet.get("message") or "")[:350]
            )

    @staticmethod
    def _similarity_line(title: str, metrics: dict) -> str:
        by = metrics.get("by_label", {})
        def label_stat(label):
            row = by.get(label) or {}
            total = int(row.get("total") or 0)
            correct = int(row.get("correct") or 0)
            pct = f"{100 * correct / total:.1f}%" if total else "N/D"
            return f"{label} {correct}/{total} ({pct})"
        return (
            f"<b>{escape(title)}</b> | "
            f'<span style="color:#4ade80">{label_stat("OK")}</span> • '
            f'<span style="color:#ff7777">{label_stat("NG")}</span> • '
            f"NG→OK {int(metrics.get('NG_as_OK') or 0)}"
        )

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
        if self._run_type == "similarity":
            self._handle_similarity(packet)
            return
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
                label = str(packet.get("expected") or "").upper()
                if label in ("OK", "NG"):
                    values = self._by_label[model][label]
                    values["checked"] += 1
                    if packet.get("status") == "PASSOU":
                        values["passed"] += 1
                target.setText(self._live_metric_html(model))
            name = Path(str(packet.get("path", ""))).name
            display_model = "KNN" if model == "KNN_MEMORIA" else "CNN"
            if not self._cancel_requested:
                label = str(packet.get("expected") or "").upper()
                badge = (
                    '<span style="color:#4ade80;font-weight:900;">OK</span>'
                    if label == "OK" else
                    '<span style="color:#ff7777;font-weight:900;">NG</span>'
                    if label == "NG" else ""
                )
                self.status.setText(
                    f'{done}/{total} • {display_model} • {badge} • {escape(name)}'
                )
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
            self.similarity_button.setEnabled(True)
            self.geometry_experiment_box.setEnabled(True)
            self.cancel_button.setEnabled(False)
            if not self._cancel_requested and self._last_finished is None:
                self.status.setText("Não foi possível iniciar a validação.")

    def _finished(self, code: int, _status):
        self._read_stdout()
        self._read_stderr()
        self.start_button.setEnabled(True)
        self.similarity_button.setEnabled(True)
        self.geometry_experiment_box.setEnabled(True)
        self.cancel_button.setEnabled(False)
        result = self._last_finished
        if self._run_type == "similarity":
            self._finish_similarity(code, result)
            if self._cancel_path is not None:
                self._cancel_path.unlink(missing_ok=True)
            if self.process is not None:
                self.process.deleteLater()
            self.process = None
            return
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
        if self.process is not None:
            self.process.deleteLater()
        self.process = None

    def _finish_similarity(self, code: int, packet: dict | None) -> None:
        if packet is not None:
            metrics = packet.get("results") or {}
            self.similarity_metric.setText(
                self._similarity_line("KNN POR SIMILARIDADE", metrics)
            )
            candidate = packet.get("shadow_candidate") or {}
            ok_c = (candidate.get("by_label") or {}).get("OK") or {}
            ng_c = (candidate.get("by_label") or {}).get("NG") or {}
            precise = candidate.get("conditional_precision")
            precision_text = (
                f"{100 * float(precise):.2f}%"
                if precise is not None else "N/D"
            )
            coverage = candidate.get("coverage")
            coverage_text = (
                f"{100 * float(coverage):.1f}%"
                if coverage is not None else "N/D"
            )
            self.shadow_candidate_metric.setText(
                "<b>KNN CANDIDATO ACERVO • META 97%</b> | "
                f'<span style="color:#4ade80">OK '
                f'{ok_c.get("SUGERE_OK", 0)}/{ok_c.get("total", 0)}</span> • '
                f'<span style="color:#ff7777">NG '
                f'{ng_c.get("SUGERE_NG", 0)}/{ng_c.get("total", 0)}</span> • '
                f'NG→OK {candidate.get("NG_suggested_OK", 0)} • '
                f'OK→NG {candidate.get("OK_suggested_NG", 0)} • '
                f'Revisão {candidate.get("reviews", 0)} • '
                f'Precisão nas sugestões {precision_text} '
                f'({candidate.get("correct_suggestions", 0)}/'
                f'{candidate.get("suggestions", 0)}) • '
                f'Cobertura {coverage_text}'
            )
            cohorts = packet.get("by_capture_cohort") or {}
            lights = packet.get("by_lighting") or {}
            lines = [
                "Candidato: meta 97% APENAS ENTRE SUGESTÕES; "
                "revisões ficam fora da precisão mas dentro da cobertura.",
                "Critério: diferença visual gabarito/captura + epicentros",
                "JSON: " + str(packet.get("json_path") or "-"),
                "TXT: " + str(packet.get("txt_path") or "-"),
                "Por grupo (apenas estimativa quando sem manifesto):",
            ]
            for key, label in (
                ("side_only", "Somente SIDE"),
                ("three_lights", "SIDE + TOP + MID"),
                ("partial_other", "Outros conjuntos"),
            ):
                cohort = cohorts.get(key) or {}
                m = cohort.get("results") or {}
                lines.append(
                    f"{label}: "
                    f"{m.get('correct', 0)}/{m.get('total', 0)} "
                    f"({cohort.get('capture_groups', 0)} grupos)"
                )
            focus = packet.get("focus_detection") or {}
            lines.append(
                "Epicentros AOI: caixa maior "
                f"{focus.get('outer_detected', 0)}/"
                f"{focus.get('total', 0)} • caixa menor "
                f"{focus.get('inner_detected', 0)}/"
                f"{focus.get('total', 0)}"
            )
            experiment = packet.get("geometry_experiment")
            if experiment:
                control = candidate
                alt = experiment["shadow_candidate"]
                base = metrics
                new = experiment["results"]
                lines.extend([
                    "GEOMETRIA A/B (SOMENTE EXPERIMENTO, NÃO ATIVA):",
                    f"A visual: {base.get('correct', 0)}/{base.get('total', 0)} "
                    f"NG→OK {base.get('NG_as_OK', 0)} | conservadora "
                    f"{control.get('correct_suggestions', 0)}/"
                    f"{control.get('suggestions', 0)}",
                    f"B geometria: {new.get('correct', 0)}/{new.get('total', 0)} "
                    f"NG→OK {new.get('NG_as_OK', 0)} | conservadora "
                    f"{alt.get('correct_suggestions', 0)}/"
                    f"{alt.get('suggestions', 0)} | "
                    f"cobertura {alt.get('coverage', 0) * 100:.1f}% | "
                    f"NG→OK candidato {alt.get('NG_suggested_OK', 0)}",
                    "A/B no mesmo acervo não comprova generalização.",
                ])
            lines.append(
                "Por luz: " + " | ".join(
                    f"{light}: {m.get('correct', 0)}/{m.get('total', 0)}"
                    for light, m in sorted(lights.items())
                )
            )
            lines.append(
                "Grupos não verificados são pistas de nome, "
                "não evidência física de um mesmo evento."
            )
            self.details.setText("\n".join(lines))
            if packet.get("canceled") or self._cancel_requested:
                self.status.setText(
                    "Similaridade cancelada • relatório parcial salvo."
                )
            elif candidate.get("target_97pct_met"):
                self.status.setText(
                    "Candidato KNN atingiu meta diagnóstica de 97% "
                    "ENTRE SUGESTÕES do acervo, com zero NG→OK observado. "
                    "Não libera produção; confira a cobertura."
                )
            else:
                self.status.setText(
                    "Candidato KNN: meta diagnóstica de 97% não atingida "
                    "ou sem cobertura; revisar erros, NG→OK e abstenções."
                )
        elif self._cancel_requested:
            self.status.setText(
                "Similaridade cancelada; relatório parcial pode não existir."
            )
        else:
            self.status.setText(
                f"Similaridade terminou com erro (código {code}). "
                + self._stderr[-350:]
            )

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
