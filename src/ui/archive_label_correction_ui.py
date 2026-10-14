"""Ferramenta administrativa do ODIN: modo manutenção sem recepção XP."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
    QMessageBox, QPushButton, QTextEdit, QVBoxLayout,
)

from src.config.settings import BASE_DIR
from src.services.archive_label_correction import (
    apply_correction, prepare_correction,
)
from src.services.faltando_cnn_recovery_preflight import (
    latest_completed_correction, validate_recovery_inputs,
)
from src.services.faltando_cnn_candidate_archive_validation import (
    candidate_preflight,
)
from src.services.faltando_cnn_regression_triage import triage_report


def wait_for_parent_to_exit(pid: int) -> None:
    """Garante que watcher/filas do ODIN antigo terminaram antes do maintenance."""
    if pid <= 0:
        return
    if os.name == "nt":
        import ctypes
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x00100000, 0, pid)  # SYNCHRONIZE
        if handle:
            try:
                result = kernel.WaitForSingleObject(handle, 60000)
                if result != 0:  # WAIT_OBJECT_0
                    raise RuntimeError("ODIN principal não encerrou em 60s")
            finally:
                kernel.CloseHandle(ctypes.c_void_p(handle))
        return

    for _ in range(600):
        try:
            os.kill(pid, 0)  # Unix: verifica existência, não envia sinal
        except ProcessLookupError:
            return
        time.sleep(0.1)
    raise RuntimeError("ODIN principal não encerrou em 60s")


class ArchiveLabelCorrectionDialog(QDialog):
    def __init__(self, root: Path = BASE_DIR):
        super().__init__()
        self.root = Path(root).resolve()
        self.files: list[str] = []
        self.plan = None
        self.train_process = None
        self._active_workflow = None
        self.setWindowTitle("ODIN • Corrigir julgamento humano")
        self.setMinimumSize(760, 520)

        layout = QVBoxLayout(self)
        heading = QLabel("CORREÇÃO ADMINISTRATIVA • OK → NG")
        heading.setStyleSheet("font-size: 17px; font-weight: 800;")
        layout.addWidget(heading)
        summary = QLabel(
            "Somente com a inspeção encerrada e backup feito. "
            "Selecione exatamente 9 capturas de public/ok_archive. "
            "A ferramenta exige vínculo visual e 3 event_ids completos, "
            "corrige KNN/dataset/journal e transfere as capturas para ng_archive. "
            "Não retreina a CNN."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        buttons = QHBoxLayout()
        self.select_button = QPushButton("Selecionar 9 capturas OK")
        self.preview_button = QPushButton("Conferir vínculos e alterações")
        self.commit_button = QPushButton("Confirmar retificação para NG")
        self.commit_button.setEnabled(False)
        buttons.addWidget(self.select_button)
        buttons.addWidget(self.preview_button)
        buttons.addWidget(self.commit_button)
        layout.addLayout(buttons)

        recovery_row = QHBoxLayout()
        self.recovery_button = QPushButton(
            "Comparar CNN FALTANDO V2 — 5 e 7 épocas"
        )
        self.light_weight_button = QPushButton(
            "Comparar CNN — perda por iluminação (0,15 / 0,50 / 1,00)"
        )
        self.recovery_cancel = QPushButton("Cancelar experimento")
        self.recovery_cancel.setEnabled(False)
        recovery_row.addWidget(self.recovery_button)
        recovery_row.addWidget(self.light_weight_button)
        recovery_row.addWidget(self.recovery_cancel)
        layout.addLayout(recovery_row)

        validation_row = QHBoxLayout()
        self.validation_button = QPushButton(
            "Validar candidato CNN 0,50 contra acervo completo (sem ativar)"
        )
        self.triage_button = QPushButton(
            "Analisar regressões do último relatório (sem reprocessar)"
        )
        validation_row.addWidget(self.validation_button)
        validation_row.addWidget(self.triage_button)
        layout.addLayout(validation_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlainText(
            "Nenhum arquivo selecionado. Não há alterações pendentes."
        )
        layout.addWidget(self.log, stretch=1)

        foot = QLabel(
            "Em caso de divergência entre captura, event_id, dataset ou KNN, "
            "a correção é bloqueada. O snapshot de checkpoint da CNN não é modificado."
        )
        foot.setWordWrap(True)
        layout.addWidget(foot)

        self.select_button.clicked.connect(self.select_files)
        self.preview_button.clicked.connect(self.preview)
        self.commit_button.clicked.connect(self.commit)
        self.recovery_button.clicked.connect(self.start_recovery)
        self.light_weight_button.clicked.connect(self.start_light_weight_comparison)
        self.recovery_cancel.clicked.connect(self.cancel_recovery)
        self.validation_button.clicked.connect(self.start_candidate_validation)
        self.triage_button.clicked.connect(self.start_regression_triage)

    def start_regression_triage(self):
        """Classifica transições já avaliadas, sem imagem, GPU nem treinamento."""
        if self.train_process is not None:
            return
        folder = (
            self.root / "reports/startup_regression/cnn_candidate_validation"
        )
        reports = sorted(folder.glob("comparacao_acervo_cnn_peso050_*.json"))
        if not reports:
            QMessageBox.warning(
                self, "Sem relatório", "Execute primeiro a validação do acervo."
            )
            return
        try:
            path, report = triage_report(self.root, reports[-1])
        except Exception as exc:
            self.log.setPlainText(f"TRIAGEM BLOQUEADA:\n{exc}")
            QMessageBox.warning(self, "Falha na triagem", str(exc))
            return
        by_category = report["changes_by_category"]
        lines = [
            "TRIAGEM CNN — SEM TREINAMENTO OU ATIVAÇÃO",
            f"Resultado: {'REPROVADO' if report['candidate_rejected_by_archive_gate'] else 'SEM REGRESSÕES'}",
            f"Imagens avaliadas: {report['eligible_images']}",
            f"Novos OK→NG: {report['new_ok_false_ng']}",
            f"Novos NG→OK: {report['new_ng_false_ok']}",
            f"Novas revisões de imagens antes corretas: {report['new_reviews_from_previously_passed']}",
            f"Imagens que exigem conferência humana: {report['required_human_review_count']}",
            "", "Regressões por categoria:",
        ]
        for category, transitions in sorted(by_category.items()):
            lines.append(f"{category}: " + ", ".join(
                f"{key}={count}" for key, count in sorted(transitions.items())
            ))
        lines += [
            "", "Nenhuma imagem entrou no treinamento.",
            "Confirme os rótulos OK DESLOCADO e os IDs físicos antes de",
            "considerar uma recuperação com exemplos difíceis.",
            "", f"Relatório: {path}",
        ]
        self.log.setPlainText("\n".join(lines))

    def start_candidate_validation(self):
        """Duas avaliações operacionais reais com checkpoints imutáveis."""
        if self.train_process is not None:
            return
        try:
            audit = candidate_preflight(self.root)
        except Exception as exc:
            self.log.setPlainText(
                f"VALIDAÇÃO DO CANDIDATO BLOQUEADA:\n{exc}\n"
                "Nenhuma inferência executada."
            )
            QMessageBox.warning(self, "Validação bloqueada", str(exc))
            return
        confirmation = QMessageBox.question(
            self, "Validar CNN 0,50 no acervo inteiro",
            "Este diagnóstico lê o acervo OK/NG inteiro DUAS VEZES "
            "(CNN original e candidata 0,50) e poderá demorar vários minutos.\n\n"
            "Os resultados incluem revisões obrigatórias da CNN operacional. "
            "Parte das imagens participou do treinamento: isto NÃO é "
            "validação independente de generalização.\n\n"
            "Nenhum peso será ativado, nenhuma memória será alterada, "
            "nenhum comando 0/1 será enviado. Prosseguir?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirmation != QMessageBox.StandardButton.Yes:
            return
        self.log.setPlainText(
            "VALIDAÇÃO CNN 0,50 — SOMENTE LEITURA, SEM ATIVAÇÃO\n"
            f"Checkpoint candidato SHA-256: {audit['candidate_sha256']}\n"
            "Executando checkpoint ORIGINAL e CANDIDATO, acervo idêntico.\n"
            "Relatório final em reports/startup_regression/"
            "cnn_candidate_validation/\n"
        )
        self._launch_worker(
            "src.services.faltando_cnn_candidate_archive_validation",
            ["--root", str(self.root)],
            workflow="candidate_validation",
        )

    def _launch_worker(self, module: str, args: list[str], *,
                       workflow: str):
        proc = QProcess(self)
        self.train_process = proc
        self._active_workflow = workflow
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setWorkingDirectory(str(self.root))
        proc.setProgram(sys.executable)
        # O Python Windows pode herdar saída CP1252 mesmo que a aplicação
        # exiba Unicode. ScreenMonitor imprime ✅ ao detectar Tesseract:
        # forçar UTF-8 apenas neste processo offline e nas saídas capturadas.
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        proc.setProcessEnvironment(environment)
        proc.setArguments(["-X", "utf8", "-u", "-m", module, *args])
        proc.readyReadStandardOutput.connect(self._read_training_output)
        proc.finished.connect(self._training_finished)
        proc.errorOccurred.connect(self._training_error)
        self.recovery_button.setEnabled(False)
        self.light_weight_button.setEnabled(False)
        self.validation_button.setEnabled(False)
        self.triage_button.setEnabled(False)
        self.recovery_cancel.setEnabled(True)
        for button in (self.select_button, self.preview_button, self.commit_button):
            button.setEnabled(False)
        proc.start()

    def start_recovery(self):
        """Conserva a comparação anterior de 5 e 7 épocas."""
        self._start_experimental_comparison(light_weights=False)

    def start_light_weight_comparison(self):
        """Três rodadas independentes, mudando somente o peso auxiliar por luz."""
        self._start_experimental_comparison(light_weights=True)

    def _start_experimental_comparison(self, *, light_weights: bool):
        """Um único subprocesso na manutenção, sem XP e sem promoção."""
        if self.train_process is not None:
            return
        try:
            correction_id = latest_completed_correction(self.root)
            verified = validate_recovery_inputs(self.root, correction_id)
        except Exception as exc:
            QMessageBox.warning(self, "Treinamento bloqueado", str(exc))
            self.log.setPlainText(f"PRÉ-VALIDAÇÃO BLOQUEADA:\n{exc}")
            return

        proposal = (
            "Serão feitas três rodadas INDEPENDENTES de 7 épocas, "
            "com pesos auxiliares da perda SIDE/TOP/MID: 0,15 (controle), "
            "0,50 e 1,00. Todas partem do checkpoint original verificado. "
            "Os mesmos 3 eventos NG (9 recortes) e histórico entram em "
            "todos os treinamentos.\n\n"
            if light_weights else
            "Serão feitas duas rodadas INDEPENDENTES, de 5 e 7 épocas, "
            "sempre a partir dos mesmos pesos originais. Cada rodada utiliza "
            "3 eventos NG corrigidos (9 recortes) e o histórico confiável.\n\n"
        )
        answer = QMessageBox.question(
            self, "Confirmar treinamento experimental",
            proposal +
            "O experimento pode levar vários minutos e consumir CPU/RAM. "
            "Nenhum checkpoint será ativado e nenhum 0/1 será enviado ao XP.\n\n"
            "Executar comparação SOMENTE COM CANDIDATOS?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.log.setPlainText(
            ("COMPARAÇÃO PESOS POR ILUMINAÇÃO — 7 ÉPOCAS CADA\n"
             if light_weights else
             "COMPARAÇÃO 5 E 7 ÉPOCAS — PESOS ATIVOS PRESERVADOS\n") +
            f"Correção: {correction_id}\n"
            f"Eventos NG completos: {len(verified['events'])}\n"
            "Cada rodada reinicia a CNN dos pesos originais.\n"
            "Validação offline confirmada; iniciando processo CPU...\n"
        )
        module = (
            "src.scripts.compare_faltando_recovery_light_weights"
            if light_weights else "src.scripts.compare_faltando_recovery_epochs"
        )
        self._launch_worker(
            module, ["--root", str(self.root),
                     "--correction-id", correction_id],
            workflow="training_comparison",
        )

    def _read_training_output(self):
        if self.train_process is None:
            return
        chunk = bytes(self.train_process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if chunk:
            self.log.insertPlainText(chunk)
            self.log.ensureCursorVisible()

    def _training_finished(self, code, _status):
        self._read_training_output()
        if code == 0:
            if self._active_workflow == "candidate_validation":
                self.log.append(
                    "\nVALIDAÇÃO CONCLUÍDA. Envie o JSON comparacao_acervo_"
                    "cnn_peso050_*.json em reports/startup_regression/"
                    "cnn_candidate_validation/. Checkpoints intactos."
                )
            else:
                self.log.append(
                    "\nCOMPARAÇÃO DE TREINO CONCLUÍDA — consulte o JSON em "
                    "reports/neural_online/recovery_comparisons."
                )
        else:
            self.log.append(
                f"\nPROCESSO INTERROMPIDO/FALHOU (código {code}); "
                "nenhum peso foi ativado por esta ferramenta."
            )
        self.recovery_cancel.setEnabled(False)
        self.recovery_button.setEnabled(True)
        self.light_weight_button.setEnabled(True)
        self.validation_button.setEnabled(True)
        self.triage_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        # A prévia de correção deve ser refeita após qualquer experimento.
        self.plan = None
        if self.train_process is not None:
            self.train_process.deleteLater()
        self.train_process = None
        self._active_workflow = None

    def _training_error(self, _error):
        if self.train_process is not None:
            self._read_training_output()
            self.log.append("\nErro ao iniciar ou executar o subprocesso de treino.")

    def cancel_recovery(self):
        if self.train_process is None:
            return
        self.log.append("\nCancelamento solicitado. Aguarde o encerramento...")
        self.recovery_cancel.setEnabled(False)
        self.train_process.kill()

    def closeEvent(self, event):
        if self.train_process is not None:
            QMessageBox.warning(
                self, "Treinamento em execução",
                "Cancele o experimento e aguarde o encerramento antes "
                "de fechar a manutenção."
            )
            event.ignore()
            return
        event.accept()

    def select_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Selecione as nove capturas do julgamento incorreto",
            str(self.root / "public" / "ok_archive"), "Imagens PNG (*.png)",
        )
        if not paths:
            return
        self.files = paths
        self.plan = None
        self.commit_button.setEnabled(False)
        self.log.setPlainText(
            f"{len(paths)} capturas selecionadas:\n\n" +
            "\n".join(Path(p).name for p in paths) +
            "\n\nClique em Conferir vínculos e alterações."
        )

    def preview(self):
        if not self.files:
            QMessageBox.warning(self, "Sem imagens", "Selecione as nove capturas.")
            return
        self.commit_button.setEnabled(False)
        self.plan = None
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            plan = prepare_correction(self.root, self.files)
            self.plan = plan
            lines = [
                "PRÉVIA VALIDADA — nenhum arquivo foi alterado.",
                "",
                f"Capturas OK → NG: {len(plan['items'])}",
                f"Eventos reais verificados: {len(plan['events'])}",
                f"Registros de dataset: {len(plan['items'])}",
                f"Memórias KNN antigas OK: {plan['memories_ok']}",
                f"Memórias KNN já NG: {plan['memories_ng']}",
                f"Eventos de journal CNN a corrigir: {len(plan['online'])}",
                "",
                "Por evento confirmado:",
            ]
            for event in sorted(plan["events"]):
                images = [x for x in plan["items"] if x["event_id"] == event]
                lines.append(f"  {event} — {images[0]['category']}: "
                             + ", ".join(x["mode"] for x in images))
            lines += [
                "", "As imagens serão transferidas para ng_archive.",
                "O dataset correspondente passará de nao_anomalia para anomalia.",
                "Registros OK de mesma identidade visual serão substituídos por NG.",
                "O journal online terá o rótulo corrigido; checkpoints não mudam.",
                "", "A confirmação exige digitar CORRIGIR NG.",
            ]
            self.log.setPlainText("\n".join(lines))
            self.commit_button.setEnabled(True)
        except Exception as exc:
            self.log.setPlainText(f"PREVIEW BLOQUEADA:\n{exc}\n\nNenhum arquivo alterado.")
            QMessageBox.warning(self, "Correção bloqueada", str(exc))
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

    def commit(self):
        if self.plan is None:
            return
        typed, accepted = QInputDialog.getText(
            self, "Confirmar correção",
            "Confirma que as 9 capturas são defeitos reais (NG)?\n"
            "Digite CORRIGIR NG para autorizar a retificação.",
        )
        if not accepted or typed.strip().upper() != "CORRIGIR NG":
            return
        self.commit_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            result = apply_correction(self.plan)
            self.plan = None
            self.files = []
            self.log.setPlainText(
                "CORREÇÃO CONCLUÍDA.\n\n"
                f"Imagens retificadas: {result['images_corrected']}\n"
                f"Eventos: {result['events_corrected']}\n"
                f"Registros dataset: {result['dataset_records_corrected']}\n"
                f"Journal CNN: {result['cnn_online_journal_corrected']}\n"
                f"Memórias OK removidas: {result['knn']['old_ok_removed']}\n"
                f"Backup transacional: {result['backup_folder']}\n\n"
                "Os pesos da CNN não foram alterados. Próxima etapa depende de aprovação."
            )
            QMessageBox.information(
                self, "Retificação concluída",
                "As 9 capturas foram corrigidas. Feche esta janela antes "
                "de voltar a iniciar o ODIN normal. A CNN será avaliada em etapa separada."
            )
        except Exception as exc:
            self.log.setPlainText(f"RETIFICAÇÃO NÃO CONCLUÍDA:\n{exc}")
            QMessageBox.critical(self, "Falha na correção", str(exc))
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            self.select_button.setEnabled(True)
            self.preview_button.setEnabled(True)


def run_archive_label_correction(parent_pid: int = 0) -> int:
    wait_for_parent_to_exit(parent_pid)
    app = QApplication(["ODIN Label Correction"])
    dialog = ArchiveLabelCorrectionDialog()
    dialog.show()
    return app.exec()


__all__ = ["ArchiveLabelCorrectionDialog", "run_archive_label_correction"]
