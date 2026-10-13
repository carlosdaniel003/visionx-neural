"""Ferramenta administrativa do ODIN: modo manutenção sem recepção XP."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
    QMessageBox, QPushButton, QTextEdit, QVBoxLayout,
)

from src.config.settings import BASE_DIR
from src.services.archive_label_correction import (
    apply_correction, prepare_correction,
)


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
