"""Controles administrativos disponíveis exclusivamente no Modo Teste."""

from __future__ import annotations

from PyQt6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton

from src.services.dataset_cleaner import clear_local_dataset


def install_test_mode_dataset_controls(window) -> None:
    """Adiciona ao painel o botão de limpeza do dataset local."""
    if hasattr(window, "btn_clear_dataset"):
        return

    window.btn_clear_dataset = QPushButton("Excluir Dataset Local")
    if hasattr(window.btn_clear_dataset, "setObjectName"):
        window.btn_clear_dataset.setObjectName("deleteDatasetButton")
    window.btn_clear_dataset.setToolTip(
        "Apaga as amostras OK/NG e os metadados salvos no computador."
    )

    button_row = QHBoxLayout()
    if hasattr(button_row, "setContentsMargins"):
        button_row.setContentsMargins(0, 0, 0, 0)
    button_row.addStretch()
    button_row.addWidget(window.btn_clear_dataset)

    action_layout = window.action_widget.layout()
    action_layout.addLayout(button_row)

    def update_visibility(mode_text: str) -> None:
        window.btn_clear_dataset.setVisible(mode_text == "Modo Teste")

    def handle_clear_dataset() -> None:
        if window.combo_mode.currentText() != "Modo Teste":
            return

        answer = QMessageBox.question(
            window,
            "Excluir dataset local",
            "Esta ação apagará permanentemente todas as amostras OK/NG e "
            "os metadados JSON armazenados em public/dataset.\n\n"
            "Deseja continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        result = clear_local_dataset()

        # A memória em RAM deve refletir o estado atual do disco, mesmo se a
        # limpeza tiver sido apenas parcial.
        window.orchestrator.reload_memory()

        if result["success"]:
            if hasattr(window, "skip_image"):
                window.skip_image()
            if hasattr(window, "lbl_db_info"):
                window.lbl_db_info.setText("Dataset local vazio.")
            window.update_brain_status(
                "Dataset local excluído. Memória KNN zerada.", False
            )

            if hasattr(window.ui_builder, "lbl_status_history"):
                window.ui_builder.lbl_status_history.setText(
                    "Dataset local excluído no Modo Teste"
                )

            QMessageBox.information(
                window,
                "Dataset excluído",
                f"Limpeza concluída.\n\n"
                f"Arquivos excluídos: {result['deleted_files']}\n"
                f"Subpastas excluídas: {result['deleted_directories']}",
            )
            return

        details = "\n".join(result["errors"])
        window.update_brain_status(
            "Falha parcial ao excluir o dataset local.", False
        )
        QMessageBox.warning(
            window,
            "Falha ao excluir dataset",
            "A limpeza não foi concluída integralmente.\n\n" + details,
        )

    window.btn_clear_dataset.clicked.connect(handle_clear_dataset)
    window.combo_mode.currentTextChanged.connect(update_visibility)
    update_visibility(window.combo_mode.currentText())
