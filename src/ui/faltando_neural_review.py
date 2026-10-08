"""Painel offline PyQt6 de qualificação humana do dataset CNN FALTANDO.

Executar: python -m src.ui.faltando_neural_review
Não importa main.py, não abre AOI/XP, não carrega IA e não treina.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.services.faltando_neural_qualification import (
    LIGHTS, QualificationStore, latest_manifest,
)


class FitImage(QLabel):
    """Exibe o par completo ajustado à largura; nunca recorta a região."""

    def __init__(self):
        super().__init__()
        self.original = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(220, 170))
        self.setStyleSheet(
            "QLabel { background: #161b22; border: 1px solid #303844; "
            "border-radius: 3px; color: #dbe2e8; }"
        )

    def set_path(self, path: Path | None):
        self.original = QPixmap(str(path)) if path is not None else QPixmap()
        if self.original.isNull():
            self.setText("Imagem ausente / ilegível")
        else:
            self.setText("")
            self._fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        if self.original.isNull():
            return
        size = QSize(max(1, self.width()-8), max(1, self.height()-8))
        self.setPixmap(self.original.scaled(
            size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))


class FaltandoReviewWindow(QMainWindow):
    def __init__(self, store: QualificationStore):
        super().__init__()
        self.store = store
        self.selected = None
        self.setWindowTitle("ODIN — Qualificação visual CNN FALTANDO (offline)")
        self.resize(1410, 930)
        self.setMinimumSize(910, 620)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #10151b; color: #dee5ed; }
            QLineEdit, QListWidget, QScrollArea {
                background: #171f28; border: 1px solid #323b46;
                color: #e5edf4; padding: 5px;
            }
            QListWidget::item:selected { background: #424c5a; }
            QPushButton { background: #27313c; color: #f0f4f8;
                border: 1px solid #53606b; padding: 9px; }
            QPushButton:hover { background: #384653; }
            QPushButton:disabled { color: #77818b; }
            QGroupBox { border: 1px solid #39424e; margin-top: 8px; }
            QGroupBox::title { padding: 0 6px; }
        """)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(split)
        sidebar = QWidget()
        left = QVBoxLayout(sidebar)
        left.addWidget(QLabel("CASOS E TRINCAS — FALTANDO"))
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        left.addWidget(self.stats)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filtrar por componente, arquivo, grupo ou status")
        self.filter.textChanged.connect(self._filter_items)
        left.addWidget(self.filter)
        self.items = QListWidget()
        self.items.currentItemChanged.connect(self._selected_changed)
        left.addWidget(self.items, 1)
        left.addWidget(QLabel(
            "Cada confirmação é salva em qualification.json. "
            "Não modifica imagens ou dataset/KNN."
        ))

        content = QWidget()
        right = QVBoxLayout(content)
        self.heading = QLabel("Selecione um caso ou trinca.")
        self.heading.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.heading.setWordWrap(True)
        right.addWidget(self.heading)
        self.description = QLabel()
        self.description.setWordWrap(True)
        right.addWidget(self.description)
        self.similar_label = QLabel()
        self.similar_label.setWordWrap(True)
        right.addWidget(self.similar_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.preview_container = QWidget()
        self.preview_grid = QGridLayout(self.preview_container)
        self.preview_grid.setSpacing(10)
        self.preview_grid.setColumnStretch(1, 1)
        self.preview_grid.setColumnStretch(2, 1)
        scroll.setWidget(self.preview_container)
        right.addWidget(scroll, 1)

        group = QGroupBox("Confirmação visual humana")
        options = QVBoxLayout(group)
        self.confirm_box = QCheckBox(
            "Conferi as imagens de gabarito e teste e a identidade do componente."
        )
        options.addWidget(self.confirm_box)
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("Observações do operador (opcional)")
        options.addWidget(self.notes)
        row = QHBoxLayout()
        self.ok = QPushButton("Confirmar OK / presente")
        self.ng = QPushButton("Confirmar NG / ausente")
        self.reject = QPushButton("Rejeitar recorte")
        self.ok.clicked.connect(lambda: self._review_case("CONFIRMED_OK"))
        self.ng.clicked.connect(lambda: self._review_case("CONFIRMED_NG"))
        self.reject.clicked.connect(lambda: self._review_case("REJECTED"))
        for control in (self.ok, self.ng, self.reject):
            row.addWidget(control)
        options.addLayout(row)

        group_row = QHBoxLayout()
        self.confirm_group = QPushButton("Confirmar vínculo SIDE/TOP/MID")
        self.reject_group = QPushButton("Rejeitar vínculo multilight")
        self.confirm_group.clicked.connect(
            lambda: self._review_group("CONFIRMED_VISUAL_ASSOCIATION")
        )
        self.reject_group.clicked.connect(lambda: self._review_group("REJECTED"))
        group_row.addWidget(self.confirm_group)
        group_row.addWidget(self.reject_group)
        options.addLayout(group_row)
        right.addWidget(group)
        split.addWidget(sidebar)
        split.addWidget(content)
        split.setSizes([425, 985])

        self._populate()
        if self.items.count():
            self.items.setCurrentRow(0)

    def _populate(self):
        self.items.clear()
        for name, group in sorted(self.store.groups.items()):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, ("GROUP", name))
            self.items.addItem(item)
        for path, case in sorted(self.store.by_path.items()):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, ("CASE", path))
            self.items.addItem(item)
        self._refresh()

    def _refresh(self):
        counts = self.store.summary()
        self.stats.setText(
            f"Casos: {counts['total']} | OK: {counts['confirmed_ok']} | "
            f"NG: {counts['confirmed_ng']} | rejeitados: {counts['rejected']} | "
            f"pendentes: {counts['pending']} | "
            f"trincas confirmadas: {counts['confirmed_groups']}/{len(self.store.groups)}"
        )
        for i in range(self.items.count()):
            item = self.items.item(i)
            kind, key = item.data(Qt.ItemDataRole.UserRole)
            if kind == "CASE":
                data = self.store.by_path[key]
                item.setText(
                    f"[{self.store.case_status(key)}] "
                    f"{data['expected_label_from_archive']} • "
                    f"{data['lighting_mode']} • "
                    f"{Path(key).name} • "
                    f"{data.get('ocr_observed', {}).get('parts', '')}"
                )
            else:
                item.setText(
                    f"[{self.store.group_status(key)}] "
                    f"3 LUZES • {self.store.groups[key]['label']} • {key}"
                )
        self._filter_items(self.filter.text())

    def _filter_items(self, value: str):
        text = value.strip().casefold()
        for i in range(self.items.count()):
            item = self.items.item(i)
            item.setHidden(text not in item.text().casefold())

    def _clear_preview(self):
        while self.preview_grid.count():
            entry = self.preview_grid.takeAt(0)
            if entry.widget() is not None:
                entry.widget().deleteLater()

    def _add_pair(self, row: int, light: str, path: str):
        sample = self.store.by_path[path]
        ref = self.store.files[path][1]
        test = self.store.files[path][2]
        description = QLabel(
            f"{light} • {sample['expected_label_from_archive']} (pasta) • "
            f"{sample.get('ocr_observed', {}).get('parts', '')}\n"
            f"{Path(path).name}\n"
            f"Revisão: {self.store.case_status(path)}"
        )
        description.setWordWrap(True)
        self.preview_grid.addWidget(description, row, 0)
        for col, (title, file) in enumerate(
            (("GABARITO", ref), ("TESTE", test)), start=1
        ):
            box = QWidget()
            layout = QVBoxLayout(box)
            label = QLabel(title)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            viewer = FitImage()
            viewer.set_path(file)
            layout.addWidget(viewer, 1)
            self.preview_grid.addWidget(box, row, col)

    def _selected_changed(self, current, previous):
        self._clear_preview()
        self.confirm_box.setChecked(False)
        self.notes.clear()
        if current is None:
            self.selected = None
            return
        kind, key = current.data(Qt.ItemDataRole.UserRole)
        self.selected = kind, key
        is_group = kind == "GROUP"
        for w in (self.ok, self.ng, self.reject):
            w.setVisible(not is_group)
        for w in (self.confirm_group, self.reject_group):
            w.setVisible(is_group)
        if is_group:
            group = self.store.groups[key]
            self.heading.setText(f"TRINCA CANDIDATA • {key}")
            self.description.setText(
                "Vínculo por nome ainda NÃO é event_id comprovado. "
                "Confira que as três iluminações mostram a MESMA peça e "
                "que o rótulo de cada par está correto. Confirme primeiro "
                "cada caso individualmente na lista."
            )
            self.confirm_box.setText(
                "Confirmei visualmente que SIDE/TOP/MID pertencem à mesma peça."
            )
            for row, mode in enumerate(LIGHTS):
                self._add_pair(row, mode, group["paths"][mode])
            review = self.store.group_reviews.get(key, {})
            self.notes.setText(review.get("notes", ""))
            self.similar_label.setText(
                "Estado do vínculo: " + self.store.group_status(key)
            )
        else:
            item = self.store.by_path[key]
            self.heading.setText(f"{item['expected_label_from_archive']} • "
                                 f"{item['lighting_mode']} • {Path(key).name}")
            self.description.setText(
                "Rótulo da pasta é provisório; inspecione o corpo do componente, "
                "os pads e as duas imagens integrais. Caso a classe humana "
                "contradiga o arquivo original, a divergência será registrada "
                "sem modificar a pasta."
            )
            self.confirm_box.setText(
                "Inspecionei gabarito e teste: recorte e classe humana estão corretos."
            )
            self._add_pair(0, item["lighting_mode"], key)
            review = self.store.case_reviews.get(key, {})
            self.notes.setText(review.get("notes", ""))
            similar = self.store.similar.get(key, [])
            if similar:
                others = ", ".join(
                    f"{Path(entry['other_path']).name} "
                    f"(dH={entry['reference_distance']}/{entry['test_distance']})"
                    for entry in similar
                )
                self.similar_label.setText(
                    "Possíveis repetições VISUAIS (não comprovadas): " + others
                )
            else:
                self.similar_label.setText(
                    "Nenhuma repetição próxima encontrada pela triagem dHash."
                )

    def _confirm_checked(self) -> bool:
        if self.confirm_box.isChecked():
            return True
        QMessageBox.information(
            self, "Revisão obrigatória",
            "Confirme a inspeção visual antes de gravar uma decisão."
        )
        return False

    def _run_action(self, callback):
        if not self._confirm_checked():
            return
        try:
            callback()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Qualificação não gravada", str(exc))
            return
        self._refresh()
        self._selected_changed(self.items.currentItem(), None)
        self.statusBar().showMessage("Qualificação salva em qualification.json", 5500)

    def _review_case(self, status: str):
        if not self.selected or self.selected[0] != "CASE":
            return
        path = self.selected[1]
        self._run_action(
            lambda: self.store.review_case(path, status, self.notes.text())
        )

    def _review_group(self, status: str):
        if not self.selected or self.selected[0] != "GROUP":
            return
        group = self.selected[1]
        self._run_action(
            lambda: self.store.review_group(group, status, self.notes.text())
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Revisão visual offline FALTANDO (sem treinamento)."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Manifesto da execução desejada; padrão: último run_*/manifest.json",
    )
    args = parser.parse_args(argv)
    manifest = args.manifest if args.manifest else latest_manifest(args.root)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    store = QualificationStore(manifest)
    window = FaltandoReviewWindow(store)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
