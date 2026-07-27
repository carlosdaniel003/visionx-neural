"""Aplica iconografia SVG aos componentes visuais do VisionX Neural."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from src.ui.iconography_model import sanitize_visual_text, status_icon_name


ICON_DIR = Path(__file__).resolve().parent / "icons"


def icon_path(name: str) -> Path:
    return ICON_DIR / f"{name}.svg"


def svg_icon(name: str) -> QIcon:
    path = icon_path(name)
    return QIcon(str(path)) if path.is_file() else QIcon()


def install_iconography_hooks(control_panel_cls, operational_presenter_cls=None) -> None:
    """Instala sanitização antes que sinais sejam conectados ao controller."""
    if not getattr(control_panel_cls, "_svg_iconography_hooks", False):
        original_network_status = control_panel_cls.update_network_status
        original_brain_status = control_panel_cls.update_brain_status
        original_history_status = control_panel_cls.update_history_status

        def update_network_status(self, message: str):
            raw_message = str(message or "")
            clean_message = sanitize_visual_text(raw_message)
            original_network_status(self, clean_message)

            icon_name = status_icon_name("network", raw_message)
            label = getattr(getattr(self, "ui_builder", None), "lbl_status_network", None)
            if label is not None:
                if icon_name == "defect":
                    color = "#ff6262"
                elif icon_name == "warning":
                    color = "#f5c518"
                else:
                    color = "#4ade80"
                label.setStyleSheet(
                    f"color: {color}; font-size: 11px; font-weight: bold; border: none;"
                )

            presenter = getattr(self, "_svg_iconography", None)
            if presenter is not None:
                presenter.set_status_icon("network", icon_name)

        def update_brain_status(self, message: str, is_active: bool = False):
            raw_message = str(message or "")
            clean_message = sanitize_visual_text(raw_message)
            original_brain_status(self, clean_message, is_active)

            presenter = getattr(self, "_svg_iconography", None)
            if presenter is not None:
                presenter.set_status_icon(
                    "brain",
                    status_icon_name("brain", raw_message, active=is_active),
                )

        def update_history_status(self, label: str, source: str):
            original_history_status(self, label, source)
            history_label = getattr(
                getattr(self, "ui_builder", None),
                "lbl_status_history",
                None,
            )
            if history_label is not None:
                history_label.setText(sanitize_visual_text(history_label.text()))

            presenter = getattr(self, "_svg_iconography", None)
            if presenter is not None:
                presenter.set_status_icon(
                    "history",
                    status_icon_name("history", label),
                )

        control_panel_cls.update_network_status = update_network_status
        control_panel_cls.update_brain_status = update_brain_status
        control_panel_cls.update_history_status = update_history_status
        control_panel_cls._svg_iconography_hooks = True

    if (
        operational_presenter_cls is not None
        and not getattr(operational_presenter_cls, "_svg_iconography_hooks", False)
    ):
        original_apply_texts = operational_presenter_cls._apply_texts

        def apply_texts(self, state_name: str):
            original_apply_texts(self, state_name)
            self._set_text(self.panel.btn_light_mid, "Luz MID")
            self._set_text(self.panel.btn_light_side, "Luz SIDE")
            self._set_text(self.panel.btn_light_top, "Luz TOP")

        operational_presenter_cls._apply_texts = apply_texts
        operational_presenter_cls._svg_iconography_hooks = True


class SvgIconographyPresenter:
    """Associa os SVGs aos botões e aos três estados da barra inferior."""

    BUTTON_ICONS = (
        ("btn_start", "capture", 19),
        ("btn_skip", "discard", 18),
        ("btn_save_ok", "approve", 19),
        ("btn_save_ng", "defect", 19),
        ("btn_light_mid", "light-left", 18),
        ("btn_light_side", "light-down", 18),
        ("btn_light_top", "light-right", 18),
        ("btn_clear_dataset", "database-delete", 18),
    )

    def __init__(self, panel):
        self.panel = panel
        self.status_icons: dict[str, QLabel] = {}
        self.status_groups: list[QWidget] = []
        panel._svg_iconography = self

        self._configure_window()
        self._configure_buttons()
        self._rebuild_status_bar()
        self._sanitize_existing_texts()
        self._initialize_status_icons()

    def _configure_window(self) -> None:
        self.panel.setWindowIcon(svg_icon("processor"))

    def _configure_buttons(self) -> None:
        for attribute, icon_name, size in self.BUTTON_ICONS:
            button = getattr(self.panel, attribute, None)
            if button is None:
                continue
            button.setIcon(svg_icon(icon_name))
            button.setIconSize(QSize(size, size))
            button.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        for attribute, clean_text in (
            ("btn_light_mid", "Luz MID"),
            ("btn_light_side", "Luz SIDE"),
            ("btn_light_top", "Luz TOP"),
        ):
            button = getattr(self.panel, attribute, None)
            if button is not None:
                button.setText(clean_text)

        tooltips = {
            "btn_start": "Inicia uma nova captura para análise.",
            "btn_skip": "Descarta a captura atual sem registrar decisão.",
            "btn_save_ok": "Confirma a peça como OK após revisar os diagnósticos.",
            "btn_save_ng": "Confirma a peça como defeito NG após revisar os diagnósticos.",
            "btn_light_mid": "Seleciona a iluminação MID. Atalho: seta esquerda.",
            "btn_light_side": "Seleciona a iluminação SIDE. Atalho: seta para baixo.",
            "btn_light_top": "Seleciona a iluminação TOP. Atalho: seta direita.",
            "btn_clear_dataset": "Apaga as amostras e metadados do dataset local.",
        }
        for attribute, tooltip in tooltips.items():
            button = getattr(self.panel, attribute, None)
            if button is not None:
                button.setToolTip(tooltip)

    @staticmethod
    def _make_icon_label(parent: QWidget) -> QLabel:
        label = QLabel(parent)
        label.setObjectName("svgStatusIcon")
        label.setFixedSize(18, 18)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return label

    def _make_status_group(
        self,
        slot: str,
        text_label: QLabel,
        alignment: str,
    ) -> QWidget:
        group = QWidget(self.panel.status_frame)
        group.setObjectName("statusGroup")
        group_layout = QHBoxLayout(group)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(6)

        icon_label = self._make_icon_label(group)
        self.status_icons[slot] = icon_label

        if alignment in {"center", "right"}:
            group_layout.addStretch()
        group_layout.addWidget(icon_label)
        group_layout.addWidget(text_label)
        if alignment in {"left", "center"}:
            group_layout.addStretch()

        self.status_groups.append(group)
        return group

    def _rebuild_status_bar(self) -> None:
        status_frame = getattr(self.panel, "status_frame", None)
        if status_frame is None:
            return
        layout = status_frame.layout()
        if layout is None:
            return

        while layout.count():
            layout.takeAt(0)

        network_label = self.panel.lbl_status_network
        brain_label = self.panel.lbl_status_brain
        history_label = self.panel.lbl_status_history

        layout.addWidget(
            self._make_status_group("network", network_label, "left"),
            stretch=1,
        )
        layout.addWidget(
            self._make_status_group("brain", brain_label, "center"),
            stretch=1,
        )
        layout.addWidget(
            self._make_status_group("history", history_label, "right"),
            stretch=1,
        )

    def _sanitize_existing_texts(self) -> None:
        for attribute in (
            "lbl_status_network",
            "lbl_status_brain",
            "lbl_status_history",
        ):
            label = getattr(self.panel, attribute, None)
            if label is not None:
                label.setText(sanitize_visual_text(label.text()))

    def _initialize_status_icons(self) -> None:
        self.set_status_icon(
            "network",
            status_icon_name("network", self.panel.lbl_status_network.text()),
        )
        self.set_status_icon(
            "brain",
            status_icon_name("brain", self.panel.lbl_status_brain.text()),
        )
        self.set_status_icon(
            "history",
            status_icon_name("history", self.panel.lbl_status_history.text()),
        )

    def set_status_icon(self, slot: str, icon_name: str) -> None:
        label = self.status_icons.get(slot)
        if label is None:
            return
        label.setPixmap(svg_icon(icon_name).pixmap(QSize(16, 16)))
        label.setToolTip(icon_name.replace("-", " ").title())


def install_svg_iconography(panel) -> None:
    """Instala os SVGs depois que todos os controles já foram criados."""
    if getattr(panel, "_svg_iconography_installed", False):
        return
    SvgIconographyPresenter(panel)
    panel._svg_iconography_installed = True
