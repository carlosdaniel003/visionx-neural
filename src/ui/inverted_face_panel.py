"""Integra o debugger do motor INVERTIDO ao carrossel de especialistas."""

from __future__ import annotations

from types import MethodType

from src.ui.widgets.inverted_face_debugger import InvertedFaceDebuggerWidget


def install_inverted_face_panel(panel) -> None:
    if getattr(panel, "_inverted_face_panel_installed", False):
        return

    panel.frame_inverted = InvertedFaceDebuggerWidget()
    panel.frame_inverted.setVisible(False)

    layout = panel.scroll_layout
    if layout.count() > 0:
        last_item = layout.itemAt(layout.count() - 1)
        if last_item is not None and last_item.spacerItem() is not None:
            layout.takeAt(layout.count() - 1)

    wrapped = panel.ui_builder._wrap_debug_widget(
        "ASSINATURA DA FACE • MOTOR INVERTIDO",
        panel.frame_inverted,
    )
    layout.addWidget(wrapped)
    layout.addStretch()

    original_reference_update = panel._update_reference_panel
    original_reference_reset = panel._reset_reference_panel

    def wrapped_reference_update(self, analysis):
        result = original_reference_update(analysis)
        detail = (analysis or {}).get("detail", {})
        active_engines = (analysis or {}).get("active_engines", [])
        active = (
            "inverted_expert.py" in active_engines
            and bool(detail.get("inverted_active", False))
        )
        self.frame_inverted.setVisible(active)
        if active:
            self.frame_inverted.update_data(detail)
        return result

    def wrapped_reference_reset(self):
        result = original_reference_reset()
        self.frame_inverted.update_data({})
        self.frame_inverted.setVisible(False)
        return result

    panel._update_reference_panel = MethodType(wrapped_reference_update, panel)
    panel._reset_reference_panel = MethodType(wrapped_reference_reset, panel)
    panel._inverted_face_panel_installed = True


__all__ = ["install_inverted_face_panel"]
