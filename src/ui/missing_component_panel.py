"""Integra o debugger do motor FALTANDO sem alterar a grade responsiva."""

from __future__ import annotations

from types import MethodType

from src.ui.widgets.missing_debugger import MissingDebuggerWidget


def install_missing_component_panel(panel) -> None:
    if getattr(panel, "_missing_component_panel_installed", False):
        return

    panel.frame_missing = MissingDebuggerWidget()
    panel.frame_missing.setVisible(False)

    layout = panel.scroll_layout
    # O último item é o stretch usado para empurrar os cards à esquerda.
    if layout.count() > 0:
        last_item = layout.itemAt(layout.count() - 1)
        if last_item is not None and last_item.spacerItem() is not None:
            layout.takeAt(layout.count() - 1)

    wrapped = panel.ui_builder._wrap_debug_widget(
        "EXPECTATIVA DO PATCH • MOTOR FALTANDO",
        panel.frame_missing,
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
            "missing_expert.py" in active_engines
            and bool(detail.get("missing_active", False))
        )
        self.frame_missing.setVisible(active)
        if active:
            self.frame_missing.update_data(detail)
        return result

    def wrapped_reference_reset(self):
        result = original_reference_reset()
        self.frame_missing.update_data({})
        self.frame_missing.setVisible(False)
        return result

    panel._update_reference_panel = MethodType(wrapped_reference_update, panel)
    panel._reset_reference_panel = MethodType(wrapped_reference_reset, panel)
    panel._missing_component_panel_installed = True


__all__ = ["install_missing_component_panel"]
