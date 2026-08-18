"""Resumo visual das novas camadas de memória no painel principal."""

from __future__ import annotations

from src.ui.memory_status_model import memory_status_from_detail, memory_summary_text


def install_memory_status_ui(control_panel_cls) -> None:
    """Atualiza apenas textos da UI; não interfere na decisão da inspeção."""
    if getattr(control_panel_cls, "_memory_status_ui_installed", False):
        return

    original_update_confidence = control_panel_cls._update_confidence_panel

    def update_confidence_panel(self, analysis: dict):
        original_update_confidence(self, analysis)
        detail = (analysis or {}).get("detail", {})
        model = memory_status_from_detail(detail)

        if hasattr(self, "lbl_db_info"):
            self.lbl_db_info.setText(memory_summary_text(detail))
            if model["conflict"]:
                self.lbl_db_info.setStyleSheet(
                    "color: #ffb454; font-size: 12px; font-weight: 800; "
                    "border: none; background: transparent;"
                )
            elif model["leading_hypothesis"] == "NG" and model["has_memory"]:
                self.lbl_db_info.setStyleSheet(
                    "color: #ff7b72; font-size: 12px; font-weight: 700; "
                    "border: none; background: transparent;"
                )
            elif model["leading_hypothesis"] == "OK" and model["has_memory"]:
                self.lbl_db_info.setStyleSheet(
                    "color: #3fb950; font-size: 12px; font-weight: 700; "
                    "border: none; background: transparent;"
                )
            else:
                self.lbl_db_info.setStyleSheet(
                    "color: #8b949e; font-size: 12px; font-weight: 600; "
                    "border: none; background: transparent;"
                )

        # Destaque exclusivamente visual. O analysis original permanece intacto.
        if model["conflict"] and hasattr(self, "lbl_verdict"):
            self.lbl_verdict.setText("CONFLITO DE MEMÓRIA • REVISÃO OBRIGATÓRIA")
            self.lbl_verdict.setStyleSheet(
                "color: #ffb454; font-size: 16px; font-weight: 800; border: none;"
            )

    control_panel_cls._update_confidence_panel = update_confidence_panel
    control_panel_cls._memory_status_ui_installed = True


__all__ = ["install_memory_status_ui"]
