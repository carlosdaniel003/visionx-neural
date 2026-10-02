"""Fundo dinâmico do ODIN conforme o estado final da análise.

Esta camada é exclusivamente visual. Ela observa o veredito já produzido pelo
pipeline e não altera classificação, confiança, memória, persistência ou ciclo.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget


VALID_BACKGROUND_STATES = {"neutral", "ok", "ng"}

BACKGROUND_OBJECT_NAMES = {
    "rootWindow",
    "rootContent",
    "rootViewport",
    "headerFrame",
    "sectionPanel",
    "infoSection",
    "confidenceFrame",
    "controlsSection",
    "statusBar",
}


def decision_background_state(analysis: dict | None) -> str:
    """Converte somente o veredito final existente em um estado visual."""
    if not isinstance(analysis, dict) or not analysis:
        return "neutral"
    return "ng" if bool(analysis.get("is_defect", False)) else "ok"


def _background_widgets(panel):
    """Seleciona somente containers de fundo; nunca reestiliza botões."""
    widgets = [panel]

    root_content = getattr(panel, "root_content", None)
    if root_content is not None:
        widgets.append(root_content)

    root_scroll = getattr(panel, "root_scroll", None)
    if root_scroll is not None:
        try:
            viewport = root_scroll.viewport()
        except Exception:
            viewport = None
        if viewport is not None:
            widgets.append(viewport)

    try:
        for widget in panel.findChildren(QWidget):
            if str(widget.objectName() or "") in BACKGROUND_OBJECT_NAMES:
                widgets.append(widget)
    except Exception:
        pass

    unique = []
    seen = set()
    for widget in widgets:
        marker = id(widget)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(widget)
    return unique


def apply_decision_background(panel, state: str) -> str:
    """Atualiza só propriedades dos containers e preserva todo stylesheet existente."""
    normalized = str(state or "neutral").strip().lower()
    if normalized not in VALID_BACKGROUND_STATES:
        normalized = "neutral"

    for widget in _background_widgets(panel):
        try:
            widget.setProperty("decisionState", normalized)
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)
            widget.update()
        except Exception:
            pass

    return normalized


def install_decision_background(control_panel_cls) -> None:
    """Observa resultado/reset do controller sem participar da regra de negócio."""
    if getattr(control_panel_cls, "_decision_background_installed", False):
        return

    original_reference_update = control_panel_cls._update_reference_panel
    original_reset = control_panel_cls._reset_confidence_panel
    original_save_label = control_panel_cls.save_label

    def wrapped_reference_update(self, analysis):
        result = original_reference_update(self, analysis)
        apply_decision_background(
            self,
            decision_background_state(analysis),
        )
        return result

    def wrapped_reset(self):
        result = original_reset(self)
        apply_decision_background(self, "neutral")
        return result

    def wrapped_save_label(self, *args, **kwargs):
        result = original_save_label(self, *args, **kwargs)

        # A cadeia operacional externa limpa current_analysis/is_locked ao
        # concluir o ciclo. Nesse ponto o ODIN voltou a esperar outra imagem.
        if (
            getattr(self, "current_analysis", None) is None
            or not bool(getattr(self, "is_locked", False))
        ):
            apply_decision_background(self, "neutral")
        return result

    control_panel_cls._update_reference_panel = wrapped_reference_update
    control_panel_cls._reset_confidence_panel = wrapped_reset
    control_panel_cls.save_label = wrapped_save_label
    control_panel_cls._decision_background_installed = True


__all__ = [
    "BACKGROUND_OBJECT_NAMES",
    "VALID_BACKGROUND_STATES",
    "apply_decision_background",
    "decision_background_state",
    "install_decision_background",
]
