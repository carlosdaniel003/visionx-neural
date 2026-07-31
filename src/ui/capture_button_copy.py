"""Textos operacionais específicos do botão de captura."""

from __future__ import annotations


def install_capture_button_copy(presenter_cls) -> None:
    if getattr(presenter_cls, "_capture_button_copy_installed", False):
        return

    original_apply_texts = presenter_cls._apply_texts

    def apply_texts(self, state_name: str):
        original_apply_texts(self, state_name)
        if state_name == "idle":
            self._set_text(self.panel.btn_start, "Capturar local (MSS)")
            self.panel.btn_start.setToolTip(
                "Minimiza o VisionX e inicia uma captura local da tela usando MSS."
            )
        elif state_name == "review_test":
            self._set_text(
                self.panel.btn_start,
                "Capturar nova peça (descarta a atual)",
            )
            self.panel.btn_start.setToolTip(
                "Descarta a análise atual e inicia uma nova captura local por MSS."
            )

    presenter_cls._apply_texts = apply_texts
    presenter_cls._capture_button_copy_installed = True


__all__ = ["install_capture_button_copy"]
