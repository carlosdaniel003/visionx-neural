"""Adapta sinais ``clicked(bool)`` aos métodos operacionais sem argumentos.

QPushButton.clicked envia um booleano ``checked``. Como os wrappers de segurança
aceitam ``*args``, esse valor podia atravessar toda a cadeia e chegar aos métodos
originais ``ControlPanel.start_monitoring()`` e ``ControlPanel.skip_image()``,
que aceitam somente ``self``.

Esta deve ser a última camada instalada sobre o ControlPanel. Ela consome o
booleano do Qt e chama a cadeia operacional sem argumentos adicionais.
"""

from __future__ import annotations

from functools import wraps


def install_qt_button_signal_adapter(control_panel_cls) -> None:
    """Remove o argumento ``checked`` antes de entrar nos wrappers internos."""
    if getattr(control_panel_cls, "_qt_button_signal_adapter_installed", False):
        return

    original_start_monitoring = control_panel_cls.start_monitoring
    original_skip_image = control_panel_cls.skip_image

    @wraps(original_start_monitoring)
    def start_monitoring_from_button(self, _checked: bool = False):
        # Não encaminhar _checked. A captura local original recebe apenas self.
        return original_start_monitoring(self)

    @wraps(original_skip_image)
    def skip_image_from_button(self, _checked: bool = False):
        # Não encaminhar _checked. O descarte original recebe apenas self.
        return original_skip_image(self)

    control_panel_cls.start_monitoring = start_monitoring_from_button
    control_panel_cls.skip_image = skip_image_from_button
    control_panel_cls._qt_button_signal_adapter_installed = True


__all__ = ["install_qt_button_signal_adapter"]
