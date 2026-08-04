"""Filtro final para imagens AOI recebidas pela rede.

O Windows XP pode enviar a tela da central durante a transição entre peças. O
receptor confirma estabilidade temporal; este módulo confirma o conteúdo da
inspeção antes que ``ControlPanel.process_aoi_images`` substitua a peça atual.
Capturas MSS locais não passam por este filtro.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.core.epicenter_extractor import EpicenterExtractor
from src.core.inspection import detect_anomalies


MIN_FOCUS_SIDE = 4


def _valid_image(value: Any) -> bool:
    return bool(
        isinstance(value, np.ndarray)
        and value.size > 0
        and value.ndim in {2, 3}
        and value.shape[0] >= MIN_FOCUS_SIDE
        and value.shape[1] >= MIN_FOCUS_SIDE
    )


def validate_network_inspection(
    sample_crop: np.ndarray,
    ng_crop: np.ndarray,
) -> tuple[bool, str, dict]:
    """Exige recortes utilizáveis e o epicentro menor escolhido pelo sistema."""
    if not _valid_image(sample_crop) or not _valid_image(ng_crop):
        return False, "gabarito ou teste vazio/inválido", {
            "valid": False,
            "reason": "invalid_crops",
        }

    try:
        (
            raw_anomalies,
            old_epicenters,
            global_box_info,
            _gab_focus,
            _test_focus,
        ) = detect_anomalies(sample_crop, ng_crop)
        real_epicenters, focus_gab, focus_ng = EpicenterExtractor.extract_focus(
            sample_crop,
            ng_crop,
            old_epicenters,
            global_box_info,
        )
    except Exception as exc:
        return False, f"falha ao validar epicentro: {exc}", {
            "valid": False,
            "reason": "validation_exception",
            "error": str(exc),
        }

    if not real_epicenters:
        return False, "tela sem epicentro de anomalia", {
            "valid": False,
            "reason": "missing_epicenter",
            "raw_anomaly_count": int(len(raw_anomalies or [])),
        }

    try:
        x, y, width, height = (
            int(round(float(value))) for value in real_epicenters[0][:4]
        )
    except Exception:
        return False, "coordenadas do epicentro inválidas", {
            "valid": False,
            "reason": "invalid_epicenter_box",
        }

    if width < MIN_FOCUS_SIDE or height < MIN_FOCUS_SIDE:
        return False, "epicentro menor que o mínimo operacional", {
            "valid": False,
            "reason": "epicenter_too_small",
            "focus_box": [x, y, width, height],
        }

    if not _valid_image(focus_gab) or not _valid_image(focus_ng):
        return False, "epicentro não gerou o par gabarito/teste", {
            "valid": False,
            "reason": "empty_focus_pair",
            "focus_box": [x, y, width, height],
        }

    return True, "epicentro válido", {
        "valid": True,
        "reason": "valid_epicenter",
        "focus_box": [x, y, width, height],
        "focus_shape": [
            int(focus_ng.shape[1]),
            int(focus_ng.shape[0]),
        ],
        "raw_anomaly_count": int(len(raw_anomalies or [])),
        "epicenter_count": int(len(real_epicenters)),
    }


def _safe_button(button, *, enabled: bool | None = None, text: str | None = None):
    if button is None:
        return
    try:
        if enabled is not None:
            button.setEnabled(bool(enabled))
        if text is not None:
            button.setText(str(text))
    except Exception:
        pass


def reject_invalid_network_capture(panel, reason: str, audit: dict | None = None) -> None:
    """Libera a procura sem permitir que a central ocupe uma peça ativa."""
    receiver = getattr(panel, "network_receiver", None)
    if receiver is not None and hasattr(receiver, "mark_reserved_image_rejected"):
        try:
            receiver.mark_reserved_image_rejected(reason)
        except Exception as exc:
            print(f"Falha não fatal ao registrar tela rejeitada: {exc}")

    # Invalida o watchdog criado pelo wrapper de recepção atual.
    panel.capture_cycle_network_generation = int(
        getattr(panel, "capture_cycle_network_generation", 0)
    ) + 1
    panel.capture_cycle_active = False
    panel.capture_cycle_ignored_signals = 0
    panel.capture_cycle_source = None
    panel.current_analysis = None
    panel.current_sample = None
    panel.current_ng = None
    panel.current_aoi_info = {}
    panel.is_locked = False
    panel.network_intake_last_validation = dict(audit or {})

    if hasattr(panel, "production_review_pending"):
        panel.production_review_pending = False

    for method_name in (
        "_reset_confidence_panel",
        "_reset_reference_panel",
        "_reset_aoi_info",
    ):
        method = getattr(panel, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:
                print(f"Falha não fatal em {method_name}: {exc}")

    _safe_button(
        getattr(panel, "btn_start", None),
        enabled=True,
        text="Capturar local (MSS)",
    )
    for name in ("btn_save_ok", "btn_save_ng", "btn_skip"):
        _safe_button(getattr(panel, name, None), enabled=False)

    if receiver is not None and hasattr(receiver, "release_image_gate"):
        try:
            receiver.release_image_gate()
        except Exception as exc:
            print(f"Falha não fatal ao liberar receptor após rejeição: {exc}")

    presenter = getattr(panel, "_operational_controls", None)
    if presenter is not None:
        try:
            presenter.sync(force=True)
        except Exception as exc:
            print(f"Falha não fatal ao sincronizar controles: {exc}")

    try:
        panel.update_brain_status(
            "Tela da central/transição ignorada. "
            "Aguardando a próxima anomalia AOI válida.",
            False,
        )
    except Exception:
        pass


def install_network_aoi_intake_filter(control_panel_cls) -> None:
    """Protege somente o caminho de imagens externas recebidas pela rede."""
    if getattr(control_panel_cls, "_network_aoi_intake_filter_installed", False):
        return

    original_process_aoi_images = control_panel_cls.process_aoi_images

    def process_aoi_images(self, sample_crop, ng_crop, aoi_info):
        if getattr(self, "capture_cycle_source", None) != "network":
            return original_process_aoi_images(
                self,
                sample_crop,
                ng_crop,
                aoi_info,
            )

        valid, reason, audit = validate_network_inspection(
            sample_crop,
            ng_crop,
        )
        self.network_intake_last_validation = dict(audit)
        if not valid:
            reject_invalid_network_capture(self, reason, audit)
            return False

        receiver = getattr(self, "network_receiver", None)
        mode = ""
        try:
            mode = str(self.combo_mode.currentText()).strip()
        except Exception:
            pass

        # Em Produção, o método original pode concluir automaticamente a peça
        # antes de retornar; por isso a reserva precisa ser confirmada antes.
        if mode == "Modo Produção" and receiver is not None and hasattr(
            receiver,
            "confirm_reserved_image",
        ):
            receiver.confirm_reserved_image()

        result = original_process_aoi_images(
            self,
            sample_crop,
            ng_crop,
            aoi_info,
        )

        if mode != "Modo Produção" and receiver is not None and hasattr(
            receiver,
            "confirm_reserved_image",
        ):
            receiver.confirm_reserved_image()

        analysis = getattr(self, "current_analysis", None)
        if isinstance(analysis, dict):
            detail = analysis.setdefault("detail", {})
            detail["network_intake_validation"] = dict(audit)
            detail["network_intake_stable_required"] = 2
            detail["network_intake_source"] = "windows_xp"

        return result

    control_panel_cls.process_aoi_images = process_aoi_images
    control_panel_cls._network_aoi_intake_filter_installed = True


__all__ = [
    "MIN_FOCUS_SIDE",
    "install_network_aoi_intake_filter",
    "reject_invalid_network_capture",
    "validate_network_inspection",
]
