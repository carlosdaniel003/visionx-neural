"""CNN V3 na estação: intervenção NG somente, por autorização local explícita.

A época 3 é experimental: 13/14 humanos no próprio treino, 3 NG
liberados na validação e teste independente NÃO medido nessa época.
Consequentemente a V3 não concede PRESS_0, não substitui memória
humana confirmada e não altera os pesos/ponteiros da CNN V2.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from threading import RLock

import numpy as np

# PyTorch e a arquitetura V3 são carregados somente quando o operador
# ativa o ponteiro local, preservando o startup legado da estação.

from src.core.neural.aoi_v3_trial_contract import (
    TRIAL_POINTER, POINTER_SCHEMA, SNAPSHOT_RELATIVE_PATH,
    SNAPSHOT_SHA256, SOURCE_MANIFEST_SHA256, INITIAL_V3_SHA256, LIGHTS,
)


def validate_trial_snapshot(root: Path) -> tuple[dict, Path]:
    """Confere a época 3 e o SHA bruto; não escreve nada nem autoriza 0/OK."""
    root = Path(root).resolve()
    path = root / SNAPSHOT_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise ValueError("Pesos V3 época 3 ausentes ou simbólicos")
    if sha256(path.read_bytes()).hexdigest() != SNAPSHOT_SHA256:
        raise ValueError("SHA da época 3 não corresponde à revisão humana")
    import torch
    from src.core.neural.aoi_quality_cnn_v3 import MODEL_SCHEMA_V3
    from src.services.aoi_v3_epoch_diagnostics import EPOCH_SCHEMA
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema") != EPOCH_SCHEMA
        or checkpoint.get("architecture_schema") != MODEL_SCHEMA_V3
        or checkpoint.get("epoch") != 3
        or checkpoint.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or checkpoint.get("previous_candidate_sha256") != INITIAL_V3_SHA256
        or checkpoint.get("image_size") != 96
        or checkpoint.get("focus_fraction") != .70
        or tuple(checkpoint.get("lights", ())) != LIGHTS
        or checkpoint.get("diagnostics_only") is not True
        or checkpoint.get("candidate_only") is not True
        or checkpoint.get("production_approved") is not False
        or checkpoint.get("automatic_xp_authorized") is not False
        or checkpoint.get("test_evaluated_for_this_epoch") is not False
        or checkpoint.get("selected_by_safe_gate") is not False
        or not isinstance(checkpoint.get("validation"), dict)
        or checkpoint["validation"].get("NG_false_OK") != 3
        or checkpoint["validation"].get("OK_false_NG") != 0
        or not isinstance(checkpoint.get("operator_training_metrics"), dict)
        or checkpoint["operator_training_metrics"].get("NG_correct") != 8
        or checkpoint["operator_training_metrics"].get("OK_correct") != 5
        or checkpoint["operator_training_metrics"].get("correct") != 13
        or not isinstance(
            checkpoint.get("model_weights_not_production_state_dict"), dict
        )
    ):
        raise ValueError("Época V3 não corresponde à evidência offline")
    return checkpoint, path


def trial_pointer_payload() -> dict:
    return {
        "schema": POINTER_SCHEMA,
        "enabled": True,
        "operator_authorized": True,
        "deployment_mode": "ng_veto_only",
        "checkpoint_relative_path": SNAPSHOT_RELATIVE_PATH,
        "checkpoint_sha256": SNAPSHOT_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "epoch": 3,
        "automatic_ok_from_v3_authorized": False,
        "production_weights_replaced": False,
    }


def _verified_enabled_pointer(root: Path) -> bool:
    pointer = root / TRIAL_POINTER
    if not pointer.exists() and not pointer.is_symlink():
        return False
    if pointer.is_symlink() or not pointer.is_file():
        raise ValueError("Ponteiro V3 inválido")
    doc = json.loads(pointer.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc != trial_pointer_payload():
        raise ValueError("Ponteiro V3 divergente do contrato NG-veto")
    return True


class AOIQualityV3ProductionTrial:
    """Executa apenas inferência visual de uma iluminação, não transmite XP."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._lock = RLock()
        self._model = None
        self._checkpoint_stat = None

    def enabled(self) -> bool:
        return _verified_enabled_pointer(self.root)

    def predict(self, reference: np.ndarray, test: np.ndarray, light: str) -> dict | None:
        # Ausência de ponteiro = comportamento anterior sem alteração.
        with self._lock:
            if not self.enabled():
                self._model = None
                self._checkpoint_stat = None
                return None
            mode = str(light or "").strip().upper()
            if mode not in LIGHTS:
                raise ValueError("Iluminação V3 inválida")
            for image in (reference, test):
                if (
                    not isinstance(image, np.ndarray)
                    or image.dtype != np.uint8
                    or image.ndim != 3 or image.shape[-1] != 3
                    or min(image.shape[:2]) < 12
                ):
                    raise ValueError("Par gabarito/teste inválido para CNN V3")
            path = self.root / SNAPSHOT_RELATIVE_PATH
            stat = path.stat()
            current_stat = (stat.st_mtime_ns, stat.st_size)
            if self._model is None or current_stat != self._checkpoint_stat:
                checkpoint, _ = validate_trial_snapshot(self.root)
                from src.core.neural.aoi_quality_cnn_v3 import AOIQualityCNNV3
                model = AOIQualityCNNV3()
                model.load_state_dict(
                    checkpoint["model_weights_not_production_state_dict"],
                    strict=True,
                )
                model.eval()
                self._model = model
                self._checkpoint_stat = current_stat
            import torch
            from src.core.neural.faltando_preprocessing import (
                _focus_crop, _letterbox_rgb,
            )
            inputs = [
                torch.zeros((1, 3, 3, 96, 96), dtype=torch.float32)
                for _ in range(4)
            ]
            index = LIGHTS.index(mode)
            inputs[0][0, index] = _letterbox_rgb(reference, 96)
            inputs[1][0, index] = _letterbox_rgb(test, 96)
            inputs[2][0, index] = _letterbox_rgb(_focus_crop(reference, .70), 96)
            inputs[3][0, index] = _letterbox_rgb(_focus_crop(test, .70), 96)
            mask = torch.zeros((1, 3), dtype=torch.float32)
            mask[0, index] = 1.
            with torch.inference_mode():
                event_logit, _ = self._model(*inputs, mask)
                score = float(torch.sigmoid(event_logit)[0].item())
            if not math.isfinite(score):
                raise ValueError("CNN V3 retornou score inválido")
            return {
                "schema": POINTER_SCHEMA,
                "status": "READY",
                "epoch": 3,
                "light": mode,
                "checkpoint_sha256": SNAPSHOT_SHA256,
                "ng_score_uncalibrated": score,
                "predicted": "NG" if score >= .5 else "OK",
                "ng_veto": score >= .5,
                "automatic_ok_from_v3_authorized": False,
                "training_replay_13_of_14": True,
                "blind_test_of_epoch_3": False,
            }


def install_aoi_quality_v3_production_trial(orchestrator_cls, *,
                                             predictor=None) -> None:
    """Inserir DEPOIS do roteador de memória e da KNN C.

    Eventos conhecidos mantêm o voto humano. Eventos novos recebem
    veto NG adicional à evidência legada, sem autorização automática OK.
    """
    if getattr(orchestrator_cls, "_aoi_v3_trial_installed", False):
        return
    previous_inspect = orchestrator_cls.inspect
    engine = predictor if predictor is not None else AOIQualityV3ProductionTrial(
        Path(__file__).resolve().parents[3]
    )

    def inspect(self, full_gab, full_test, raw_anomalies,
                aoi_info, global_box_info, aoi_epicenters):
        result = previous_inspect(
            self, full_gab, full_test, raw_anomalies, aoi_info,
            global_box_info, aoi_epicenters,
        )
        info = aoi_info if isinstance(aoi_info, dict) else {}
        if (
            info.get("_replay_without_memory")
            or type(self).__name__ == "PhysicalOnlyOrchestrator"
            or not isinstance(result, dict)
        ):
            return result
        detail = result.get("detail")
        if not isinstance(detail, dict):
            return result
        # Uma V3 já usada como motor principal não deve ser inferida duas vezes.
        if detail.get("cnn_v3_primary") is True:
            return result
        # Categoria AOI não filtra a CNN V3: ambas as rotas novas entram.
        if detail.get("recognition_route") not in {"NEW_CNN", "NEW_EXPERTS"}:
            return result
        try:
            decision = engine.predict(
                full_gab, full_test, info.get("lighting_mode") or "SIDE"
            )
        except Exception as exc:
            # Uma configuração/peso adulterado não pode virar OK.
            detail["aoi_v3_trial"] = {
                "status": "ERROR",
                "error": type(exc).__name__ + ": " + str(exc),
                "automatic_ok_from_v3_authorized": False,
            }
            return result
        if decision is None:
            return result
        detail["aoi_v3_trial"] = decision
        # Um NG da V3 é apenas contraprova: não permitir que o legado
        # promova o OK sem a política de veto de segurança.
        return result

    orchestrator_cls.inspect = inspect
    orchestrator_cls._aoi_v3_trial_installed = True
    orchestrator_cls._aoi_v3_trial_engine = engine


__all__ = [
    "AOIQualityV3ProductionTrial", "install_aoi_quality_v3_production_trial",
    "validate_trial_snapshot", "trial_pointer_payload", "TRIAL_POINTER",
    "SNAPSHOT_SHA256", "SNAPSHOT_RELATIVE_PATH", "POINTER_SCHEMA",
]
