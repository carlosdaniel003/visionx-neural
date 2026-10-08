"""Roteador operacional CNN v2 EXCLUSIVO para categoria FALTANDO.

O checkpoint foi auditado no replay de 117 PNGs conhecidos. Este módulo
NÃO garante generalização, NÃO usa KNN/memória e NÃO habilita AUTO-OK:
o gate produtivo exige revisão humana enquanto o modelo for experimental.

Instalar depois de todos os demais wrappers MoE em main.py.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from src.core.anomaly_memory_integration import is_missing_category

LIGHTS = ("SIDE", "TOP", "MID")
PINNED_RELATIVE_CHECKPOINT = (
    "reports/faltando_neural/models/"
    "experiment_v2_20261008T155311_256670Z/"
    "faltando_cnn_v2_candidate.pt"
)
PINNED_CHECKPOINT_SHA256 = (
    "6e4a31e8826d7b2afa18fbecb579a4d8979067713faa032d329f37f54729b599"
)
MODEL_KIND = "faltando_cnn_v2"
REVIEW_LOW_NG = .10
REVIEW_HIGH_NG = .90


def _fail_review(code: str, reason: str, mode: str) -> dict:
    """Nenhuma exceção ou ausência de pesos pode virar OK silencioso."""
    trace = {
        "schema": "visionx.decision.v1",
        "dominant_engine": MODEL_KIND,
        "fusion_rule": "cnn_v2_unavailable_review",
        "final_score": .5,
        "physical_score": 0.0,
        "cutoff": .5,
        "operator_review_required": True,
        "cnn_error_code": code,
        "weights": {"knn": 0.0, "physical": 0.0, "cnn": 1.0},
    }
    return {
        "is_defect": False,
        "confidence": .50,
        "verdict": "REVISÃO OBRIGATÓRIA",
        "reason": "CNN FALTANDO v2 indisponível: " + reason,
        "production_review_required": True,
        "active_engines": ["faltando_cnn_v2.py"],
        "bounding_box": None,
        "all_boxes": {},
        "lighting_mode": mode,
        "detail": {
            "cnn_v2_active": True,
            "cnn_v2_experimental": True,
            "cnn_v2_status": code,
            "cnn_v2_checkpoint_verified": False,
            "final_score": .5,
            "physical_score": 0.0,
            "fusion_rule": "cnn_v2_unavailable_review",
            "dominant_engine": MODEL_KIND,
            "operator_review_required": True,
            "decision_trace": trace,
        },
    }


class FaltandoCNNLive:
    """Prepara pares com a MESMA rotina usada no treinamento v2."""

    def __init__(
        self, checkpoint_path: Path | None = None,
        expected_sha256: str = PINNED_CHECKPOINT_SHA256,
    ):
        self.path = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else Path(__file__).resolve().parents[3] / PINNED_RELATIVE_CHECKPOINT
        )
        self.expected_sha256 = expected_sha256
        self._model = None
        self._metadata = {}
        self._error: tuple[str, str] | None = None
        self._online_pointer_seen = False

    def _refresh_live_pointer(self) -> None:
        """Ativa atomicamente pesos online SOMENTE após o gate de regressão."""
        from src.scripts.train_faltando_cnn_v2_online import POINTER_SCHEMA
        root = Path(__file__).resolve().parents[3]
        pointer = root / "reports" / "neural_online" / "live_active.json"
        if not pointer.exists():
            if self._online_pointer_seen:
                raise ValueError("Ponteiro online anteriormente ativo desapareceu")
            return
        if pointer.is_symlink():
            raise ValueError("Ponteiro online é link simbólico")
        row = json.loads(pointer.read_text(encoding="utf-8"))
        folder = (root / "reports" / "neural_online" / "checkpoints").resolve()
        candidate = (root / row["checkpoint_relative_path"]).resolve()
        digest = str(row.get("checkpoint_sha256", ""))
        if (row.get("schema") != POINTER_SCHEMA
                or row.get("category") != "FALTANDO"
                or row.get("experimental") is not True
                or row.get("production_approved") is not False
                or folder not in candidate.parents
                or candidate.suffix != ".pt"
                or len(digest) != 64):
            raise ValueError("Ponteiro online não passou no contrato de segurança")
        self._online_pointer_seen = True
        if candidate != self.path or digest != self.expected_sha256:
            self.path = candidate
            self.expected_sha256 = digest
            self._model = None
            self._metadata = {}
            self._error = None

    def _load(self):
        self._refresh_live_pointer()
        if self._model is not None:
            return self._model
        if self._error is not None:
            raise ValueError(self._error[1])
        try:
            import torch
            from src.core.neural.faltando_cnn_v2 import (
                FaltandoCNNV2, MODEL_SCHEMA_V2,
            )
            if not self.path.is_file() or self.path.is_symlink():
                raise FileNotFoundError("checkpoint treinado ausente na estação")
            if sha256(self.path.read_bytes()).hexdigest() != self.expected_sha256:
                raise ValueError("hash do checkpoint não corresponde ao modelo aprovado no replay")
            state = torch.load(self.path, map_location="cpu", weights_only=True)
            if (
                not isinstance(state, dict)
                or state.get("schema") != MODEL_SCHEMA_V2
                or state.get("experimental") is not True
                or state.get("production_approved") is not False
                or tuple(state.get("lights", ())) != LIGHTS
                or not isinstance(state.get("image_size"), int)
                or state["image_size"] < 64
                or state["image_size"] > 512
                or state["image_size"] % 32
                or not isinstance(state.get("focus_fraction"), (int, float))
                or not .5 <= state["focus_fraction"] <= .90
            ):
                raise ValueError("metadados do checkpoint v2 incompatíveis")
            model = FaltandoCNNV2()
            model.load_state_dict(state["state_dict"], strict=True)
            model.eval()
            self._metadata = {
                "image_size": state["image_size"],
                "focus_fraction": float(state["focus_fraction"]),
                "best_epoch": state.get("best_epoch"),
                "source_manifest_sha256": state.get("source_manifest_sha256"),
            }
            self._model = model
            return model
        except Exception as exc:
            # Fail-closed: qualquer erro de leitura/deserialização do modelo
            # deve resultar em revisão, não permitir ausência de decisão.
            self._error = (type(exc).__name__, str(exc))
            raise

    def inspect(self, reference: np.ndarray, test: np.ndarray, mode: str) -> dict:
        light = str(mode or "SIDE").strip().upper()
        if light not in LIGHTS:
            return _fail_review("INVALID_LIGHT", "iluminação fora de SIDE/TOP/MID", light)
        if (
            not isinstance(reference, np.ndarray)
            or not isinstance(test, np.ndarray)
            or reference.ndim != 3
            or test.ndim != 3
            or reference.shape[-1] != 3
            or test.shape[-1] != 3
            or min(reference.shape[:2]) < 12
            or min(test.shape[:2]) < 12
        ):
            return _fail_review("INVALID_PAIR", "par de gabarito/teste inválido", light)
        try:
            model = self._load()
            import torch
            from src.scripts.train_faltando_cnn_v2 import (
                _focus_crop, _letterbox_rgb,
            )
            size, fraction = (
                self._metadata["image_size"],
                self._metadata["focus_fraction"],
            )
            inputs = [torch.zeros((1, 3, 3, size, size)) for _ in range(4)]
            offset = LIGHTS.index(light)
            inputs[0][0, offset] = _letterbox_rgb(reference, size)
            inputs[1][0, offset] = _letterbox_rgb(test, size)
            inputs[2][0, offset] = _letterbox_rgb(
                _focus_crop(reference, fraction), size
            )
            inputs[3][0, offset] = _letterbox_rgb(
                _focus_crop(test, fraction), size
            )
            mask = torch.zeros((1, 3), dtype=torch.float32)
            mask[0, offset] = 1.0
            with torch.inference_mode():
                score_logit, per_light = model(*inputs, mask)
                ng_score = float(torch.sigmoid(score_logit)[0].item())
            if not math.isfinite(ng_score):
                raise ValueError("CNN emitiu score não finito")
        except Exception as exc:
            # Falhas inesperadas do PyTorch/OpenCV também são revisão.
            return _fail_review(type(exc).__name__, str(exc), light)

        is_defect = ng_score >= .5
        review = REVIEW_LOW_NG < ng_score < REVIEW_HIGH_NG
        verdict = (
            "REVISÃO OBRIGATÓRIA" if review else
            "DEFEITO REAL" if is_defect else "FALHA FALSA"
        )
        trace = {
            "schema": "visionx.decision.v1",
            "dominant_engine": MODEL_KIND,
            "fusion_rule": "cnn_v2_dualscale_direct",
            "final_score": ng_score,
            "physical_score": 0.0,
            "cutoff": .5,
            "operator_review_required": review,
            "confidence": max(ng_score, 1.0-ng_score),
            "weights": {"knn": 0.0, "physical": 0.0, "cnn": 1.0},
            "cnn_ng_score_uncalibrated": ng_score,
            "cnn_lighting_mode": light,
        }
        return {
            "is_defect": bool(is_defect and not review),
            "verdict": verdict,
            "confidence": max(ng_score, 1.0-ng_score),
            "reason": (
                f"CNN FALTANDO v2: {verdict}, luz {light}; "
                f"score NG não calibrado={ng_score:.6f}; "
                "checkpoint verificado, KNN não consultado. "
                "Validação independente ainda pendente."
            ),
            "production_review_required": review,
            "active_engines": ["faltando_cnn_v2.py"],
            "bounding_box": None,
            "all_boxes": {},
            "lighting_mode": light,
            "detail": {
                "cnn_v2_active": True,
                "cnn_v2_experimental": True,
                "cnn_v2_status": "INFERENCE_OK",
                "cnn_v2_checkpoint_verified": True,
                "cnn_v2_checkpoint_sha256": self.expected_sha256,
                "cnn_v2_checkpoint_best_epoch": self._metadata.get("best_epoch"),
                "cnn_v2_ng_score_uncalibrated": ng_score,
                "cnn_v2_lighting_mode": light,
                "final_score": ng_score,
                "physical_score": 0.0,
                "decision_cutoff": .5,
                "dominant_engine": MODEL_KIND,
                "fusion_rule": "cnn_v2_dualscale_direct",
                "operator_review_required": review,
                "decision_trace": trace,
            },
        }


def install_faltando_cnn_live(orchestrator_cls, *, predictor=None) -> None:
    """Wrapper externo: não executar especialistas físicos nem KNN em FALTANDO."""
    if getattr(orchestrator_cls, "_faltando_cnn_live_installed", False):
        return

    previous_inspect = orchestrator_cls.inspect
    engine = predictor if predictor is not None else FaltandoCNNLive()

    def inspect(
        self, full_gab, full_test, raw_anomalies,
        aoi_info, global_box_info, aoi_epicenters,
    ):
        info = aoi_info if isinstance(aoi_info, dict) else {}
        if (
            not is_missing_category(info.get("category", ""))
            or info.get("_replay_without_memory", False)
            or type(self).__name__ == "PhysicalOnlyOrchestrator"
        ):
            return previous_inspect(
                self, full_gab, full_test, raw_anomalies,
                aoi_info, global_box_info, aoi_epicenters,
            )
        return engine.inspect(
            full_gab, full_test, info.get("lighting_mode") or "SIDE"
        )

    orchestrator_cls.inspect = inspect
    orchestrator_cls._faltando_cnn_live_installed = True


__all__ = [
    "FaltandoCNNLive", "install_faltando_cnn_live",
    "PINNED_CHECKPOINT_SHA256", "PINNED_RELATIVE_CHECKPOINT",
]
