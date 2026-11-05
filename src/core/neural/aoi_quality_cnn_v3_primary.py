"""CNN V3 como classificador visual primário das categorias antes cobertas pela V2.

Não importa pesos V2, não reclassifica memória humana exata e não concede
PRESS_0 por score experimental. Um erro de integridade fecha a rota.
"""
from __future__ import annotations

from pathlib import Path

from src.core.neural.aoi_quality_cnn_v3_production_trial import AOIQualityV3ProductionTrial
from src.core.neural.aoi_quality_cnn_v3_trial_consensus import _valid_vote


ENGINE = "aoi_quality_cnn_v3"
LIGHTS = ("SIDE", "TOP", "MID")


def _unavailable(reason: str, light: str) -> dict:
    trace = {
        "schema": "visionx.decision.v1",
        "dominant_engine": ENGINE,
        "fusion_rule": "cnn_v3_primary_unavailable",
        "final_score": .5,
        "physical_score": 0.0,
        "operator_review_required": True,
    }
    return {
        "is_defect": False,
        "confidence": .5,
        "verdict": "REVISÃO OBRIGATÓRIA",
        "reason": "CNN V3 indisponível: " + reason,
        "production_review_required": True,
        "active_engines": ["aoi_quality_cnn_v3.py"],
        "bounding_box": None,
        "all_boxes": {},
        "lighting_mode": light,
        "detail": {
            "cnn_v3_primary": True,
            "cnn_v3_status": "ERROR",
            "cnn_v3_checkpoint_verified": False,
            "aoi_v3_trial": {
                "status": "ERROR",
                "automatic_ok_from_v3_authorized": False,
                "reason": reason,
            },
            "final_score": .5,
            "physical_score": 0.0,
            "fusion_rule": "cnn_v3_primary_unavailable",
            "dominant_engine": ENGINE,
            "operator_review_required": True,
            "decision_trace": trace,
        },
    }


class AOIQualityV3Primary:
    """Adapta a inferência V3 validada por SHA ao contrato de inspeção AOI.

    O score identifica um candidato visual OK/NG; não é probabilidade
    calibrada nem autorização para liberar uma peça na linha.
    """

    is_v3_primary = True

    def __init__(self, *, engine=None, root=None):
        self.engine = engine if engine is not None else AOIQualityV3ProductionTrial(
            Path(root).resolve() if root is not None else Path(__file__).resolve().parents[3]
        )

    def warmup(self) -> dict:
        try:
            if not self.engine.enabled():
                return {"state": "disabled", "reason": "Ponteiro V3 ausente"}
            # Preflight de checkpoint antes das capturas reais. O carregador
            # também reconfere o SHA na primeira inferência.
            root = getattr(self.engine, "root", None)
            if root is not None:
                self.engine.validate_active_snapshot()
            return {"state": "ready", "engine": ENGINE}
        except Exception as exc:
            return {"state": "error", "reason": type(exc).__name__ + ": " + str(exc)}

    def inspect(self, reference, test, mode: str) -> dict:
        light = str(mode or "SIDE").strip().upper()
        if light not in LIGHTS:
            return _unavailable("Iluminação fora de SIDE/TOP/MID", light)
        try:
            vote = self.engine.predict(reference, test, light)
        except Exception as exc:
            return _unavailable(type(exc).__name__ + ": " + str(exc), light)
        if vote is None:
            return _unavailable("Ativação local da V3 não encontrada", light)
        if not _valid_vote(vote, light):
            return _unavailable("Voto V3 não corresponde ao contrato verificado", light)

        ng_score = float(vote["ng_score_uncalibrated"])
        ng = vote["predicted"] == "NG"
        verdict = "DEFEITO REAL" if ng else "FALHA FALSA"
        # Visualmente binário para Teste/MSS, mas ainda experimental:
        # a autorização XP é decidida e revalidada pelo gate de Produção.
        trace = {
            "schema": "visionx.decision.v1",
            "dominant_engine": ENGINE,
            "fusion_rule": "cnn_v3_primary_direct",
            "final_score": ng_score,
            "physical_score": 0.0,
            "cutoff": .5,
            "operator_review_required": False,
            "confidence": max(ng_score, 1.0 - ng_score),
            "weights": {"knn": 0.0, "physical": 0.0, "cnn": 1.0},
            "cnn_ng_score_uncalibrated": ng_score,
            "cnn_lighting_mode": light,
        }
        return {
            "is_defect": ng,
            "confidence": max(ng_score, 1.0 - ng_score),
            "verdict": verdict,
            "reason": (
                f"CNN AOI V3 época {vote['epoch']} • {verdict}, luz {light}; "
                f"score NG não calibrado={ng_score:.6f}. "
                "Classificador visual principal; qualidade em peças novas "
                "não validada. Nenhum OK automático autorizado apenas pela V3."
            ),
            "production_review_required": False,
            "active_engines": ["aoi_quality_cnn_v3.py"],
            "bounding_box": None,
            "all_boxes": {},
            "lighting_mode": light,
            "detail": {
                "cnn_v3_primary": True,
                "cnn_v3_status": "INFERENCE_OK",
                "cnn_v3_checkpoint_verified": True,
                "cnn_v3_checkpoint_sha256": vote["checkpoint_sha256"],
                "cnn_v3_checkpoint_epoch": vote["epoch"],
                "cnn_v3_ng_score_uncalibrated": ng_score,
                "cnn_v3_lighting_mode": light,
                "cnn_v3_automatic_ok_authorized": False,
                "aoi_v3_trial": dict(vote),
                "final_score": ng_score,
                "physical_score": 0.0,
                "decision_cutoff": .5,
                "dominant_engine": ENGINE,
                "fusion_rule": "cnn_v3_primary_direct",
                "operator_review_required": False,
                "decision_trace": trace,
            },
        }


__all__ = ["AOIQualityV3Primary"]
