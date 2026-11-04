"""Prioridade de memória humana multilight sobre falso NG da CNN experimental.

A memória visual é uma contraprova supervisionada, não identidade RGB exata
nem autorização de saída 0/XP. A mesma função é usada em Teste, Produção e
Sombra, antes de a política do modo decidir sobre o XP.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite

from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.strict_category_memory import canonical_memory_category

LIGHTS = ("SIDE", "TOP", "MID")
SCHEMA = "visionx.operator_visual_memory_consensus.v1"
RULE = "operator_visual_memory_multilight_ok_v1"
MIN_HUMAN = {"SIDE": 0.98, "TOP": 0.98, "MID": 0.965}
MIN_HUMAN_MARGIN = 0.07
MIN_VISUAL_OK = 0.96
MIN_VISUAL_MARGIN = 0.07

# Rota complementar: duas luzes com memória HUMANA vinculada ao próprio
# arquivo OK verificado; terceira luz é contraprova direcional, e não
# precisa ter o mesmo recorte/iluminação da fonte humana mais próxima.
ANCHOR_HUMAN_MIN = 0.98
ANCHOR_MARGIN_MIN = 0.06
ANCHOR_ARCHIVE_MIN = 0.98
ANCHOR_ARCHIVE_MARGIN_MIN = 0.06
SUPPORT_HUMAN_MIN = 0.95
SUPPORT_ARCHIVE_MIN = 0.95
SUPPORT_MARGIN_MIN = 0.075


def _score(value):
    if type(value) not in (float, int):
        return None
    number = float(value)
    return number if isfinite(number) and 0 <= number <= 1 else None


def _no_hard_ng(fused: dict) -> bool:
    detail = fused.get("detail")
    if not isinstance(detail, dict):
        return False
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict):
        return False
    if any(
        row.get(key) is True
        for row in (fused, detail, trace)
        for key in ("hard_missing_evidence", "raw_hard_missing_evidence")
    ):
        return False
    physical = detail.get("physical_missing_veto")
    if isinstance(physical, dict) and (
        physical.get("status") in {"SUSPECT_ABSENCE", "NG_CORROBORATED", "UNAVAILABLE"}
        or physical.get("suspect_lights")
        or physical.get("unavailable_lights")
    ):
        return False
    return _score(detail.get("physical_score")) == 0.0


def _archive_human_anchor(local: dict, human_id: str, light: str) -> bool:
    """Conferência em mesma métrica: JSON humano + PNG OK da luz atual.

    A auditoria KNNArchiveAlignment já verifica hash e proveniência no
    índice persistente. Aqui revalidamos o contrato antes de reutilizá-lo.
    """
    knn = local.get("knn_c")
    aligned = knn.get("aligned_archive") if isinstance(knn, dict) else None
    if not isinstance(aligned, dict):
        return False
    source = aligned.get("archive_source")
    if not isinstance(source, dict):
        return False
    score = _score(aligned.get("archive_similarity"))
    link = _score(aligned.get("human_archive_similarity"))
    opposite = _score(aligned.get("best_opposite_archive_similarity"))
    return bool(
        aligned.get("schema") == "visionx.knn_c_archive_alignment.v1"
        and aligned.get("status") == "RECONCILED_OK"
        and aligned.get("reason") == "HUMAN_AND_ARCHIVE_ALIGNED"
        and aligned.get("diagnostic_only") is True
        and aligned.get("auto_decision_authorized") is False
        and aligned.get("category") == canonical_memory_category(
            local.get("cnn_v2_aoi_category")
        )
        and aligned.get("label") == "OK"
        and aligned.get("human_record_id") == human_id
        and source.get("record_id") == human_id
        and source.get("label") == "OK"
        and source.get("lighting_mode") == light
        and isinstance(source.get("source_sha256"), str)
        and len(source["source_sha256"]) == 64
        and isinstance(source.get("path"), str)
        and source["path"].startswith("public/ok_archive/")
        and ".." not in source["path"].split("/")
        and score is not None
        and link is not None
        and opposite is not None
        and score >= ANCHOR_ARCHIVE_MIN
        and link >= 0.97
        and score - opposite >= ANCHOR_ARCHIVE_MARGIN_MIN
    )


def memory_ok_counterevidence(
    analyses: dict, fused: dict, *, diagnostic: dict | None = None,
) -> dict | None:
    """Verifica as três iluminações e duas fontes visuais independentes.

    Não decide por quantidade de imagens na pasta nem por semelhança única.
    NG físico ou memória NG competitiva impede qualquer promoção.
    """
    def _skip(reason: str):
        if isinstance(diagnostic, dict):
            diagnostic.update({
                "schema": SCHEMA, "status": "NOT_APPLIED",
                "reason": reason, "diagnostic_only": True,
            })
        return None

    if not isinstance(fused, dict) or not isinstance(analyses, dict):
        return _skip("INVALID_INPUT")
    detail = fused.get("detail")
    if not isinstance(detail, dict):
        return _skip("MISSING_FINAL_DETAIL")
    category = canonical_memory_category(fused.get("multilight_category", ""))
    if not (
        uses_faltando_v2(category)
        and fused.get("multilight_final") is True
        and fused.get("verdict") == "DEFEITO REAL"
        and fused.get("is_defect") is True
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("cnn_v2_consensus") == "REVIEW"
        and detail.get("cnn_v2_consensus_reason") == "INVALID_LIGHT_EVIDENCE"
        # Pode existir só UMA CNN NG forte; as outras luzes ficam em
        # revisão. Exigir três NG fortes anulava o objetivo da memória.
        and bool(set(detail.get("multilight_strong_positive_modes") or ()) & set(LIGHTS))
        and set(detail.get("multilight_strong_positive_modes") or ()).issubset(
            set(LIGHTS)
        )
        and _no_hard_ng(fused)
    ):
        return _skip("BASE_CONTEXT_INELIGIBLE")
    report = detail.get("knn_visual_corroboration")
    if not isinstance(report, dict) or not (
        report.get("schema") == "visionx.knn_visual_corroboration.v1"
        and report.get("category") == category
        and report.get("status") == "NO_CONCLUSION"
        and report.get("strong_ng_lights") == []
        and report.get("trend_ng_lights") == []
    ):
        return _skip("UNTRUSTED_OR_CONTRADICTORY_VISUAL_REPORT")
    rows = report.get("lights")
    if not isinstance(rows, dict) or set(rows) != set(LIGHTS):
        return _skip("INCOMPLETE_VISUAL_LIGHTS")

    # A KNN C oferece uma terceira contraprova da SIDE. Nas outras luzes
    # seu voto pode estar em revisão, mas a busca independente deve
    # demonstrar margem OK consistente em cada luz.
    side = analyses.get("SIDE")
    side_detail = side.get("detail") if isinstance(side, dict) else None
    knn = side_detail.get("knn_c") if isinstance(side_detail, dict) else None
    if not isinstance(knn, dict) or not (
        knn.get("schema") == "visionx.knn_c_live_review_gated.v1"
        and knn.get("bank_status") == "READY"
        and knn.get("status") == "SUGERE_OK"
        and knn.get("reason") == "A_E_B_CONCORDAM"
        and knn.get("vote_A") == knn.get("vote_B") == "OK"
        and knn.get("suggestion_A") == knn.get("suggestion_B") == "SUGERE_OK"
        and knn.get("diagnostic_only") is True
        and knn.get("auto_decision_authorized") is False
    ):
        return _skip("SIDE_KNN_C_NOT_SUGGESTING_OK")
    for key, minimum in (("neighbors_A", 0.96), ("neighbors_B", 0.90)):
        neighbors = knn.get(key)
        if not isinstance(neighbors, list) or len(neighbors) < 5 or any(
            not isinstance(item, dict) or item.get("label") != "OK"
            for item in neighbors[:5]
        ):
            return _skip("SIDE_KNN_C_INCONSISTENT_TOP_FIVE")
        if (_score(neighbors[0].get("similarity")) or 0) < minimum:
            return _skip("SIDE_KNN_C_SIMILARITY_WEAK")

    observations = {}
    model_sha = None
    strict_lights = []
    aligned_anchors = []
    supporting_lights = []
    for light in LIGHTS:
        item = analyses.get(light)
        local = item.get("detail") if isinstance(item, dict) else None
        if not isinstance(local, dict) or not (
            item.get("lighting_mode") == light
            and local.get("cnn_v2_lighting_mode") == light
            and local.get("cnn_v2_checkpoint_verified") is True
            and local.get("cnn_v2_status") == "INFERENCE_OK"
            and local.get("recognition_route") == "NEW_CNN"
            and canonical_memory_category(local.get("cnn_v2_aoi_category")) == category
        ):
            return _skip("INVALID_CNN_LIGHT")
        # Veto físico em QUALQUER luz tem precedência: o frame dominante
        # pode não carregar todos os metadados locais de ausência.
        trace = local.get("decision_trace")
        for entry in (item, local, trace):
            if isinstance(entry, dict) and any(
                entry.get(field) is True
                for field in (
                    "hard_missing_evidence", "raw_hard_missing_evidence",
                    "missing_hard_absence", "memory_conflict",
                )
            ):
                return _skip("LOCAL_PHYSICAL_NG_OR_MEMORY_CONFLICT")
        physical = local.get("physical_missing_veto")
        if isinstance(physical, dict) and (
            physical.get("status") in {"SUSPECT_ABSENCE", "NG_CORROBORATED", "UNAVAILABLE"}
            or physical.get("suspect_lights")
            or physical.get("unavailable_lights")
        ):
            return _skip("LOCAL_PHYSICAL_ABSENCE_VETO")

        digest = local.get("cnn_v2_checkpoint_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            return _skip("MODEL_CHECKPOINT_UNVERIFIED")
        if model_sha is not None and digest != model_sha:
            return _skip("MODEL_CHECKPOINT_CHANGED")
        model_sha = digest

        human = local.get("confirmed_visual_recall")
        if not isinstance(human, dict) or not (
            human.get("schema") == "visionx.human_visual_recall.v1"
            and human.get("source") == "PERSISTENT_HUMAN_LABELS"
            and human.get("diagnostic_only") is True
            and human.get("auto_ok_authorized") is False
            and human.get("category") == category
            and human.get("status") in {"HUMAN_OK_VISUAL_CANDIDATE", "NO_MATCH"}
            and human.get("reason") == "VISUAL_SIMILARITY_NOT_EXACT_IDENTITY"
        ):
            return _skip("MISSING_OPERATOR_MEMORY")
        if light != "MID" and human["status"] != "HUMAN_OK_VISUAL_CANDIDATE":
            return _skip("MISSING_HUMAN_OK_CANDIDATE")
        source = human.get("best_ok_source")
        if not isinstance(source, dict) or not (
            source.get("source") == "OPERATOR_CONFIRMED"
            and source.get("label") == "OK"
            and source.get("lighting_mode") in LIGHTS
            and isinstance(source.get("record_id"), str)
            and len(source["record_id"]) == 64
        ):
            return _skip("UNTRUSTED_OPERATOR_SOURCE")
        same_light = source["lighting_mode"] == light
        score_ok = _score(human.get("best_ok_similarity"))
        score_ng = _score(human.get("best_ng_similarity"))
        if score_ok is None or score_ng is None:
            return _skip("INVALID_OPERATOR_SCORES")
        human_margin = score_ok - score_ng

        visual = rows[light]
        if not isinstance(visual, dict) or not (
            visual.get("status") == "NO_CONCLUSION"
            and visual.get("cross_category_ng") is False
            and visual.get("ng_support") in (0, 1)
        ):
            return _skip("UNTRUSTED_VISUAL_LIGHT")
        ok_summary = visual.get("ok")
        ng_summary = visual.get("ng")
        if not isinstance(ok_summary, dict) or not isinstance(ng_summary, dict):
            return _skip("INVALID_VISUAL_SUMMARIES")
        best_ok = _score(ok_summary.get("best_similarity"))
        best_ng = _score(ng_summary.get("best_similarity"))
        margin = visual.get("ng_margin_over_ok")
        if (
            best_ok is None or best_ng is None
            or type(margin) not in (float, int) or not isfinite(margin)
            or abs(float(margin) - (best_ng - best_ok)) > 0.00001
        ):
            return _skip("INVALID_VISUAL_MARGIN")
        archive_margin = best_ok - best_ng
        strict = bool(
            same_light
            and score_ok >= MIN_HUMAN[light]
            and human_margin >= MIN_HUMAN_MARGIN
            and best_ok >= MIN_VISUAL_OK
            and archive_margin >= MIN_VISUAL_MARGIN
        )
        if strict:
            strict_lights.append(light)

        # A evidência forte necessita vínculo de proveniência entre
        # registro humano e PNG OK arquivado na MESMA iluminação.
        anchored = bool(
            same_light
            and human["status"] == "HUMAN_OK_VISUAL_CANDIDATE"
            and score_ok >= ANCHOR_HUMAN_MIN
            and human_margin >= ANCHOR_MARGIN_MIN
            and best_ok >= ANCHOR_ARCHIVE_MIN
            and archive_margin >= ANCHOR_ARCHIVE_MARGIN_MIN
            and _archive_human_anchor(local, source["record_id"], light)
        )
        if anchored:
            aligned_anchors.append(light)
        supports = bool(
            anchored or (
                score_ok >= SUPPORT_HUMAN_MIN
                and human_margin >= SUPPORT_MARGIN_MIN
                and best_ok >= SUPPORT_ARCHIVE_MIN
                and archive_margin >= SUPPORT_MARGIN_MIN
            )
        )
        if supports:
            supporting_lights.append(light)
        observations[light] = {
            "human_ok": score_ok, "human_ng": score_ng,
            "human_record_id": source["record_id"],
            "human_source_lighting": source["lighting_mode"],
            "archive_ok": best_ok, "archive_ng": best_ng,
            "same_light_human": same_light,
            "aligned_human_archive_anchor": anchored,
        }

    # Dois contratos independentes: (A) três luzes estritas (histórico);
    # (B) duas âncoras alinhadas + terceira luz apenas apoiando OK.
    # O contrato B tolera iluminação cruzada na terceira luz, mas exige
    # validação de origem/arquivo nas outras duas. Nenhum deles libera XP.
    strict_consensus = len(strict_lights) == len(LIGHTS)
    anchored_consensus = (
        len(aligned_anchors) >= 2
        and len(supporting_lights) == len(LIGHTS)
    )
    if not strict_consensus and not anchored_consensus:
        if isinstance(diagnostic, dict):
            diagnostic.update({
                "strict_lights": list(strict_lights),
                "aligned_anchor_lights": list(aligned_anchors),
                "support_lights": list(supporting_lights),
                "human_archive_observations": observations,
            })
        return _skip("INSUFFICIENT_OPERATOR_ARCHIVE_ANCHORS_OR_SUPPORT")
    return {
        "schema": SCHEMA,
        "label": "OK",
        "category": category,
        "model_sha256": model_sha,
        "lights": observations,
        "source": "OPERATOR_CONFIRMED_PLUS_VISUAL_KNN",
        "evidence_mode": (
            "THREE_STRICT_LIGHTS" if strict_consensus
            else "TWO_ALIGNED_ARCHIVE_ANCHORS"
        ),
        "aligned_anchor_lights": list(aligned_anchors),
        "support_lights": list(supporting_lights),
        "rgb_exact_pair": False,
        "automatic_xp_authorized": False,
        "reason": "THREE_LIGHT_HUMAN_OK_WITH_KNN_COUNTEREVIDENCE",
    }


def apply_operator_visual_memory_consensus(
    fused: dict | None, analyses: dict | None,
) -> dict | None:
    """Prioridade do conhecimento supervisionado no veredito, sem liberar XP."""
    if not isinstance(fused, dict):
        return fused
    diagnostic = {}
    proof = memory_ok_counterevidence(analyses, fused, diagnostic=diagnostic)
    if proof is None:
        if diagnostic.get("reason") not in {
            "INVALID_INPUT", "MISSING_FINAL_DETAIL", "BASE_CONTEXT_INELIGIBLE",
        }:
            result = deepcopy(fused)
            detail = result.get("detail")
            if isinstance(detail, dict):
                detail["operator_visual_memory_attempt"] = diagnostic
            return result
        return fused
    result = deepcopy(fused)
    detail = result["detail"]
    detail["operator_visual_memory_consensus"] = proof
    detail["cnn_raw_verdict_before_memory"] = fused["verdict"]
    result["verdict"] = "FALHA FALSA"
    result["is_defect"] = False
    result["confidence"] = 0.5
    result["production_review_required"] = True
    result["eligible_for_final_decision"] = False
    evidence_description = (
        "duas âncoras humanas OK alinhadas a PNGs do acervo e terceira "
        "iluminação com suporte OK"
        if proof["evidence_mode"] == "TWO_ALIGNED_ARCHIVE_ANCHORS" else
        "três iluminações com memória humana OK estritamente corroborada"
    )
    result["reason"] = (
        "MEMÓRIA HUMANA MULTILIGHT: " + evidence_description
        + "; KNN C SIDE sugere OK. CNN experimental contradiz o histórico. "
        "Sem par RGB exato: XP exige confirmação humana; 0 automático bloqueado."
    )
    detail["fusion_rule"] = RULE
    detail["operator_review_required"] = True
    detail["eligible_for_final_decision"] = False
    detail["cnn_v2_supervised_auto_eligible"] = False
    detail["final_score"] = 0.5
    trace = detail.get("decision_trace")
    if isinstance(trace, dict):
        trace.update(
            verdict="FALHA FALSA", physical_defect=False,
            operator_review_required=True,
            fusion_rule=RULE, final_score=0.5,
        )
    return result


__all__ = [
    "SCHEMA", "RULE", "memory_ok_counterevidence",
    "apply_operator_visual_memory_consensus",
]
