"""Encerramento antecipado da captura multilight após SIDE confiável.

A confiança exibida é um score do motor, não uma probabilidade calibrada.
A política de Produção pode emitir OK/NG após a apresentação, se o gate
industrial revalidar o resultado. A confiança não é acurácia certificada.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite

from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
from src.core.neural.aoi_v3_epoch5_contract import EPOCH as V3_EPOCH_5
from src.core.neural.aoi_quality_cnn_v3_trial_consensus import _valid_vote
from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.strict_category_memory import canonical_memory_category


SIDE_EARLY_EXIT_THRESHOLD = 0.97
SIDE_EARLY_EXIT_RULE = "side_confidence_early_exit_v1"
SIDE_TRIAL_NG_REVIEW_RULE = "side_trial_ng_human_review_v1"
SIDE_TRIAL_OK_REVIEW_RULE = "side_trial_ok_human_review_v1"
SIDE_V3_SUPERVISED_RULE = "side_v3_epoch5_corroborated_test_review_v1"


def _uncorroborated_missing_ng_vs_two_knn_ok(analysis: dict, category: str) -> bool:
    """Do not finish trial SIDE when both KNN votes contradict CNN NG.

    KNN A/B review votes are weak diagnostic signals, NOT proof of OK.
    Em qualquer categoria, capturar TOP/MID antes de confirmar NG incerto.
    Never changes the CNN verdict, label, production policy or XP command.
    """
    if (analysis.get("is_defect") is not True
            or analysis.get("hard_missing_evidence") is True):
        return False
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return False
    if detail.get("hard_missing_evidence") is True:
        return False
    knn = detail.get("knn_c")
    return bool(
        isinstance(knn, dict)
        and knn.get("schema") == "visionx.knn_c_live_review_gated.v1"
        and knn.get("diagnostic_only") is True
        and knn.get("auto_decision_authorized") is False
        and knn.get("bank_status") == "READY"
        and knn.get("status") == "REVISAO"
        and knn.get("reason") == "SEM_SUGESTAO_CONFIAVEL"
        and knn.get("vote_A") == "OK"
        and knn.get("vote_B") == "OK"
        and knn.get("suggestion_A") == "REVISAO"
        and knn.get("suggestion_B") == "REVISAO"
    )



def _ng_with_stronger_nearest_ok(analysis: dict, category: str) -> bool:
    """Acquire TOP/MID when independent nearest OK neighbors contradict SIDE NG.

    This only prevents a premature 1/NG: similarity never authorizes 0/OK.
    It also preserves the immediate NG path for physical missing evidence.
    """
    if (analysis.get("is_defect") is not True
            or analysis.get("hard_missing_evidence") is True):
        return False
    detail = analysis.get("detail")
    if not isinstance(detail, dict) or detail.get("hard_missing_evidence") is True:
        return False
    physical = detail.get("physical_missing_veto")
    if isinstance(physical, dict) and physical.get("status") == "SUSPECT_ABSENCE":
        return False
    knn = detail.get("knn_c")
    if not (isinstance(knn, dict)
            and knn.get("schema") == "visionx.knn_c_live_review_gated.v1"
            and knn.get("bank_status") == "READY"
            and knn.get("diagnostic_only") is True
            and knn.get("auto_decision_authorized") is False):
        return False
    support = knn.get("reference_support")
    if not (isinstance(support, dict)
            and support.get("OK", 0) > 0 and support.get("NG", 0) > 0):
        return False
    for key, minimum_ok in (("neighbors_A", .90), ("neighbors_B", .85)):
        rows = knn.get(key)
        if (not isinstance(rows, list) or len(rows) < 5
                or any(not isinstance(row, dict)
                       or row.get("label") not in {"OK", "NG"}
                       for row in rows[:5])):
            return False
        try:
            first = rows[0]
            if first.get("label") != "OK":
                return False
            ok_score = float(first["similarity"])
            ng_scores = [
                float(row["similarity"]) for row in rows[1:5]
                if row.get("label") == "NG"
            ]
        except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
            return False
        if (not isfinite(ok_score) or ok_score < minimum_ok
                or not ng_scores or any(not isfinite(v) for v in ng_scores)
                or ok_score - max(ng_scores) < .04):
            return False
    return True


def _confirmed_human_ok_requires_other_lights(analysis: dict, category: str) -> bool:
    """A previous human OK requires corroboration before a new SIDE 1/NG.

    Only prevents premature rejection. Never authorizes 0/OK or changes
    the current CNN's NG label. Explicit physical missing evidence wins.
    """
    if (analysis.get("is_defect") is not True
            or analysis.get("hard_missing_evidence") is True):
        return False
    detail = analysis.get("detail")
    if not isinstance(detail, dict) or detail.get("hard_missing_evidence") is True:
        return False
    physical = detail.get("physical_missing_veto")
    if isinstance(physical, dict) and physical.get("status") == "SUSPECT_ABSENCE":
        return False
    visual = detail.get("confirmed_visual_recall")
    if not isinstance(visual, dict):
        return False
    return (
        visual.get("schema") == "visionx.human_visual_recall.v1"
        and visual.get("status") == "HUMAN_OK_VISUAL_CANDIDATE"
        and visual.get("diagnostic_only") is True
        and visual.get("auto_ok_authorized") is False
        and visual.get("source") == "PERSISTENT_HUMAN_LABELS"
    )


def _eligible_side_analysis(analysis: dict | None) -> bool:
    """Revalida a classificação SIDE sem depender de flags de UI ou Produção."""
    if not isinstance(analysis, dict):
        return False

    verdict = str(analysis.get("verdict", "") or "").strip().upper()
    if verdict not in {"DEFEITO REAL", "FALHA FALSA"}:
        return False
    if type(analysis.get("is_defect")) is not bool:
        return False
    if analysis["is_defect"] != (verdict == "DEFEITO REAL"):
        return False

    try:
        confidence = float(analysis.get("confidence"))
    except (TypeError, ValueError, OverflowError):
        return False
    if not isfinite(confidence) or not SIDE_EARLY_EXIT_THRESHOLD <= confidence <= 1.0:
        return False

    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return False
    trace = detail.get("decision_trace", {})
    if not isinstance(trace, dict):
        return False
    if (
        analysis.get("production_review_required") is True
        or detail.get("operator_review_required") is True
        or trace.get("operator_review_required") is True
        or detail.get("memory_conflict") is True
        or trace.get("memory_conflict") is True
        or detail.get("cnn_v3_primary") is True
    ):
        # V3 só pode encerrar SIDE pelo contrato verificado época 5+KNN C;
        # nunca pelo atalho genérico baseado apenas em confiança.
        return False

    route = str(detail.get("recognition_route", "") or "").strip().upper()
    if route == "MEMORY_CONFLICT":
        return False
    if route == "KNOWN_KNN" and not (
        detail.get("recognition_memory_verified") is True
        and detail.get("recognition_match") == "EXACT_PAIR"
        and detail.get("recognition_known_label") == (
            "NG" if analysis["is_defect"] else "OK"
        )
    ):
        return False

    if route == "NEW_CNN" or detail.get("cnn_v2_active") is True:
        if not (
            detail.get("cnn_v2_active") is True
            and detail.get("cnn_v2_status") == "INFERENCE_OK"
            and detail.get("cnn_v2_checkpoint_verified") is True
            and detail.get("cnn_v2_trial_manual_only") is not True
        ):
            return False

    # Nunca promover o checkpoint experimental explicitamente bloqueado.
    if detail.get("cnn_v2_checkpoint_sha256") == TRIAL_97P70_SHA256:
        return False
    return True


def _eligible_trial_side_for_review(
    analysis: dict | None, category: str,
) -> bool:
    """Verifica o resultado binário da SIDE experimental antes do encerramento.

    A exigência herdada de revisão é preservada para auditoria. Autorizar
    envio ao XP é responsabilidade exclusiva do gate de Produção.
    """
    # Categoria AOI indica o roteamento, não o tipo real do defeito.
    # O checkpoint é compartilhado por FALTANDO/INVERTIDO/EMBORCADO/
    # DESLOCADO; nunca limitar o encerramento à palavra "FALTANDO".
    if not isinstance(analysis, dict) or not uses_faltando_v2(category):
        return False
    category_scope = canonical_memory_category(category)
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return False
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict):
        return False
    if not (
        analysis.get("lighting_mode") == "SIDE"
        and analysis.get("verdict") in {"DEFEITO REAL", "FALHA FALSA"}
        and analysis.get("is_defect") is (analysis.get("verdict") == "DEFEITO REAL")
        and analysis.get("production_review_required") is True
        and detail.get("operator_review_required") is True
        and trace.get("operator_review_required") is True
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("cnn_v2_active") is True
        and detail.get("cnn_v2_experimental") is True
        and detail.get("cnn_v2_status") == "INFERENCE_OK"
        and detail.get("cnn_v2_checkpoint_verified") is True
        and detail.get("cnn_v2_trial_manual_only") is True
        and detail.get("cnn_v2_checkpoint_sha256") == TRIAL_97P70_SHA256
        and canonical_memory_category(detail.get("cnn_v2_aoi_category")) == category_scope
        and detail.get("cnn_v2_lighting_mode") == "SIDE"
        and trace.get("fusion_rule") == "cnn_v2_dualscale_direct"
        and detail.get("memory_conflict") is not True
        and trace.get("memory_conflict") is not True
    ):
        return False
    # A revisão herdada do checkpoint é tolerada; contraprovas independentes
    # e discordâncias permanecem impeditivas para 0/OK e 1/NG.
    defect = analysis["is_defect"]
    opposite_c = "SUGERE_OK" if defect else "SUGERE_NG"
    candidate = detail.get("knn_c")
    if isinstance(candidate, dict) and (
        candidate.get("status") == opposite_c
        or candidate.get("conflicts_with_existing_verdict") is True
    ):
        return False
    neighbors = detail.get("knn_visual_corroboration")
    if (not defect and isinstance(neighbors, dict)
            and neighbors.get("status") in {"NG_CORROBORATED", "CONFLICT"}):
        return False
    veto = detail.get("physical_missing_veto")
    if (not defect and isinstance(veto, dict)
            and veto.get("status") in {
                "SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW",
            }):
        return False
    if detail.get("multilight_physical_disagreement") is True:
        return False
    if detail.get("multilight_missing_verified_presence"):
        return False
    # Provenance: 26/10/2026 false NG (SIDE FALTANDO, CNN .99919,
    # KNN A/B vote OK but both still REVISAO). High, uncalibrated CNN
    # confidence cannot by itself skip the two independent lights.
    if _uncorroborated_missing_ng_vs_two_knn_ok(analysis, category):
        return False
    # Mesmo com maioria KNN NG, dois vizinhos OK visualmente mais próximos
    # exigem observar TOP/MID antes de uma rejeição XP irreversível.
    if _ng_with_stronger_nearest_ok(analysis, category):
        return False
    try:
        confidence = float(analysis["confidence"])
        ng_score = float(detail["cnn_v2_ng_score_uncalibrated"])
    except (KeyError, ValueError, TypeError, OverflowError):
        return False
    return bool(
        isfinite(confidence)
        and isfinite(ng_score)
        and SIDE_EARLY_EXIT_THRESHOLD <= confidence <= 1.0
        and 0.0 <= ng_score <= 1.0
        and (
            ng_score >= SIDE_EARLY_EXIT_THRESHOLD if defect
            else ng_score <= 1.0 - SIDE_EARLY_EXIT_THRESHOLD
        )
        and abs(confidence - (ng_score if defect else 1.0 - ng_score)) <= 0.000001
    )


def _nonselective_knn_ok_diagnostic(
    knn: dict, human_recall: dict | None,
) -> bool:
    """Corroboração fraca: A/B votam OK, mas C não autoriza sugestão.

    Distâncias baixas e vizinhos NG não viram 'par exato'. A KNN é
    diagnóstica: confere-se apenas que A e B apontam na mesma direção,
    as referências existem e nenhuma fonte humana indica NG mais próximo.
    """
    if not (
        knn.get("status") == "REVISAO"
        and knn.get("reason") == "SEM_SUGESTAO_CONFIAVEL"
        and knn.get("suggestion_A") == "REVISAO"
        and knn.get("suggestion_B") == "REVISAO"
        and knn.get("reason_A") == "SIMILARIDADE_VISUAL_INSUFICIENTE"
        and knn.get("reason_B") == "SIMILARIDADE_VISUAL_INSUFICIENTE"
        and knn.get("uses_aoi_category") is False
        and knn.get("requires_human_confirmation") is not True
        and knn.get("pre_c_review_required") is not True
        and knn.get("vote_A") == "OK"
        and knn.get("vote_B") == "OK"
    ):
        return False

    # A memória não é prova favorável, mas um NG humano mais próximo
    # proíbe o atalho. NO_MATCH é diferente de índice indisponível.
    if not isinstance(human_recall, dict) or not (
        human_recall.get("schema") == "visionx.human_visual_recall.v1"
        and human_recall.get("status") == "NO_MATCH"
        and human_recall.get("source") == "PERSISTENT_HUMAN_LABELS"
        and human_recall.get("diagnostic_only") is True
        and human_recall.get("auto_ok_authorized") is False
        and human_recall.get("cross_light_candidate") is False
    ):
        return False
    ok_score = human_recall.get("best_ok_similarity")
    ng_score = human_recall.get("best_ng_similarity")
    if (type(ok_score) not in (int, float)
            or type(ng_score) not in (int, float)
            or not (isfinite(ok_score) and isfinite(ng_score))
            or not (0 <= ng_score <= ok_score <= 1)):
        return False

    # Não exige unanimidade de top-5 nem limiar artificial de KNN;
    # exige top-1 OK em A/B, ordenação e proveniência de todas as fontes.
    for descriptor in ("neighbors_A", "neighbors_B"):
        rows = knn.get(descriptor)
        if not isinstance(rows, list) or len(rows) < 5:
            return False
        previous = 1.0
        for index, neighbor in enumerate(rows[:5]):
            if not isinstance(neighbor, dict) or neighbor.get("label") not in {"OK", "NG"}:
                return False
            score = neighbor.get("similarity")
            digest = neighbor.get("source_pixel_sha256")
            if (type(score) not in (float, int)
                    or not isfinite(score) or not 0 <= score <= previous
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest.lower())):
                return False
            if index == 0 and neighbor["label"] != "OK":
                return False
            previous = score
    return True


def _mixed_knn_verified_human_ok(
    knn: dict, visual: dict | None, *, category: str,
) -> bool:
    """Corrobora OK humano na SIDE apesar de vizinhos KNN mistos.

    Não altera a classe atribuída pela KNN nem interpreta proximidade
    como identidade exata. Exige vantagem OK contra NG em A/B e
    memória humana confirmada na mesma iluminação.
    """
    from src.core.confirmed_visual_recall import (
        MIN_OPPOSITE_MARGIN, qualified_human_ok_visual_candidate,
    )
    if not qualified_human_ok_visual_candidate(
        visual, lighting_mode="SIDE", category=category,
    ):
        return False
    if not (
        knn.get("status") == "REVISAO"
        and knn.get("reason") == "SEM_SUGESTAO_CONFIAVEL"
        and knn.get("reason_A") == "VIZINHOS_VISUAIS_DIVERGENTES"
        and knn.get("reason_B") == "VIZINHOS_VISUAIS_DIVERGENTES"
        and knn.get("suggestion_A") == "REVISAO"
        and knn.get("suggestion_B") == "REVISAO"
        and knn.get("vote_A") == "OK"
        and knn.get("vote_B") == "OK"
        and knn.get("uses_aoi_category") is False
        and knn.get("requires_human_confirmation") is not True
        and knn.get("pre_c_review_required") is not True
    ):
        return False
    for descriptor in ("neighbors_A", "neighbors_B"):
        rows = knn.get(descriptor)
        if not isinstance(rows, list) or len(rows) < 5:
            return False
        previous = 1.0
        max_ng = None
        for index, neighbor in enumerate(rows[:5]):
            if not isinstance(neighbor, dict) or neighbor.get("label") not in {"OK", "NG"}:
                return False
            score = neighbor.get("similarity")
            digest = neighbor.get("source_pixel_sha256")
            if (
                type(score) not in (float, int)
                or not isfinite(score) or not 0 <= score <= previous
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest.lower())
            ):
                return False
            if index == 0 and neighbor["label"] != "OK":
                return False
            if neighbor["label"] == "NG":
                max_ng = score if max_ng is None else max(max_ng, score)
            previous = score
        # A divergência da KNN só é tolerada se os NG no top-5 forem
        # efetivamente mais distantes do que o OK top-1 em AMBOS os eixos.
        if max_ng is None or rows[0]["similarity"] - max_ng < MIN_OPPOSITE_MARGIN:
            return False
    return True


def _eligible_v3_side_for_review(
    analysis: dict | None, category: str,
) -> str | None:
    """Veredito VISUAL SIDE V3 por evidências concordantes e verificáveis.

    A via usual exige V3 >=97%. A alternativa de memória humana OK
    comprovada exige concordância da V3 e KNN A/B com vantagem sobre NG,
    sem reduzir o limiar global ou autorizar comando XP.
    """
    if (not isinstance(analysis, dict)
            or analysis.get("lighting_mode") != "SIDE"
            or analysis.get("production_review_required") is True):
        return None
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return None
    trace = detail.get("decision_trace")
    if not isinstance(trace, dict):
        return None
    verdict = analysis.get("verdict")
    if verdict not in {"FALHA FALSA", "DEFEITO REAL"}:
        return None
    defect = verdict == "DEFEITO REAL"
    if (analysis.get("is_defect") is not defect
            or detail.get("recognition_route") != "NEW_CNN"
            or detail.get("cnn_v3_primary") is not True
            or detail.get("cnn_v3_status") != "INFERENCE_OK"
            or detail.get("cnn_v3_checkpoint_verified") is not True
            or detail.get("cnn_v3_lighting_mode") != "SIDE"
            or detail.get("fusion_rule") != "cnn_v3_primary_direct"
            or trace.get("fusion_rule") != "cnn_v3_primary_direct"
            or detail.get("cnn_v2_active") is True
            or detail.get("operator_review_required") is True
            or trace.get("operator_review_required") is True
            or detail.get("memory_conflict") is True
            or detail.get("raw_memory_conflict") is True
            or trace.get("memory_conflict") is True
            or detail.get("hard_missing_evidence") is True
            or analysis.get("hard_missing_evidence") is True):
        return None

    physical = detail.get("physical_missing_veto")
    if isinstance(physical, dict) and physical.get("status") in {
        "SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW", "NG_CORROBORATED",
    }:
        return None
    if (detail.get("capture_error") is True
            or detail.get("multilight_physical_disagreement") is True):
        return None
    vote = detail.get("aoi_v3_trial")
    if (not _valid_vote(vote, "SIDE")
            or vote.get("epoch") != V3_EPOCH_5
            or detail.get("cnn_v3_checkpoint_epoch") != V3_EPOCH_5
            or detail.get("cnn_v3_checkpoint_sha256") != vote["checkpoint_sha256"]
            or vote.get("predicted") != ("NG" if defect else "OK")):
        return None
    try:
        score = float(vote["ng_score_uncalibrated"])
        local_score = float(detail["cnn_v3_ng_score_uncalibrated"])
        confidence = float(analysis["confidence"])
        physical_score = float(detail["physical_score"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (not all(isfinite(v) for v in (score, local_score, confidence, physical_score))
            or abs(score - local_score) > 0.000001
            or abs(confidence - (score if defect else 1.0 - score)) > 0.000001
            or confidence < 0.0 or confidence > 1.0
            or physical_score != 0.0):
        return None

    knn = detail.get("knn_c")
    label = "NG" if defect else "OK"
    selected = "SUGERE_" + label
    if (not isinstance(knn, dict)
            or knn.get("schema") != "visionx.knn_c_live_review_gated.v1"
            or knn.get("bank_status") != "READY"
            or knn.get("diagnostic_only") is not True
            or knn.get("auto_decision_authorized") is not False
            or knn.get("conflicts_with_existing_verdict") is True
            or knn.get("vote_A") != label or knn.get("vote_B") != label):
        return None

    # A rota antiga de KNN seletiva continua independente da memória.
    selective = bool(
        knn.get("status") == selected
        and knn.get("requires_human_confirmation") is False
        and knn.get("suggestion_A") == selected
        and knn.get("suggestion_B") == selected
        and knn.get("pre_c_verdict") == verdict
        and knn.get("pre_c_is_defect") is defect
        and knn.get("pre_c_review_required") is False
    )

    # Sem sugestão seletiva, só o OK tem alternativa: o registro humano
    # confirmado precisa corroborar a CNN, com distância da classe NG.
    # Os rótulos de PNG continuam sendo apenas diagnósticos.
    corroborated_human_ok = False
    if (
        not defect
        and knn.get("status") == "REVISAO"
        and knn.get("reason") == "SEM_SUGESTAO_CONFIAVEL"
        and knn.get("suggestion_A") == "REVISAO"
        and knn.get("suggestion_B") == "REVISAO"
        and knn.get("reason_A") == "SIMILARIDADE_VISUAL_INSUFICIENTE"
        and knn.get("reason_B") == "SIMILARIDADE_VISUAL_INSUFICIENTE"
        and knn.get("uses_aoi_category") is False
        and knn.get("requires_human_confirmation") is not True
        and knn.get("pre_c_review_required") is not True
    ):
        from src.core.confirmed_visual_recall import (
            qualified_human_ok_visual_candidate,
        )
        corroborated_human_ok = qualified_human_ok_visual_candidate(
            detail.get("confirmed_visual_recall"),
            lighting_mode="SIDE", category=category,
        )
    nonselective_ok = (
        not defect and not selective and not corroborated_human_ok
        and _nonselective_knn_ok_diagnostic(
            knn, detail.get("confirmed_visual_recall"),
        )
    )
    # A exceção ao limiar do *score CNN* é estrutural e requer três
    # sinais alinhados: CNN OK, KNN A/B com OK mais próximo que NG,
    # e um registro humano OPERATOR_CONFIRMED robustamente recuperado.
    mixed_human_ok = (
        not defect and not selective and not corroborated_human_ok
        and _mixed_knn_verified_human_ok(
            knn, detail.get("confirmed_visual_recall"), category=category,
        )
    )
    if (confidence < SIDE_EARLY_EXIT_THRESHOLD and not mixed_human_ok):
        return None
    if not (selective or corroborated_human_ok or nonselective_ok or mixed_human_ok):
        return None

    if mixed_human_ok:
        return "HUMAN_OK_KNN_C_A_B_NEAREST_OK_MIXED"
    if nonselective_ok:
        return "CNN_V3_KNN_C_A_B_OK_NONSELECTIVE"

    # As duas rotas já existentes mantêm unanimidade de cinco vizinhos.
    for descriptor in ("neighbors_A", "neighbors_B"):
        neighbors = knn.get(descriptor)
        if not isinstance(neighbors, list) or len(neighbors) < 5:
            return None
        for neighbor in neighbors[:5]:
            if not isinstance(neighbor, dict) or neighbor.get("label") != label:
                return None
            try:
                similarity = float(neighbor["similarity"])
            except (KeyError, TypeError, ValueError, OverflowError):
                return None
            if not isfinite(similarity) or not 0.0 <= similarity <= 1.0:
                return None
            if corroborated_human_ok:
                digest = neighbor.get("source_pixel_sha256")
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest.lower())):
                    return None
    return (
        "KNN_C_A_B_TOP5_UNANIMOUS"
        if selective else "HUMAN_OK_KNN_C_A_B_TOP5_UNANIMOUS"
    )



def build_side_early_decision(
    analysis: dict | None, category: str, *, allow_supervised_v3: bool = False,
) -> dict | None:
    """Finaliza a inspeção pela SIDE; separa classificação de comando XP."""
    if (isinstance(analysis, dict)
            and _confirmed_human_ok_requires_other_lights(analysis, category)):
        # Uma fonte OK humana APROXIMADA precisa de TOP/MID antes de rejeitar.
        return None
    automatic = _eligible_side_analysis(analysis)
    trial_review = (
        not automatic and _eligible_trial_side_for_review(analysis, category)
    )
    v3_corroboration = (
        _eligible_v3_side_for_review(analysis, category)
        if allow_supervised_v3 is True and not automatic and not trial_review
        else None
    )
    v3_review = bool(v3_corroboration)
    if not automatic and not trial_review and not v3_review:
        return None
    review_only = trial_review

    verdict = str(analysis["verdict"]).strip().upper()
    confidence = float(analysis["confidence"])
    final = deepcopy(analysis)
    final_detail = final["detail"]
    final["reason"] = (
        f"SIDE: encerramento antecipado com confiança reportada de "
        f"{confidence:.1%}; TOP e MID dispensadas. "
        + ("V3 época 5 corroborada por KNN C A/B"
           + (" e memória humana OK" if v3_corroboration
              == "HUMAN_OK_KNN_C_A_B_TOP5_UNANIMOUS" else
              " e memória humana OK (top-1 acima de NG em A/B)"
              if v3_corroboration == "HUMAN_OK_KNN_C_A_B_NEAREST_OK_MIXED" else
              " não seletiva (memória sem match)" if v3_corroboration
              == "CNN_V3_KNN_C_A_B_OK_NONSELECTIVE" else "")
           + "; classificação visual em qualquer modo, sem autorização XP. "
           if v3_review else
           "Checkpoint experimental: autorização XP depende do gate de Produção. "
           if trial_review else "")
        + str(analysis.get("reason", "") or "")
    )
    final["lighting_mode"] = "SIDE"
    final["multilight_final"] = True
    final["multilight_category"] = str(category or "").strip().upper()
    final["eligible_for_final_decision"] = True
    # "Revisão obrigatória" reflete CONFLITO real do classificador, não
    # falta de permissão XP. A V3 concordante é veredito visual definitivo
    # neste ciclo, mas não concede aprovação 0/OK para Produção.
    final["production_review_required"] = review_only

    final_detail["multilight_final"] = True
    final_detail["multilight_category"] = final["multilight_category"]
    final_detail["eligible_for_final_decision"] = True
    final_detail["operator_review_required"] = review_only
    trace = final_detail.get("decision_trace")
    if isinstance(trace, dict):
        trace["operator_review_required"] = review_only
    final_detail["side_early_exit"] = {
        "rule": (
            SIDE_V3_SUPERVISED_RULE if v3_review else
            (SIDE_TRIAL_NG_REVIEW_RULE if final["is_defect"]
             else SIDE_TRIAL_OK_REVIEW_RULE) if trial_review
            else SIDE_EARLY_EXIT_RULE
        ),
        "threshold": SIDE_EARLY_EXIT_THRESHOLD,
        "reported_confidence": confidence,
        "verdict": verdict,
        "analyzed_lights": ["SIDE"],
        "skipped_lights": ["TOP", "MID"],
        "production_auto_allowed": automatic,
        "operator_review_required": review_only,
    }
    if v3_review:
        final_detail["side_early_exit"].update({
            "model": "aoi_quality_cnn_v3",
            "model_epoch": V3_EPOCH_5,
            "checkpoint_sha256": final_detail["cnn_v3_checkpoint_sha256"],
            "corroboration": v3_corroboration,
            "cnn_97pct_threshold_met": confidence >= SIDE_EARLY_EXIT_THRESHOLD,
            "scope": "ALL_MODES_VISUAL_ONLY",
        })
    return final


def side_trial_automatic_decision(analysis: dict | None) -> str | None:
    """Permite 0/OK ou 1/NG somente para a SIDE trial final íntegra.

    O checkpoint mantém seu histórico de erro; os resultados da CNN são
    escores não calibrados e não comprovam acerto em todas as peças.
    """
    if not isinstance(analysis, dict):
        return None
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return None
    marker = detail.get("side_early_exit")
    if not isinstance(marker, dict):
        return None
    category = analysis.get("multilight_category")
    if not _eligible_trial_side_for_review(analysis, category):
        return None
    try:
        reported = float(marker.get("reported_confidence"))
    except (ValueError, TypeError, OverflowError):
        return None
    expected_rule = (
        SIDE_TRIAL_NG_REVIEW_RULE if analysis["is_defect"]
        else SIDE_TRIAL_OK_REVIEW_RULE
    )
    if not (
        analysis.get("multilight_final") is True
        and analysis.get("eligible_for_final_decision") is True
        and detail.get("multilight_final") is True
        and detail.get("eligible_for_final_decision") is True
        and detail.get("multilight_category") == category
        and uses_faltando_v2(category)
        and canonical_memory_category(detail.get("cnn_v2_aoi_category"))
            == canonical_memory_category(category)
        and marker.get("rule") == expected_rule
        and marker.get("threshold") == SIDE_EARLY_EXIT_THRESHOLD
        and marker.get("verdict") == analysis.get("verdict")
        and marker.get("analyzed_lights") == ["SIDE"]
        and marker.get("skipped_lights") == ["TOP", "MID"]
        and marker.get("production_auto_allowed") is False
        and marker.get("operator_review_required") is True
        and isfinite(reported)
        and reported == float(analysis.get("confidence"))
    ):
        return None
    return "NG" if analysis["is_defect"] else "OK"


def side_automatic_decision(analysis: dict | None) -> str | None:
    """Retorna OK/NG somente para a SIDE final válida, nunca para um flag isolado.

    A política de Produção ainda executa os vetos e verifica o equipamento.
    """
    if not _eligible_side_analysis(analysis):
        return None

    detail = analysis["detail"]
    marker = detail.get("side_early_exit")
    if not isinstance(marker, dict):
        return None
    try:
        reported = float(marker.get("reported_confidence"))
    except (TypeError, ValueError, OverflowError):
        return None

    if not (
        analysis.get("lighting_mode") == "SIDE"
        and analysis.get("multilight_final") is True
        and analysis.get("eligible_for_final_decision") is True
        and analysis.get("production_review_required") is False
        and detail.get("multilight_final") is True
        and detail.get("eligible_for_final_decision") is True
        and detail.get("operator_review_required") is False
        and bool(analysis.get("multilight_category"))
        and analysis["multilight_category"] == detail.get("multilight_category")
        and marker.get("rule") == SIDE_EARLY_EXIT_RULE
        and marker.get("threshold") == SIDE_EARLY_EXIT_THRESHOLD
        and isfinite(reported)
        and reported == float(analysis["confidence"])
        and marker.get("verdict") == str(analysis["verdict"]).strip().upper()
        and marker.get("analyzed_lights") == ["SIDE"]
        and marker.get("skipped_lights") == ["TOP", "MID"]
        and marker.get("production_auto_allowed") is True
    ):
        return None

    return "NG" if analysis["is_defect"] else "OK"


__all__ = [
    "SIDE_EARLY_EXIT_THRESHOLD",
    "SIDE_EARLY_EXIT_RULE",
    "SIDE_TRIAL_NG_REVIEW_RULE",
    "SIDE_TRIAL_OK_REVIEW_RULE",
    "SIDE_V3_SUPERVISED_RULE",
    "build_side_early_decision",
    "side_automatic_decision",
    "side_trial_automatic_decision",
]
