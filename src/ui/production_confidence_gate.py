"""Política de intervenção humana do Modo Produção.

A versão atual não usa mais confiança mínima de 99% para autorizar OK
automático. O veredito final já renderizado é a referência:

- FALHA FALSA -> pode ser automatizado como 0/OK;
- DEFEITO REAL/NG -> intervenção humana;
- REVISÃO OBRIGATÓRIA -> intervenção humana.

A apresentação visual e o momento do envio automático pertencem ao
ProductionAutonomyController. Este módulo mantém apenas a trava operacional de
revisão e a compatibilidade com decisões 0/1 do operador.
"""

from __future__ import annotations

from typing import Any

from src.core.corroborated_ng_verdict import corroborated_ng_verdict
from src.core.knn_c_supervised_policy import supervised_knn_c_decision
from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
from src.core.side_early_decision import (
    SIDE_EARLY_EXIT_THRESHOLD,
    side_automatic_decision,
    side_trial_automatic_decision,
)


# Mantido por compatibilidade com imports antigos. Não participa mais da política.
PRODUCTION_AUTO_CONFIDENCE_THRESHOLD = None


def normalized_confidence(analysis: dict | None) -> float:
    """Mantém a confiança como telemetria, sem usá-la como gate de decisão."""
    try:
        value = float((analysis or {}).get("confidence", 0.0))
    except (TypeError, ValueError):
        value = 0.0
    if value != value:
        value = 0.0
    return max(0.0, min(1.0, value))


def _review_required(analysis: dict | None) -> bool:
    if not isinstance(analysis, dict):
        return True

    detail = analysis.get("detail", {})
    detail = detail if isinstance(detail, dict) else {}
    trace = detail.get("decision_trace", {})
    trace = trace if isinstance(trace, dict) else {}

    return bool(
        analysis.get("production_review_required", False)
        or detail.get("operator_review_required", False)
        or trace.get("operator_review_required", False)
    )


def _verdict(analysis: dict | None) -> str:
    if not isinstance(analysis, dict):
        return "REVISÃO OBRIGATÓRIA"

    verdict = str(analysis.get("verdict", "") or "").strip().upper()
    if verdict in {
        "FALHA FALSA",
        "DEFEITO REAL",
        "DEFEITO",
        "REVISÃO OBRIGATÓRIA",
    }:
        return "DEFEITO REAL" if verdict == "DEFEITO" else verdict

    if _review_required(analysis):
        return "REVISÃO OBRIGATÓRIA"

    if "is_defect" in analysis:
        return "DEFEITO REAL" if bool(analysis.get("is_defect")) else "FALHA FALSA"

    return "REVISÃO OBRIGATÓRIA"


def _base_production_decision_policy(analysis: dict | None) -> dict[str, Any]:
    """Preserve all independently validated legacy recognition contracts."""
    verdict = _verdict(analysis)
    confidence = normalized_confidence(analysis)

    detail = (analysis or {}).get("detail", {}) if isinstance(analysis, dict) else {}
    detail = detail if isinstance(detail, dict) else {}
    # Contraprova física independente: uma inconsistência ou ausência dos
    # pares visuais proíbe 0/OK até confirmação humana. Esse segundo gate
    # continua válido mesmo se uma tela intermediária mudar o veredito.
    # Histórico trial permanece registrado na auditoria. Uma exceção de
    # Produção explicitamente validada só se aplica à SIDE final com contrato
    # de inferência, SHA, direção OK/NG, memória e episódio íntegros.
    diagnostics = detail.get("cnn_v2_light_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    trial_used = bool(
        detail.get("cnn_v2_checkpoint_sha256") == TRIAL_97P70_SHA256
        or detail.get("cnn_v2_trial_manual_only") is True
        or any(
            isinstance(data, dict) and (
                data.get("checkpoint_sha256") == TRIAL_97P70_SHA256
                or data.get("trial_manual_only") is True
            ) for data in diagnostics.values()
        )
    )
    # Classificacao humana OK multilight: valida novamente o registro
    # OPERATOR_CONFIRMED, as luzes, a margem e cada sonda fisica.
    # O veto fisico preserva o OK visual, mas bloqueia o comando XP.
    from src.core.human_multilight_priority import (
        RULE as HUMAN_MULTILIGHT_RULE, verified_human_multilight_priority,
        MIN_HUMAN as HUMAN_MULTILIGHT_THRESHOLD,
    )
    if detail.get("fusion_rule") == HUMAN_MULTILIGHT_RULE:
        valid = verified_human_multilight_priority(analysis)
        if valid is None:
            return {
                "mode": "production", "confidence": confidence,
                "threshold": HUMAN_MULTILIGHT_THRESHOLD,
                "verdict": "REVISÃO OBRIGATÓRIA",
                "auto_allowed": False,
                "operator_review_required": True,
                "proposed_decision": "",
                "reason": "operator_human_multilight_invalid_contract",
                "operator_shortcuts": {"0": "OK", "1": "NG"},
            }
        auto = bool(
            valid is True and detail.get("production_continuous_auto") is True
        )
        return {
            "mode": "production", "confidence": confidence,
            "threshold": HUMAN_MULTILIGHT_THRESHOLD,
            "verdict": "FALHA FALSA",
            "auto_allowed": auto,
            "operator_review_required": not auto,
            "proposed_decision": "OK" if auto else "",
            "reason": (
                "operator_human_multilight_97pct_auto_ok"
                if auto else
                "operator_human_multilight_97pct_requires_operator"
            ),
            "operator_shortcuts": {"0": "OK", "1": "NG"},
            "unverified_physical_label": True,
        }

    # Memória humana visual precede a CNN somente sob concordância
    # independente A/B e contraprova física da captura ATUAL.
    # O controlador valida este contrato novamente imediatamente antes
    # de enviar qualquer comando XP; nenhuma decisão automática vira treino.
    from src.core.side_memory_first import memory_first_side_decision, MIN_KNN
    memory_decision = memory_first_side_decision(analysis)
    if memory_decision in {"OK", "NG"}:
        side_evidence = detail.get("side_memory_first", {})
        side_threshold = (
            side_evidence.get("min_similarity", MIN_KNN)
            if isinstance(side_evidence, dict) else MIN_KNN
        )
        return {
            "mode": "production", "confidence": confidence,
            "threshold": side_threshold,
            "verdict": "FALHA FALSA" if memory_decision == "OK" else "DEFEITO REAL",
            "auto_allowed": True,
            "operator_review_required": False,
            "proposed_decision": memory_decision,
            "reason": "side_operator_memory_first_97pct_auto_" + memory_decision.lower(),
            "operator_shortcuts": {"0": "OK", "1": "NG"},
            "unverified_physical_label": True,
        }

    # Reconhecimento visual humano excepcionalmente forte permanece
    # FALHA FALSA, mas a sonda física é veto absoluto de XP 0/OK.
    # Não deixar o fallback de Produção transformá-lo em NG PREVENTIVO.
    from src.core.side_memory_first import (
        verified_side_memory_visual_review,
        verified_side_supervised_probe_override,
    )
    if verified_side_memory_visual_review(analysis):
        # O opt-in e estabelecido pelo controlador somente ENTRE pecas.
        # Sem ele, o alerta fisico continua obrigando tecla humana 0/1.
        armed = bool(
            detail.get("production_continuous_auto") is True
            and detail.get("side_strong_ok_supervised_armed") is True
            and verified_side_supervised_probe_override(analysis)
        )
        return {
            "mode": "production", "confidence": confidence,
            "threshold": 0.99, "verdict": "FALHA FALSA",
            "auto_allowed": armed,
            "operator_review_required": not armed,
            "proposed_decision": "OK" if armed else "",
            "reason": (
                "side_strong_human_ok_supervised_probe_override_auto_ok"
                if armed else
                "side_strong_human_ok_physical_probe_requires_operator"
            ),
            "operator_shortcuts": {"0": "OK", "1": "NG"},
            "unverified_physical_label": True,
            "physical_veto_supervised_override": armed,
        }

    # Conhecimento visual SIDE aproximado pode economizar TOP/MID para a
    # conferência humana. Jamais equiparar isso a par exato nem a auto-OK.
    from src.core.side_recurrent_ok_review import side_recurrent_review_valid
    if side_recurrent_review_valid(analysis):
        return {
            "mode": "production",
            "confidence": confidence,
            "threshold": None,
            "verdict": "FALHA FALSA",
            "auto_allowed": False,
            "operator_review_required": True,
            "proposed_decision": "",
            "reason": "side_verified_archive_visual_requires_operator",
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }
    trial_side = side_trial_automatic_decision(analysis) if trial_used else None
    if trial_side in {"OK", "NG"}:
        return {
            "mode": "production", "confidence": confidence,
            "threshold": SIDE_EARLY_EXIT_THRESHOLD,
            "verdict": "FALHA FALSA" if trial_side == "OK" else "DEFEITO REAL",
            "auto_allowed": True, "operator_review_required": False,
            "proposed_decision": trial_side,
            "reason": "side_trial_verified_auto_" + trial_side.lower(),
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }
    if trial_used:
        # Uma referência OK de captura recorrente pode recuperar a
        # CLASSIFICAÇÃO do operador na interface, mas nunca transmitir 0.
        # Validar novamente a identidade da fonte e as evidências brutas;
        # o flag calculado pela fusão não basta para liberar a placa.
        from src.core.knn_visual_corroboration import (
            _operator_confirmed_ok_pattern_review, _recognized_recurrent_ok,
            _recognized_two_light_operator_ok, _mixed_light_ok_candidate,
            _operator_corrected_side_ok_recall,
            _recognized_recurrent_visual_only_20261029,
        )
        recalled = detail.get("recognized_recurrent_ok")
        pattern = detail.get("operator_confirmed_ok_pattern_review")
        counterevidence = detail.get("knn_ok_counterevidence")
        two_light = detail.get("recognized_two_light_operator_ok")
        mixed = detail.get("mixed_light_ok_candidate")
        corrected = detail.get("operator_corrected_archive_ok_recall")
        newer = detail.get("recognized_top_mid_ok_20261029")
        trace = detail.get("decision_trace")
        verified_recall = _recognized_recurrent_ok(detail, counterevidence)
        verified_pattern = _operator_confirmed_ok_pattern_review(
            detail, counterevidence,
        )
        valid_known_ok = bool(
            isinstance(recalled, dict)
            and verified_recall is not None
            and recalled == verified_recall
            and detail.get("fusion_rule")
            == "operator_confirmed_recurrent_ok_three_light_review_v1"
        )
        valid_pattern = bool(
            isinstance(pattern, dict)
            and verified_pattern is not None
            and pattern == verified_pattern
            and detail.get("fusion_rule")
            == "operator_confirmed_ok_pattern_three_light_review_v1"
        )
        verified_two_light = _recognized_two_light_operator_ok(
            {
                **analysis, "verdict": detail.get("cnn_ng_verdict_before_knn_review"),
                "is_defect": detail.get("cnn_ng_verdict_before_knn_review")
                == "DEFEITO REAL",
            } if isinstance(analysis, dict) else {},
            detail.get("knn_visual_corroboration"), "FALTANDO",
        )
        # Reavaliar o padrão TOP/MID histórico a partir dos votos e vetores.
        # O Modo Produção pode automatizar 0 sob gate reforçado, mas o marcador
        # da memória sozinho continua sem autorização de saída XP.
        verified_newer = _recognized_recurrent_visual_only_20261029(
            {
                **analysis,
                "verdict": detail.get("cnn_ng_verdict_before_knn_review"),
                "is_defect": (
                    detail.get("cnn_ng_verdict_before_knn_review")
                    == "DEFEITO REAL"
                ),
            } if isinstance(analysis, dict) else {},
            detail.get("knn_visual_corroboration"), "FALTANDO",
        )
        valid_newer = bool(
            isinstance(newer, dict)
            and verified_newer is not None
            and newer == verified_newer
            and detail.get("fusion_rule")
            == "operator_known_top_mid_ok_review_20261029_v1"
        )
        # Recompute using original CNN NG evidence rather than trusting
        # a final UI label; the helper accepts only raw NG multilight.
        valid_two_light = bool(
            isinstance(two_light, dict)
            and verified_two_light is not None
            and two_light == verified_two_light
            and detail.get("fusion_rule")
            == "operator_recurrent_ok_top_mid_review_v1"
        )
        validated_mixed = _mixed_light_ok_candidate(
            {
                **analysis,
                "verdict": detail.get("verdict_before_mixed_light_ok"),
                "is_defect": False,
                "detail": {**detail, "fusion_rule": "multilight_local_review"},
            } if isinstance(analysis, dict) else {},
            detail.get("knn_visual_corroboration"),
            "FALTANDO",
        )
        valid_mixed = bool(
            detail.get("verdict_before_mixed_light_ok") == "REVISÃO OBRIGATÓRIA"
            and detail.get("fusion_rule")
            == "mixed_light_knn_ok_cnn_review_manual_v1"
            and isinstance(mixed, dict)
            and validated_mixed is not None
            and mixed == validated_mixed
        )
        verified_corrected = _operator_corrected_side_ok_recall(
            {
                **analysis,
                "verdict": detail.get("verdict_before_corrected_archive_ok"),
                "is_defect": False,
                "detail": {**detail, "fusion_rule": "multilight_local_review"},
            } if isinstance(analysis, dict) else {},
            detail.get("knn_visual_corroboration"),
            "FALTANDO",
        )
        valid_corrected = bool(
            detail.get("verdict_before_corrected_archive_ok")
            == "REVISÃO OBRIGATÓRIA"
            and detail.get("fusion_rule")
            == "operator_corrected_archive_ok_three_light_v1"
            and isinstance(corrected, dict)
            and verified_corrected is not None
            and corrected == verified_corrected
        )
        if (
            verdict == "FALHA FALSA"
            and (valid_known_ok or valid_pattern or valid_two_light
                 or valid_mixed or valid_corrected or valid_newer)
            and detail.get("operator_review_required") is True
            and isinstance(analysis, dict)
            and analysis.get("production_review_required") is True
            and analysis.get("eligible_for_final_decision") is False
            and isinstance(trace, dict)
            and trace.get("operator_review_required") is True
            and trace.get("verdict") == "FALHA FALSA"
            and detail.get("physical_score") == 0.0
            and detail.get("cnn_v2_consensus") == "REVIEW"
        ):
            # Rotas de memória explicitamente autorizadas em Produção
            # contínua: 28/10 e o padrão TOP/MID confirmado em 29/10.
            # O contrato é recalculado ACIMA a partir das evidências brutas.
            # Outras recomendações de revisão continuam sem 0 automático.
            from src.core.historical_ok_mid_probe_exception import (
                historical_ok_mid_probe_disagreement,
            )
            # Somente a rota TOP/MID de fonte OK revalidada pode tolerar
            # a discordância conhecida de uma ROI MID em geometria com
            # aparagem mínima; a auditoria SUSPECT_ABSENCE NÃO é apagada.
            limited_mid_exception = bool(
                valid_newer
                and historical_ok_mid_probe_disagreement(
                    detail.get("physical_missing_veto")
                )
            )
            supervised_auto_ok = (
                detail.get("production_continuous_auto") is True
                and (valid_two_light or valid_mixed or valid_corrected or valid_newer)
                and not valid_known_ok and not valid_pattern
                and detail.get("hard_missing_evidence") is not True
                and analysis.get("hard_missing_evidence") is not True
                and detail.get("multilight_physical_disagreement") is False
                and detail.get("multilight_missing_verified_presence") is False
                and detail.get("physical_score") == 0.0
                and (not valid_corrected or (
                    isinstance(detail.get("physical_missing_veto"), dict)
                    and detail["physical_missing_veto"].get("status")
                    == "NO_STRONG_EVIDENCE"
                ))
                # A rota de 29/10 só vale com as três capturas NEW_CNN
                # válidas, nenhuma sonda física suspeita ou geometria
                # obrigatoriamente revisável e reconhecimento íntegro.
                and (not valid_newer or (
                    detail.get("multilight_category") == "FALTANDO"
                    and detail.get("cnn_v2_trial_manual_only") is True
                    and isinstance(detail.get("recognition_light_routes"), dict)
                    and detail["recognition_light_routes"] == {
                        "SIDE": "NEW_CNN", "TOP": "NEW_CNN", "MID": "NEW_CNN",
                    }
                    and isinstance(detail.get("physical_missing_veto"), dict)
                    and (
                        detail["physical_missing_veto"].get("status") not in {
                            "SUSPECT_ABSENCE", "UNAVAILABLE",
                            "NG_CORROBORATED", "GEOMETRY_REVIEW",
                        }
                        or limited_mid_exception
                    )
                ))
                and detail.get("cnn_v2_consensus_reason") == "INVALID_LIGHT_EVIDENCE"
                and detail.get("recognition_route") == "NEW_CNN"
                and analysis.get("multilight_final") is True
                and (not valid_newer or detail.get("multilight_final") is True)
            )
            live_veto = detail.get("physical_missing_veto")
            live_veto = live_veto if isinstance(live_veto, dict) else {}
            live_veto_status = live_veto.get("status")
            has_historical_physical_veto = bool(
                valid_newer
                and live_veto_status in {
                    "SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW",
                    "NG_CORROBORATED",
                }
            )
            return {
                "mode": "production",
                "confidence": confidence,
                "threshold": None,
                "verdict": "FALHA FALSA",
                "auto_allowed": bool(supervised_auto_ok),
                "operator_review_required": not supervised_auto_ok,
                "proposed_decision": "OK" if supervised_auto_ok else "",
                "reason": (
                    "supervised_operator_known_top_mid_ok_mid_probe_disagreement_auto_v1"
                    if supervised_auto_ok and valid_newer and limited_mid_exception
                    else "supervised_operator_known_top_mid_ok_20261029_auto_v1"
                    if supervised_auto_ok and valid_newer
                    else "supervised_operator_corrected_archive_ok_auto_v1"
                    if supervised_auto_ok and valid_corrected
                    else "supervised_recurrent_ok_top_mid_auto_v1"
                    if supervised_auto_ok and valid_two_light
                    else "supervised_mixed_light_ok_auto_v1"
                    if supervised_auto_ok and valid_mixed
                    else "operator_corrected_archive_ok_requires_operator"
                    if valid_corrected
                    else "operator_confirmed_recurring_ok_requires_operator"
                    if valid_known_ok
                    else "operator_recurrent_ok_top_mid_requires_operator"
                    if valid_two_light
                    else "mixed_light_knn_ok_requires_operator"
                    if valid_mixed
                    else "known_top_mid_ok_physical_veto_" + live_veto_status.lower()
                    if has_historical_physical_veto
                    else "operator_known_top_mid_ok_20261029_requires_operator"
                    if valid_newer
                    else "operator_confirmed_ok_pattern_requires_operator"
                ),
                "operator_shortcuts": {"0": "OK", "1": "NG"},
                "physical_probe_exception": (
                    "KNOWN_OK_TOP_MID_SINGLE_SCALE_MID_DISAGREEMENT"
                    if supervised_auto_ok and limited_mid_exception else ""
                ),
            }
        # Preserve the validated historical visual verdict (not XP release).
        # The physical veto still requires intervention and blocks 0/OK.
        # A forged historical marker is not sufficient: valid_newer above
        # is recomputed from KNN/CNN observations and pinned reference hashes.
        # Distinguish a verified visual memory from a failed physical veto.
        # A current MID absence suspicion cannot be cleared by merely
        # locating a historical TOP/MID OK neighbor.
        veto = detail.get("physical_missing_veto")
        veto_status = veto.get("status") if isinstance(veto, dict) else None
        historical_conflict = bool(
            valid_newer
            and veto_status in {
                "SUSPECT_ABSENCE", "GEOMETRY_REVIEW", "UNAVAILABLE",
                "NG_CORROBORATED",
            }
        )
        return {
            "mode": "production", "confidence": confidence, "threshold": None,
            "verdict": (
                "FALHA FALSA" if historical_conflict
                else verdict if verdict == "DEFEITO REAL"
                else "REVISÃO OBRIGATÓRIA"
            ),
            "auto_allowed": False, "operator_review_required": True,
            "proposed_decision": "" if historical_conflict else
                ("NG" if verdict == "DEFEITO REAL" else ""),
            "reason": (
                "known_top_mid_ok_physical_veto_" + veto_status.lower()
                if historical_conflict else
                "cnn_97p70_unqualified_operator_review_only"
            ),
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }
    physical_veto = detail.get("physical_missing_veto", {})
    physical_veto = physical_veto if isinstance(physical_veto, dict) else {}
    # A apresentação de NG corroborado e a emissão 1/NG são dimensões
    # diferentes: mesmo sem SUSPECT_ABSENCE a convergência KNN tricolor
    # pode classificar NG, mas NUNCA autoriza XP automático.
    if corroborated_ng_verdict(analysis):
        return {
            "mode": "production", "confidence": confidence,
            "threshold": None, "verdict": "DEFEITO REAL",
            "auto_allowed": False, "operator_review_required": True,
            "proposed_decision": "NG",
            "reason": "corroborated_ng_requires_human_confirmation",
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }
    physical_blocked = bool(
        detail.get("multilight_category") == "FALTANDO"
        and physical_veto.get("status") in {"SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW"}
    )
    # Um caminho futuro que esqueça de executar a sonda física também não
    # pode reutilizar três votos CNN OK como se tivessem contraprova.
    physical_required = bool(
        isinstance(analysis, dict)
        and analysis.get("multilight_final") is True
        and detail.get("multilight_category") == "FALTANDO"
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("cnn_v2_consensus") == "OK"
        and detail.get("cnn_v2_supervised_auto_eligible") is True
        and physical_veto.get("status") != "NO_STRONG_EVIDENCE"
    )
    if physical_blocked or physical_required:
        corroborated_ng = corroborated_ng_verdict(analysis)
        return {
            "mode": "production",
            "confidence": confidence,
            "threshold": None,
            "verdict": "DEFEITO REAL" if corroborated_ng
                       else "REVISÃO OBRIGATÓRIA",
            "auto_allowed": False,
            "operator_review_required": True,
            "proposed_decision": "NG" if corroborated_ng else "",
            "reason": (
                "corroborated_ng_requires_human_confirmation" if corroborated_ng
                else "cnn_faltando_independent_physical_veto" if physical_blocked
                else "cnn_faltando_physical_evidence_required"
            ),
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }

    # Divergência forte entre vizinhos NG e CNN OK veta AUTO-OK inclusive
    # se outra camada visual reescrever o veredito de apresentação.
    neighbors = detail.get("knn_visual_corroboration")
    neighbors = neighbors if isinstance(neighbors, dict) else {}
    if (
        neighbors.get("status") == "NG_CORROBORATED"
        and detail.get("cnn_v2_consensus") == "OK"
    ):
        return {
            "mode": "production", "confidence": confidence, "threshold": None,
            "verdict": "REVISÃO OBRIGATÓRIA",
            "auto_allowed": False, "operator_review_required": True,
            "proposed_decision": "",
            "reason": "cnn_knn_visual_ng_disagreement",
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }

    # Estratégia C: somente o operador pode armá-la por sessão.
    # Executa DEPOIS dos vetos físicos, checkpoint trial e corroboração NG,
    # e ANTES das políticas genéricas. O controlador revalida no envio.
    selected_c = supervised_knn_c_decision(analysis)
    if selected_c in {"OK", "NG"}:
        return {
            "mode": "production", "confidence": confidence,
            "threshold": None,
            "verdict": "FALHA FALSA" if selected_c == "OK" else "DEFEITO REAL",
            "auto_allowed": True,
            "operator_review_required": False,
            "proposed_decision": selected_c,
            "reason": "knn_c_supervised_three_light_" + selected_c.lower(),
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }

    # A SIDE antecipada é uma alternativa explícita ao consenso tricolor.
    # O contrato completo da SIDE deve ser validado novamente aqui; todos
    # os vetos operacionais anteriores continuam tendo precedência.
    early_decision = side_automatic_decision(analysis)
    if early_decision in {"OK", "NG"}:
        return {
            "mode": "production",
            "confidence": confidence,
            "threshold": SIDE_EARLY_EXIT_THRESHOLD,
            "verdict": verdict,
            "auto_allowed": True,
            "operator_review_required": False,
            "proposed_decision": early_decision,
            "reason": "side_97pct_auto_" + early_decision.lower(),
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }

    # Um marcador SIDE existente, porém inconsistente, nunca pode cair
    # no auto-OK legado nem ser reinterpretado como consenso tricolor.
    if "side_early_exit" in detail:
        return {
            "mode": "production",
            "confidence": confidence,
            "threshold": SIDE_EARLY_EXIT_THRESHOLD,
            "verdict": (
                "DEFEITO REAL" if verdict == "DEFEITO REAL"
                else "REVISÃO OBRIGATÓRIA"
            ),
            "auto_allowed": False,
            "operator_review_required": True,
            "proposed_decision": "NG" if verdict == "DEFEITO REAL" else "",
            "reason": "side_early_exit_invalid_contract",
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }

    # AUTO-OK ou AUTO-NG pela CNN experimental somente se TRÊS luzes
    # verificadas concordarem, com mesmo SHA de modelo e sem revisão.
    # Uma pontuação isolada, até extrema, não autoriza nenhum comando.
    multilight_routes = detail.get("recognition_light_routes", {})
    multilight_routes = (
        multilight_routes if isinstance(multilight_routes, dict) else {}
    )
    cnn_experimental = bool(
        (detail.get("cnn_v2_active") and detail.get("cnn_v2_experimental"))
        or any(route == "NEW_CNN" for route in multilight_routes.values())
        or detail.get("recognition_route") == "NEW_CNN"
    )
    consensus = detail.get("cnn_v2_consensus")
    votes = detail.get("cnn_v2_light_votes", {})
    verified_lights = detail.get("cnn_v2_verified_lights", [])
    verified_lights = verified_lights if isinstance(verified_lights, list) else []
    expected_vote = "OK" if verdict == "FALHA FALSA" else (
        "NG" if verdict == "DEFEITO REAL" else ""
    )
    cnn_auto = bool(
        cnn_experimental
        and isinstance(analysis, dict)
        and analysis.get("multilight_final") is True
        and analysis.get("eligible_for_final_decision") is True
        and detail.get("multilight_final") is True
        and detail.get("eligible_for_final_decision") is True
        and uses_faltando_v2(detail.get("multilight_category", ""))
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("cnn_v2_supervised_auto_eligible") is True
        and consensus == expected_vote
        and expected_vote in {"OK", "NG"}
        and isinstance(votes, dict)
        and set(votes) == {"SIDE", "TOP", "MID"}
        and all(votes[light] == expected_vote for light in votes)
        and set(verified_lights) == {"SIDE", "TOP", "MID"}
        and not _review_required(analysis)
    )
    if cnn_auto:
        proposed_decision = expected_vote
        auto_allowed = True
        operator_review_required = False
        reason = "faltando_cnn_v2_supervised_three_light_consensus"
    elif cnn_experimental:
        auto_allowed = False
        operator_review_required = True
        if verdict == "DEFEITO REAL":
            proposed_decision = "NG"
            reason = "faltando_cnn_v2_ng_requires_operator"
        else:
            proposed_decision = ""
            verdict = "REVISÃO OBRIGATÓRIA"
            reason = "faltando_cnn_v2_experimental_requires_operator"
    elif verdict == "FALHA FALSA" and not _review_required(analysis):
        proposed_decision = "OK"
        auto_allowed = True
        operator_review_required = False
        reason = "false_failure_auto_ok"
    elif verdict == "DEFEITO REAL":
        proposed_decision = "NG"
        auto_allowed = False
        operator_review_required = True
        reason = "real_defect_manual_review"
    else:
        proposed_decision = ""
        auto_allowed = False
        operator_review_required = True
        reason = "mandatory_manual_review"

    return {
        "mode": "production",
        "confidence": confidence,
        "threshold": None,
        "verdict": verdict,
        "auto_allowed": bool(auto_allowed),
        "operator_review_required": bool(operator_review_required),
        "proposed_decision": proposed_decision,
        "reason": reason,
        "operator_shortcuts": {"0": "OK", "1": "NG"},
    }


def _legacy_continuous_policy(analysis: dict, detail: dict, policy: dict) -> dict:
    """Compatibilidade de contratos históricos de Produção pré-binária.

    O controlador corrente arma production_binary_auto=True por peça;
    esta função só é usada por clientes antigos que armam apenas
    production_continuous_auto sem o novo contrato.
    """
    if policy.get("auto_allowed") is True:
        return policy
    # Um OK visual humano com veto fisico ou contrato invalido nunca
    # pode ser transformado em NG PREVENTIVO no fallback continuo.
    from src.core.human_multilight_priority import RULE as HUMAN_MULTILIGHT_RULE
    from src.core.side_memory_first import RULE as SIDE_MEMORY_RULE
    if detail.get("fusion_rule") in {HUMAN_MULTILIGHT_RULE, SIDE_MEMORY_RULE}:
        # Tanto o OK visual valido quanto a adulteracao/inconsistencia
        # do marcador SIDE ficam fora da rejeicao preventiva automatica.
        return policy
    # Um marcador histórico de OK de 29/10 não é prova suficiente para
    # rejeição preventiva genérica: em conflito, adulteração de hash,
    # captura incompleta ou falha de sonda, PARE SEM ENVIAR 0/1.
    # O auto-OK válido já teria retornado acima do gate-base.
    if detail.get("fusion_rule") == "operator_known_top_mid_ok_review_20261029_v1":
        return policy
    # Reconhecimento visual OK com geometria inconclusiva não deve
    # virar nem 0 automático nem 1 preventivo. Preservar a revisão humana.
    if (
        detail.get("fusion_rule") == "operator_corrected_archive_ok_three_light_v1"
        and policy.get("reason") == "operator_corrected_archive_ok_requires_operator"
    ):
        return policy
    if (
        analysis.get("multilight_final") is not True
        or detail.get("multilight_final") is not True
        or detail.get("multilight_physical_disagreement") is True
    ):
        return policy
    route_info = detail.get("recognition_light_routes")
    if not isinstance(route_info, dict) or set(route_info) != {"SIDE", "TOP", "MID"}:
        return policy
    if any(route_info[light] not in {"NEW_CNN", "NEW_EXPERTS", "MEMORY_EXACT"}
           for light in ("SIDE", "TOP", "MID")):
        return policy
    if detail.get("category_unknown") is True or detail.get("capture_error") is True:
        return policy
    # No uncertain observation is promoted to OK. A conservative reject
    # still needs a complete capture and validated routing from all lights.
    return {
        **policy,
        "verdict": (
            "DEFEITO REAL" if policy.get("verdict") == "DEFEITO REAL"
            else "NG PREVENTIVO"
        ),
        "auto_allowed": True,
        "operator_review_required": False,
        "proposed_decision": "NG",
        "reason": "production_uncertain_auto_ng_quarantine_v1",
        "unverified_physical_label": True,
        "source_verdict": policy.get("verdict", ""),
    }


def _completed_continuous_inspection(analysis: dict, detail: dict) -> bool:
    """Distingue julgamento completo de falha de aquisição/roteamento.

    Uma falha operacional não é uma terceira classe da placa e não
    justifica inventar uma tecla 0/1. SIDE antecipada só participa com
    contrato visual revalidado; as três luzes exigem rotas conhecidas.
    """
    if (
        detail.get("capture_error") is True
        or detail.get("category_unknown") is True
        or analysis.get("multilight_final") is not True
        or detail.get("multilight_final") is not True
    ):
        return False
    routes = detail.get("recognition_light_routes")
    if (
        isinstance(routes, dict)
        and set(routes) == {"SIDE", "TOP", "MID"}
        and all(routes[light] in {"NEW_CNN", "NEW_EXPERTS", "MEMORY_EXACT"}
                for light in ("SIDE", "TOP", "MID"))
    ):
        return True

    # Atalho de SIDE não é trinca incompleta: a análise tem seu próprio
    # contrato de evidência. Veto físico do OK impede 0 mas não obriga
    # escolher 0/1 manualmente no Modo Produção binário.
    from src.core.side_memory_first import (
        RULE as SIDE_MEMORY_RULE,
        verified_side_memory_visual_review,
    )
    return bool(
        detail.get("fusion_rule") == SIDE_MEMORY_RULE
        and analysis.get("lighting_mode") == "SIDE"
        and verified_side_memory_visual_review(analysis)
    )


def production_decision_policy(analysis: dict | None) -> dict[str, Any]:
    """Produção contínua: somente 0/OK ou 1/NG para inspeções completas.

    Primeiro respeita uma decisão automática OK/NG já fundamentada e
    revalidada. Na falta de prova suficiente para 0, a decisão binária
    é 1/NG com proveniência de incerteza explícita. Isso não demonstra
    defeito físico e jamais vira rótulo de treino. Falhas técnicas
    param o ciclo, sem pedir julgamento de qualidade ao operador.
    """
    policy = _base_production_decision_policy(analysis)
    if not isinstance(analysis, dict):
        return policy
    detail = analysis.get("detail")
    if not isinstance(detail, dict) or detail.get("production_continuous_auto") is not True:
        return policy
    if detail.get("production_binary_auto") is not True:
        return _legacy_continuous_policy(analysis, detail, policy)

    if policy.get("auto_allowed") is True and policy.get("proposed_decision") in {"OK", "NG"}:
        decision = policy["proposed_decision"]
        return {
            **policy,
            "verdict": "FALHA FALSA" if decision == "OK" else "DEFEITO REAL",
            "operator_review_required": False,
        }

    if not _completed_continuous_inspection(analysis, detail):
        return {
            **policy,
            "verdict": "",
            "auto_allowed": False,
            "operator_review_required": False,
            "proposed_decision": "",
            "reason": "production_binary_operational_capture_fault_v1",
            "operational_error": True,
            "source_verdict": policy.get("verdict", ""),
        }

    # Inspeção completa, porém não existe autorização independente de
    # OK. Escolha 1, sem terceiro veredito visual nem espera por 0/1
    # humano. Guarde a classificação anterior para auditoria porque
    # esta rejeição NÃO é prova de defeito físico.
    return {
        **policy,
        "verdict": "DEFEITO REAL",
        "auto_allowed": True,
        "operator_review_required": False,
        "proposed_decision": "NG",
        "reason": "production_binary_unverified_ng_v1",
        "unverified_physical_label": True,
        "source_verdict": policy.get("verdict", ""),
        "source_reason": policy.get("reason", ""),
    }


def _record_policy(panel, policy: dict, resolution: str = "pending") -> None:
    panel.production_review_policy = dict(policy)
    analysis = getattr(panel, "current_analysis", None)
    if isinstance(analysis, dict):
        detail = analysis.setdefault("detail", {})
        detail["production_decision_policy"] = {
            **policy,
            "resolution": str(resolution),
        }
        # Uma peça classificada NG também pode precisar do operador para
        # enviar 1 ao XP. O flag sinaliza ação PENDENTE, não a classe.
        analysis["production_review_required"] = bool(
            policy.get("operator_review_required", False)
            and resolution == "pending"
        )


def _clear_pending(panel) -> None:
    panel.production_review_pending = False
    analysis = getattr(panel, "current_analysis", None)
    if isinstance(analysis, dict):
        analysis["production_review_required"] = False


def _show_pending_message(panel) -> None:
    policy = getattr(panel, "production_review_policy", None)
    if not isinstance(policy, dict):
        policy = production_decision_policy(
            getattr(panel, "current_analysis", None)
        )

    verdict = str(policy.get("verdict", "") or "REVISÃO OBRIGATÓRIA")
    panel.update_brain_status(
        f"Intervenção necessária: {verdict}. "
        "Aguardando operador: 0=OK | 1=NG",
        True,
    )
    if hasattr(panel, "_operational_controls"):
        panel._operational_controls.sync(force=True)


def enter_production_review(
    panel,
    analysis: dict | None = None,
    *,
    verdict_override: str = "",
) -> dict[str, Any]:
    """Congela a peça e libera somente 0/1 humano."""
    target = analysis if isinstance(analysis, dict) else getattr(
        panel,
        "current_analysis",
        None,
    )
    policy = production_decision_policy(target)

    override = str(verdict_override or "").strip().upper()
    if override:
        if override not in {"DEFEITO REAL", "REVISÃO OBRIGATÓRIA"}:
            override = "REVISÃO OBRIGATÓRIA"
        policy = {
            **policy,
            "verdict": override,
            "auto_allowed": False,
            "operator_review_required": True,
            "proposed_decision": "NG" if override == "DEFEITO REAL" else "",
            "reason": "forced_manual_intervention",
        }

    panel.production_review_pending = True
    panel.is_locked = True
    _record_policy(panel, policy, resolution="pending")
    _show_pending_message(panel)
    return policy


def install_production_confidence_gate(control_panel_cls, presenter_cls) -> None:
    """Mantém a trava humana enquanto o controlador gerencia a autonomia."""
    if getattr(control_panel_cls, "_production_confidence_gate_installed", False):
        return

    from PyQt6.QtCore import Qt

    original_init = control_panel_cls.__init__
    original_start_monitoring = control_panel_cls.start_monitoring
    original_handle_network_image = control_panel_cls.handle_network_image
    original_skip_image = control_panel_cls.skip_image
    original_save_label = control_panel_cls.save_label
    original_handle_physical_keyboard = control_panel_cls.handle_physical_keyboard
    original_key_press = control_panel_cls.keyPressEvent
    original_presenter_sync = presenter_cls.sync

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.production_review_pending = False
        self.production_review_policy = None

    def wrapped_start_monitoring(self, *args, **kwargs):
        if getattr(self, "production_review_pending", False):
            _show_pending_message(self)
            return None
        return original_start_monitoring(self, *args, **kwargs)

    def wrapped_handle_network_image(self, *args, **kwargs):
        if getattr(self, "production_review_pending", False):
            _show_pending_message(self)
            return None
        return original_handle_network_image(self, *args, **kwargs)

    def wrapped_skip_image(self, *args, **kwargs):
        # A camada externa de ciclo autoriza explicitamente o descarte XP
        # somente depois de invalidar todos os callbacks automáticos.
        authorized = bool(getattr(self, "_production_discard_authorized", False))
        if getattr(self, "production_review_pending", False) and not authorized:
            _show_pending_message(self)
            return None
        if authorized:
            _clear_pending(self)
            self.production_review_policy = None
        return original_skip_image(self, *args, **kwargs)

    def wrapped_save_label(self, user_decision: str, source="button"):
        mode = self.combo_mode.currentText() if hasattr(self, "combo_mode") else ""
        is_production = str(mode).strip() == "Modo Produção"
        normalized_source = str(source or "").strip().lower()
        had_pending = bool(getattr(self, "production_review_pending", False))

        if is_production and normalized_source in {"auto", "production_auto"}:
            policy = production_decision_policy(
                getattr(self, "current_analysis", None)
            )
            controller = getattr(self, "production_autonomy_controller", None)
            side_supervised = (
                policy.get("reason")
                == "side_strong_human_ok_supervised_probe_override_auto_ok"
            )
            if not (
                policy["auto_allowed"]
                and str(user_decision).strip().upper() == policy["proposed_decision"]
                and not bool(getattr(controller, "paused", False))
                and (
                    not side_supervised
                    or (
                        controller is not None
                        and controller.side_strong_ok_supervised_armed is True
                        and isinstance(getattr(self, "current_analysis", None), dict)
                        and self.current_analysis.get("detail", {}).get(
                            "side_strong_ok_supervised_armed"
                        ) is True
                    )
                )
            ):
                analysis = getattr(self, "current_analysis", None)
                detail = analysis.get("detail", {}) if isinstance(analysis, dict) else {}
                if isinstance(detail, dict) and detail.get("production_continuous_auto") is True:
                    # Comando inválido ou ciclo inconsistente: erro
                    # operacional, nunca solicitar 0/1 humano.
                    notice = getattr(self, "update_brain_status", None)
                    if callable(notice):
                        notice(
                            "Comando XP automático bloqueado: política ou "
                            "captura inválida. Reinicie a inspeção.",
                            True,
                        )
                    return False
                enter_production_review(self, analysis)
                return None
            _record_policy(self, policy, resolution="automatic")

        elif is_production and had_pending:
            policy = production_decision_policy(
                getattr(self, "current_analysis", None)
            )
            resolution = (
                "operator_ok"
                if str(user_decision).upper() == "OK"
                else "operator_ng"
            )
            _record_policy(self, policy, resolution=resolution)

        _clear_pending(self)
        result = original_save_label(self, user_decision, source=source)

        if (
            is_production
            and had_pending
            and not bool(getattr(self, "is_locked", False))
        ):
            callback = getattr(
                self,
                "production_operator_decision_completed",
                None,
            )
            if callable(callback):
                callback(user_decision, source=source)

        return result

    def wrapped_handle_physical_keyboard(self, comando_xp: str):
        command = str(comando_xp or "").strip().upper()
        mode = self.combo_mode.currentText() if hasattr(self, "combo_mode") else ""
        is_production = str(mode).strip() == "Modo Produção"
        pending = bool(getattr(self, "production_review_pending", False))

        if command in {"0", "OK"}:
            if is_production and not pending:
                return None
            return self.save_label("OK", source="xp_keyboard")
        if command in {"1", "NG"}:
            if is_production and not pending:
                return None
            return self.save_label("NG", source="xp_keyboard")
        if pending:
            _show_pending_message(self)
            return None
        return original_handle_physical_keyboard(self, comando_xp)

    def wrapped_key_press(self, event):
        pending = bool(getattr(self, "production_review_pending", False))
        if pending:
            if event.key() == Qt.Key.Key_0 and self.btn_save_ok.isEnabled():
                self.save_label("OK", source="button")
                event.accept()
                return
            if event.key() == Qt.Key.Key_1 and self.btn_save_ng.isEnabled():
                self.save_label("NG", source="button")
                event.accept()
                return
            event.accept()
            _show_pending_message(self)
            return
        return original_key_press(self, event)

    def wrapped_presenter_sync(self, force: bool = False):
        original_presenter_sync(self, force=force)
        panel = self.panel
        mode = panel.combo_mode.currentText() if hasattr(panel, "combo_mode") else ""
        is_production = str(mode).strip() == "Modo Produção"
        pending = bool(getattr(panel, "production_review_pending", False))

        decision_buttons_visible = (not is_production) or pending
        panel.btn_save_ok.setVisible(decision_buttons_visible)
        panel.btn_save_ng.setVisible(decision_buttons_visible)

        if not (is_production and pending):
            return

        self._set_enabled(panel.btn_start, False)
        self._set_enabled(
            panel.btn_skip,
            bool(getattr(panel, "capture_cycle_active", False))
            and getattr(panel, "capture_cycle_source", None) == "network"
            and getattr(
                getattr(panel, "production_autonomy_controller", None),
                "state", "",
            ) != "emitting_auto_decision",
        )
        self._set_enabled(panel.btn_save_ok, True)
        self._set_enabled(panel.btn_save_ng, True)

        for button in (
            panel.btn_light_mid,
            panel.btn_light_side,
            panel.btn_light_top,
        ):
            self._set_enabled(button, False)

        if hasattr(panel, "btn_clear_dataset"):
            self._set_enabled(panel.btn_clear_dataset, False)

        self._set_text(
            panel.btn_start,
            "Captura congelada — aguardando operador",
        )
        self._set_text(panel.btn_save_ok, "0 - Aprovar como OK")
        self._set_text(panel.btn_save_ng, "1 - Confirmar defeito NG")

        policy = getattr(panel, "production_review_policy", {}) or {}
        verdict = str(
            policy.get("verdict", "REVISÃO OBRIGATÓRIA") or
            "REVISÃO OBRIGATÓRIA"
        )

        panel.lbl_operation_state.setText("INTERVENÇÃO NECESSÁRIA")
        panel.lbl_operation_state.setProperty("tone", "attention")
        self._refresh_style(panel.lbl_operation_state)
        panel.lbl_operation_hint.setText(
            f"{verdict}. Pressione 0 para OK ou 1 para NG."
        )
        panel.lbl_operation_actions.setText("2 AÇÕES DISPONÍVEIS")
        self.last_state_name = "production_review"

    control_panel_cls.__init__ = wrapped_init
    control_panel_cls.start_monitoring = wrapped_start_monitoring
    control_panel_cls.handle_network_image = wrapped_handle_network_image
    control_panel_cls.skip_image = wrapped_skip_image
    control_panel_cls.save_label = wrapped_save_label
    control_panel_cls.handle_physical_keyboard = wrapped_handle_physical_keyboard
    control_panel_cls.keyPressEvent = wrapped_key_press
    presenter_cls.sync = wrapped_presenter_sync
    control_panel_cls._production_confidence_gate_installed = True


__all__ = [
    "PRODUCTION_AUTO_CONFIDENCE_THRESHOLD",
    "enter_production_review",
    "install_production_confidence_gate",
    "normalized_confidence",
    "production_decision_policy",
]
