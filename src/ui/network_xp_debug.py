"""Diagnóstico copiável da última entrada recebida do Windows XP.

Este módulo é exclusivamente de observabilidade. Não participa da validação,
classificação, memória, confiança ou controle do ciclo de produção.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.services.capture_evidence import (
    capture_debug_event_id,
    capture_debug_record,
    capture_debug_source,
    capture_image_available,
    current_copy_image_snapshot,
)
from src.ui.branding import (
    DECISION_DEBUG_TITLE,
    LOCAL_CAPTURE_DEBUG_TITLE,
    XP_DEBUG_TITLE,
)
from src.services.network_xp_frame import (
    network_xp_frame_available,
    network_xp_frame_snapshot,
)


DEBUG_SCHEMA = "visionx.network_xp_debug.v1"


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def format_network_debug_report(record: dict | None) -> str:
    data = _json_safe(record if isinstance(record, dict) else {})
    source = str(data.get("source", "") or "").strip().lower()
    report_title = (
        LOCAL_CAPTURE_DEBUG_TITLE
        if source == "local_mss"
        else XP_DEBUG_TITLE
    )
    if not data:
        return (
            f"{XP_DEBUG_TITLE}\n"
            "Nenhuma captura possui diagnóstico registrado."
        )

    validation = data.get("validation", {})
    transport = data.get("transport", {})
    cycle = data.get("cycle", {})
    hints = validation.get("diagnostic_hints", []) or []

    lines = [
        report_title,
        "=" * 72,
        f"Schema: {data.get('schema', DEBUG_SCHEMA)}",
        f"Evento: {data.get('event_id', '-')}",
        f"Data/hora: {data.get('timestamp', '-')}",
        f"Origem: {'Captura local MSS' if source == 'local_mss' else 'Windows XP'}",
        f"IP de origem: {data.get('source_ip', '-')}",
        f"Etapa: {data.get('stage', '-')}",
        f"Modo: {data.get('mode', '-')}",
        "",
        "RESULTADO DA VALIDAÇÃO",
        "-" * 72,
        f"Válida: {validation.get('valid', '-')}",
        f"Motivo técnico: {validation.get('reason', '-')}",
        f"Mensagem: {data.get('validation_message', '-')}",
        f"Anomalias brutas: {validation.get('raw_anomaly_count', '-')}",
        f"Epicentros antigos: {validation.get('old_epicenter_count', '-')}",
        f"Epicentros finais: {validation.get('epicenter_count', validation.get('real_epicenter_count', '-'))}",
        f"Caixa global: {validation.get('global_box_info', '-')}",
        f"Caixa de foco: {validation.get('focus_box', '-')}",
        "",
        "TRANSPORTE / CICLO",
        "-" * 72,
        f"Imagem recebida: {transport.get('image', '-')}",
        f"Frames estáveis exigidos: {transport.get('stable_required_frames', '-')}",
        f"Gate aceitando imagens: {cycle.get('accepting_images', '-')}",
        f"Geração do gate: {cycle.get('generation', '-')}",
        f"Imagens ignoradas no gate: {cycle.get('ignored_images', '-')}",
    ]

    decision = data.get("decision", {})
    if isinstance(decision, dict) and decision:
        missing = decision.get("missing", {})
        inverted = decision.get("inverted", {})
        memory = decision.get("memory", {})
        missing = missing if isinstance(missing, dict) else {}
        inverted = inverted if isinstance(inverted, dict) else {}
        memory = memory if isinstance(memory, dict) else {}
        lines.extend(
            [
                "",
                DECISION_DEBUG_TITLE,
                "-" * 72,
                f"Categoria: {decision.get('category', '-')}",
                f"Veredito: {decision.get('verdict', '-')}",
                f"Defeito: {decision.get('is_defect', '-')}",
                f"Confiança: {decision.get('confidence', '-')}",
                f"Tempo de análise: {decision.get('analysis_time_seconds', '-')} s",
                f"Início do tempo: {decision.get('analysis_time_start_source', '-')}",
                f"Contrato do tempo: {decision.get('analysis_time_contract', '-')}",
                f"Score final: {decision.get('final_score', '-')}",
                f"Score físico: {decision.get('physical_score', '-')}",
                f"Regra de fusão: {decision.get('fusion_rule', '-')}",
                f"Motor dominante: {decision.get('dominant_engine', '-')}",
                f"Revisão obrigatória: {decision.get('operator_review_required', '-')}",
                f"Ausência física forte efetiva: {decision.get('hard_missing_evidence', '-')}",
                f"Ausência física forte bruta: {decision.get('raw_hard_missing_evidence', '-')}",
                f"Hard missing contradito por OK quase exato: {decision.get('hard_missing_contradicted_by_exact_ok', '-')}",
                f"Hard missing contradito por presença invariável + OK forte: {decision.get('hard_missing_contradicted_by_invariant_ok', '-')}",
                f"Missing score: {missing.get('missing_score', '-')}",
                f"Missing classe: {missing.get('missing_classification', '-')}",
                f"Missing cobertura: {missing.get('missing_changed_coverage', '-')}",
                f"Missing residual médio: {missing.get('missing_residual_mean', '-')}",
                f"Missing residual p90: {missing.get('missing_residual_p90', '-')}",
                f"Missing perda estrutural: {missing.get('missing_structure_loss', '-')}",
                f"Missing similaridade direta: {missing.get('missing_direct_similarity', '-')}",
                f"Missing perda de aparência: {missing.get('missing_appearance_loss', '-')}",
                f"Missing melhor similaridade próxima: {missing.get('missing_best_similarity', '-')}",
                f"Missing fundo exposto: {missing.get('missing_background_exposure', '-')}",
                f"Missing bordas incompatíveis: {missing.get('missing_edge_mismatch', '-')}",
                f"Corpo presente: {missing.get('missing_component_body_present', '-')}",
                f"Veto por corpo presente: {missing.get('missing_body_presence_veto', '-')}",
                f"Fonte da presença: {missing.get('missing_body_presence_source', '-')}",
                f"Caixa da presença: {missing.get('missing_body_presence_box', '-')}",
                f"Presença coarse similarity: {missing.get('missing_body_coarse_similarity', '-')}",
                f"Presença silhouette dice: {missing.get('missing_body_silhouette_dice', '-')}",
                f"Presença area ratio: {missing.get('missing_body_area_ratio', '-')}",
                f"Presença centroid shift: {missing.get('missing_body_centroid_shift', '-')}",
                f"Presença box width ratio: {missing.get('missing_body_box_width_ratio', '-')}",
                f"Presença box height ratio: {missing.get('missing_body_box_height_ratio', '-')}",
                f"Presença política: {missing.get('missing_body_presence_policy', '-')}",
                f"Presença motivo: {missing.get('missing_body_presence_reason', '-')}",
                f"Presença composta invariável: {missing.get('missing_invariant_occupancy_support', '-')}",
                f"Presença composta vetou hard missing: {missing.get('missing_invariant_occupancy_veto', '-')}",
                f"Presença composta motivo: {missing.get('missing_invariant_occupancy_reason', '-')}",
                f"Envelope global ativo: {missing.get('missing_global_envelope_active', '-')}",
                f"Envelope global suporta presença: {missing.get('missing_global_envelope_support', '-')}",
                f"Envelope global vetou hard missing: {missing.get('missing_global_envelope_veto', '-')}",
                f"Envelope global caixa: {missing.get('missing_global_envelope_box', '-')}",
                f"Envelope global perfil horizontal: {missing.get('missing_global_envelope_row_profile', '-')}",
                f"Envelope global perfil vertical: {missing.get('missing_global_envelope_col_profile', '-')}",
                f"Envelope global coarse similarity: {missing.get('missing_global_envelope_coarse_similarity', '-')}",
                f"Envelope global fundo exposto: {missing.get('missing_global_envelope_background_exposure', '-')}",
                f"Envelope global limiar massa escura: {missing.get('missing_global_envelope_dark_threshold', '-')}",
                f"Envelope global massa escura gabarito: {missing.get('missing_global_envelope_reference_dark_fraction', '-')}",
                f"Envelope global massa escura teste: {missing.get('missing_global_envelope_test_dark_fraction', '-')}",
                f"Envelope global retenção massa escura: {missing.get('missing_global_envelope_dark_retention', '-')}",
                f"Envelope global perfil invariável horizontal: {missing.get('missing_global_envelope_invariant_row_profile', '-')}",
                f"Envelope global perfil invariável vertical: {missing.get('missing_global_envelope_invariant_col_profile', '-')}",
                f"Envelope global suporte invariável: {missing.get('missing_global_envelope_invariant_support', '-')}",
                f"Envelope global motivo: {missing.get('missing_global_envelope_reason', '-')}",
                f"Missing hard reason: {missing.get('missing_hard_absence_reason', '-')}",
                f"Guarda transversal: {missing.get('missing_cross_category_guard', '-')}",
                f"Guarda política: {missing.get('missing_guard_policy', '-')}",
                f"Guarda categoria origem: {missing.get('missing_guard_source_category', '-')}",
                f"Guarda suporte físico: {missing.get('missing_guard_physical_support', '-')}",
                f"Dual-scale política: {missing.get('missing_dual_scale_policy', '-')}",
                f"Dual-scale ativo: {missing.get('missing_dual_scale_active', '-')}",
                f"Dual-scale executado: {missing.get('missing_dual_scale_triggered', '-')}",
                f"Dual-scale desacordo: {missing.get('missing_scale_disagreement', '-')}",
                f"Dual-scale razão local/global: {missing.get('missing_local_global_area_ratio', '-')}",
                f"Dual-scale caixa contexto: {missing.get('missing_context_box', '-')}",
                f"Dual-scale score contexto: {missing.get('missing_context_score', '-')}",
                f"Dual-scale cobertura contexto: {missing.get('missing_context_coverage', '-')}",
                f"Dual-scale residual contexto: {missing.get('missing_context_residual_mean', '-')}",
                f"Dual-scale perda estrutural: {missing.get('missing_context_structure_loss', '-')}",
                f"Dual-scale similaridade direta: {missing.get('missing_context_direct_similarity', '-')}",
                f"Dual-scale perda aparência: {missing.get('missing_context_appearance_loss', '-')}",
                f"Dual-scale melhor match próximo: {missing.get('missing_context_best_similarity', '-')}",
                f"Dual-scale hard absence: {missing.get('missing_context_hard_absence', '-')}",
                f"Dual-scale motivo: {missing.get('missing_context_hard_reason', '-')}",
                f"Dual-scale suporte físico: {missing.get('missing_context_physical_support', '-')}",
                f"FALTANDO footprint dedicado: {missing.get('missing_dedicated_footprint_absence', '-')}",
                f"FALTANDO footprint motivo: {missing.get('missing_dedicated_footprint_reason', '-')}",
                f"INVERTIDO ativo: {inverted.get('inverted_active', '-')}",
                f"INVERTIDO defeito bruto: {inverted.get('inverted_is_defect', '-')}",
                f"INVERTIDO score: {inverted.get('inverted_score', '-')}",
                f"INVERTIDO classe: {inverted.get('inverted_classification', '-')}",
                f"INVERTIDO retenção testemunha: {inverted.get('inverted_witness_retention', '-')}",
                f"INVERTIDO perda testemunha: {inverted.get('inverted_witness_loss', '-')}",
                f"INVERTIDO perda feature: {inverted.get('inverted_feature_loss', '-')}",
                f"INVERTIDO topologia: {inverted.get('inverted_topology_mismatch', '-')}",
                f"INVERTIDO orientação: {inverted.get('inverted_orientation_mismatch', '-')}",
                f"INVERTIDO face alternativa: {inverted.get('inverted_alternate_face_signal', '-')}",
                f"INVERTIDO transformação ganho: {inverted.get('inverted_transform_gain', '-')}",
                f"INVERTIDO melhor transformação: {inverted.get('inverted_best_transform', '-')}",
                f"INVERTIDO transformação similaridade: {inverted.get('inverted_best_transform_similarity', '-')}",
                f"INVERTIDO relocação ganho: {inverted.get('inverted_relocation_gain', '-')}",
                f"INVERTIDO razão ROI/global: {inverted.get('inverted_local_global_area_ratio', '-')}",
                f"INVERTIDO ROI pequena: {inverted.get('inverted_small_witness_roi', '-')}",
                f"INVERTIDO alta autoridade: {inverted.get('inverted_high_authority', '-')}",
                f"INVERTIDO corroborado: {inverted.get('inverted_corroborated', '-')}",
                f"INVERTIDO corroborador: {inverted.get('inverted_corroboration_reason', '-')}",
                f"KNN melhor rótulo: {memory.get('best_match_label', '-')}",
                f"KNN similaridade: {memory.get('best_similarity', '-')}",
                f"KNN conflito efetivo: {memory.get('memory_conflict', '-')}",
                f"KNN conflito bruto: {memory.get('raw_memory_conflict', '-')}",
                f"KNN revisão efetiva: {memory.get('operator_review_required', '-')}",
                f"KNN revisão bruta: {memory.get('raw_operator_review_required', '-')}",
                f"KNN suprimido por ausência física: {memory.get('suppressed_by_hard_missing', '-')}",
                f"Motivo final: {decision.get('reason', '-')}",
            ]
        )

    # Para o roteamento CNN/KNN exato, os antigos campos físicos ficam
    # honestamente ausentes; não poluir o diagnóstico com dezenas de '-'.
    if isinstance(decision, dict) and decision:
        cnn = decision.get("cnn_v2", {})
        cnn = cnn if isinstance(cnn, dict) else {}
        route = decision.get("recognition", {})
        route = route if isinstance(route, dict) else {}
        if cnn.get("active") or route.get("route") == "KNOWN_KNN":
            unused_prefixes = (
                "Missing ", "Dual-scale ", "Presença ", "Corpo presente:",
                "Veto por corpo", "Fonte da presença:", "Caixa da presença:",
                "Envelope global", "FALTANDO footprint", "INVERTIDO ",
                "Guarda ", "Hard missing ", "Ausência física forte",
            )
            lines = [
                item for item in lines
                if not item.startswith(unused_prefixes)
            ]
        if cnn.get("active"):
            score = cnn.get("ng_score_uncalibrated")
            score_text = f"{score:.6%}" if isinstance(score, (int, float)) else "N/D"
            per_light = cnn.get("light_diagnostics", {})
            per_light = per_light if isinstance(per_light, dict) else {}
            lines.extend([
                "", "REDE NEURAL CNN FALTANDO v2", "-" * 72,
                f"Categoria AOI: {cnn.get('aoi_category') or decision.get('category', '-')}",
                f"Roteamento: {cnn.get('route') or route.get('route', '-')}",
                f"Status inferência: {cnn.get('status') or 'ver luzes individuais'}",
                f"Checkpoint verificado: {cnn.get('checkpoint_verified', '-')}",
                f"SHA-256 checkpoint: {cnn.get('checkpoint_sha256') or '-'}",
                f"Melhor época do treinamento: {cnn.get('checkpoint_best_epoch', '-')}",
                f"Score NG local (não calibrado): {score_text}",
                f"Consenso CNN: {cnn.get('consensus') or 'não calculado'}",
                f"Motivo consenso: {cnn.get('consensus_reason') or 'não informado'}",
                f"Elegível auto 0/1 supervisionado: {cnn.get('auto_eligible', False)}",
                "Scores não são probabilidades calibradas; não indicam acurácia industrial.",
            ])
            for light in MULTILIGHT_DEBUG_ORDER:
                entry = per_light.get(light, {})
                entry = entry if isinstance(entry, dict) else {}
                value = entry.get("ng_score_uncalibrated")
                display = f"{value:.6%}" if isinstance(value, (float, int)) else "N/D"
                lines.append(
                    f"{light}: {entry.get('route', '-') or '-'} • "
                    f"{entry.get('verdict', '-') or '-'} • score NG {display} "
                    f"• checkpoint {'OK' if entry.get('checkpoint_verified') else 'não verificado'}"
                )
        if route.get("route"):
            lines.extend([
                "", "MEMÓRIA KNN (ROTA EFETIVA)", "-" * 72,
                f"Rota: {route.get('route', '-')}",
                f"Par exato verificado: {route.get('verified', False)}",
                f"Match: {route.get('match', '-')}",
                f"Rótulo humano recuperado: {route.get('known_label') or 'não aplicável'}",
                f"Motivo da busca: {route.get('reason') or '-'}",
                "Match não encontrado não significa similaridade 0%.",
            ])

    if hints:
        lines.extend(["", "INDÍCIOS DIAGNÓSTICOS", "-" * 72])
        for hint in hints:
            lines.append(f"- {hint}")

    lines.extend(
        [
            "",
            "REGISTRO COMPLETO (JSON)",
            "-" * 72,
            json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),
        ]
    )
    return "\n".join(lines)


def _debug_record(panel) -> dict:
    record = capture_debug_record(panel)
    if record:
        return record
    legacy = getattr(panel, "network_intake_last_validation", None)
    return dict(legacy) if isinstance(legacy, dict) else {}


MULTILIGHT_DEBUG_ORDER = ("SIDE", "TOP", "MID")


def _compact_debug_value(value: Any, *, list_limit: int = 40):
    """Mantém o debug técnico legível sem despejar matrizes/imagens inteiras."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "empty": True,
            }
        summary = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "mean": float(np.mean(value)),
        }
        return summary
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _compact_debug_value(item, list_limit=list_limit)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > list_limit:
            sample = [
                _compact_debug_value(item, list_limit=list_limit)
                for item in value[: min(5, len(value))]
            ]
            return {
                "type": type(value).__name__,
                "length": len(value),
                "sample": sample,
            }
        return [
            _compact_debug_value(item, list_limit=list_limit)
            for item in value
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _multilight_event_matches(panel) -> bool:
    event_id = str(capture_debug_event_id(panel) or "")
    last_event_id = str(
        getattr(panel, "adhesive_multilight_last_event_id", "") or ""
    )
    category = str(
        getattr(panel, "adhesive_multilight_last_category", "") or ""
    ).strip()
    return bool(
        event_id
        and last_event_id
        and event_id == last_event_id
        and category
        and category.upper() not in {"UNKNOWN", "SEM CATEGORIA"}
    )


def _multilight_counts(panel) -> tuple[int, int]:
    analyses = getattr(panel, "adhesive_multilight_last_analyses", {})
    frames = getattr(panel, "adhesive_multilight_last_source_frames", {})
    analysis_count = sum(
        1
        for mode in MULTILIGHT_DEBUG_ORDER
        if isinstance(analyses, dict) and isinstance(analyses.get(mode), dict)
    )
    frame_count = sum(
        1
        for mode in MULTILIGHT_DEBUG_ORDER
        if (
            isinstance(frames, dict)
            and isinstance(frames.get(mode), np.ndarray)
            and frames.get(mode).size > 0
        )
    )
    return analysis_count, frame_count


def format_multilight_debug_report(panel) -> str:
    """Acrescenta SIDE/TOP/MID ao debug do evento multilight atual."""
    if not _multilight_event_matches(panel):
        return ""

    category = str(
        getattr(panel, "adhesive_multilight_last_category", "") or ""
    ).strip()
    analyses = getattr(panel, "adhesive_multilight_last_analyses", {})
    if not isinstance(analyses, dict):
        analyses = {}

    final_analysis = getattr(
        panel,
        "adhesive_multilight_last_final_analysis",
        None,
    )
    final_analysis = (
        final_analysis if isinstance(final_analysis, dict) else {}
    )
    final_detail = final_analysis.get("detail", {})
    final_detail = final_detail if isinstance(final_detail, dict) else {}

    title = (
        "ANÁLISES MULTILIGHT - ADESIVO"
        if category.upper() == "MUITO ADESIVO"
        else f"ANÁLISES MULTILIGHT - {category or 'CATEGORIA'}"
    )
    lines = [
        "",
        title,
        "=" * 72,
        "Escopo: diagnóstico técnico independente por iluminação.",
        f"Categoria multilight: {category or '-'}",
        f"Evento multilight: {getattr(panel, 'adhesive_multilight_last_event_id', '-')}",
        "",
        "JULGAMENTO FINAL MULTILIGHT",
        "-" * 72,
    ]

    if final_analysis:
        lines.extend(
            [
                f"Veredito final: {final_analysis.get('verdict', '-')}",
                f"Defeito final: {final_analysis.get('is_defect', '-')}",
                f"Confiança final: {final_analysis.get('confidence', '-')}",
                f"Score final: {final_detail.get('final_score', '-')}",
                f"Score físico máximo: {final_detail.get('physical_score', '-')}",
                f"Regra de fusão: {final_detail.get('fusion_rule', '-')}",
                f"Iluminação dominante: {final_detail.get('multilight_dominant_mode', final_detail.get('adhesive_multilight_dominant_mode', '-'))}",
                f"Iluminações positivas: {final_detail.get('multilight_positive_modes', final_detail.get('adhesive_multilight_positive_modes', []))}",
                f"Iluminações fortes: {final_detail.get('multilight_strong_positive_modes', final_detail.get('adhesive_multilight_strong_auxiliary_modes', []))}",
                f"Iluminações em revisão: {final_detail.get('multilight_review_modes', [])}",
                f"Papel da memória KNN: {final_detail.get('multilight_memory_role', final_detail.get('adhesive_multilight_memory_role', '-'))}",
                f"Rota final: {final_detail.get('recognition_route', '-')}",
                f"Consenso CNN: {final_detail.get('cnn_v2_consensus', 'não calculado')}",
                f"Consenso CNN motivo: {final_detail.get('cnn_v2_consensus_reason', '-')}",
                f"Consenso CNN elegível automático: {final_detail.get('cnn_v2_supervised_auto_eligible', False)}",
                f"Motivo final: {final_analysis.get('reason', '-')}",
            ]
        )
    else:
        lines.append("Julgamento final: aguardando SIDE/TOP/MID.")

    for mode in MULTILIGHT_DEBUG_ORDER:
        analysis = analyses.get(mode)
        lines.extend(
            [
                "",
                f"ILUMINAÇÃO {mode}",
                "-" * 72,
            ]
        )
        if not isinstance(analysis, dict):
            lines.append("Análise disponível: False")
            continue

        detail = analysis.get("detail", {})
        detail = detail if isinstance(detail, dict) else {}
        active_engines = list(analysis.get("active_engines", []) or [])
        lines.extend(
            [
                "Análise disponível: True",
                f"Motores ativos: {active_engines}",
                f"Veredito local (não final multilight): {analysis.get('verdict', '-')}",
                f"Defeito local: {analysis.get('is_defect', '-')}",
                f"Confiança local: {analysis.get('confidence', '-')}",
                f"Score final local: {detail.get('final_score', '-')}",
                f"Score físico local: {detail.get('physical_score', '-')}",
                f"Perfil de adesivo: {detail.get('adhesive_detector_profile', '-')}",
                f"Iluminação do motor de adesivo: {detail.get('adhesive_lighting_mode', '-')}",
                f"Testemunha MID clara - cobertura: {detail.get('mid_bright_witness_coverage', '-')}",
                f"Testemunha MID clara - pico: {detail.get('mid_bright_witness_peak', '-')}",
                f"Testemunha MID clara - score: {detail.get('mid_bright_witness_score', '-')}",
                f"Regra de fusão local: {detail.get('fusion_rule', '-')}",
                f"Motor dominante local: {detail.get('dominant_engine', '-')}",
                f"Rota memória local: {detail.get('recognition_route', '-')}",
                f"CNN ativa: {detail.get('cnn_v2_active', False)}",
                f"CNN inferência: {detail.get('cnn_v2_status', 'não executada')}",
                f"CNN checkpoint verificado: {detail.get('cnn_v2_checkpoint_verified', '-')}",
                f"CNN checkpoint SHA-256: {detail.get('cnn_v2_checkpoint_sha256', '-')}",
                f"CNN score NG (não calibrado): {detail.get('cnn_v2_ng_score_uncalibrated', 'N/D')}",
                f"Motivo local: {analysis.get('reason', '-')}",
                "Elegível para resultado final multilight: False",
                "Detalhes técnicos compactos (JSON):",
                json.dumps(
                    _compact_debug_value(analysis),
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ]
        )

    return "\n".join(lines)


def _normalize_composite_frame(image: Any) -> np.ndarray | None:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None

    array = np.ascontiguousarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        import cv2

        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim != 3:
        return None
    if array.shape[2] == 3:
        return array.copy()
    if array.shape[2] == 4:
        import cv2

        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    return None


def build_multilight_composite(frames: dict | None) -> np.ndarray | None:
    """Junta SIDE/TOP/MID lado a lado, sem redimensionar nem sobrepor."""
    source = frames if isinstance(frames, dict) else {}
    normalized = []
    for mode in MULTILIGHT_DEBUG_ORDER:
        frame = _normalize_composite_frame(source.get(mode))
        if frame is None:
            return None
        normalized.append((mode, frame))

    header_height = 44
    separator_width = 8
    max_height = max(frame.shape[0] for _mode, frame in normalized)
    total_width = (
        sum(frame.shape[1] for _mode, frame in normalized)
        + separator_width * (len(normalized) - 1)
    )

    canvas = np.full(
        (header_height + max_height, total_width, 3),
        16,
        dtype=np.uint8,
    )

    import cv2

    x = 0
    for index, (mode, frame) in enumerate(normalized):
        height, width = frame.shape[:2]
        y = header_height + max(0, (max_height - height) // 2)
        canvas[y : y + height, x : x + width] = frame

        label = f"{mode}"
        cv2.putText(
            canvas,
            label,
            (x + 14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (245, 197, 24),
            2,
            cv2.LINE_AA,
        )

        x += width
        if index < len(normalized) - 1:
            canvas[:, x : x + separator_width] = 48
            x += separator_width

    return canvas


def multilight_copy_image_snapshot(panel) -> np.ndarray | None:
    if not _multilight_event_matches(panel):
        return None
    frames = getattr(panel, "adhesive_multilight_last_source_frames", {})
    return build_multilight_composite(frames)


def _legacy_xp_image_fallback_allowed(panel) -> bool:
    """Impede reutilizar frame XP quando o diagnóstico atual é local MSS."""
    return capture_debug_source(panel).strip().lower() != "local_mss"


def network_debug_image_available(panel) -> bool:
    """Aceita a evidência do evento atual sem misturar origens."""
    if _multilight_event_matches(panel):
        return multilight_copy_image_snapshot(panel) is not None
    if capture_image_available(panel):
        return True
    if not _legacy_xp_image_fallback_allowed(panel):
        return False
    return bool(network_xp_frame_available(panel))


def _set_button_feedback(button, copied_text: str, idle_text: str) -> None:
    if button is None:
        return

    from PyQt6.QtCore import QTimer

    try:
        button.setText(copied_text)
        button.setEnabled(True)

        def restore():
            try:
                button.setText(idle_text)
            except Exception:
                pass

        QTimer.singleShot(1600, restore)
    except Exception:
        pass


def copy_network_debug_to_clipboard(panel) -> bool:
    record = _debug_record(panel)
    if not isinstance(record, dict) or not record:
        return False

    from PyQt6.QtWidgets import QApplication

    report = format_network_debug_report(record)
    multilight_report = format_multilight_debug_report(panel)
    if multilight_report:
        report = report + "\n" + multilight_report

    QApplication.clipboard().setText(report)
    _set_button_feedback(
        getattr(panel, "btn_copy_network_debug", None),
        "Debug copiado",
        "Copiar debug",
    )
    return True


def _qimage_from_bgr(image: np.ndarray):
    from PyQt6.QtGui import QImage

    if not isinstance(image, np.ndarray) or image.size == 0:
        return None

    array = np.ascontiguousarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        height, width = array.shape
        qimage = QImage(
            array.data,
            width,
            height,
            int(array.strides[0]),
            QImage.Format.Format_Grayscale8,
        )
        return qimage.copy()

    if array.ndim != 3:
        return None

    height, width, channels = array.shape
    if channels == 3:
        qimage = QImage(
            array.data,
            width,
            height,
            int(array.strides[0]),
            QImage.Format.Format_BGR888,
        )
        return qimage.copy()

    if channels == 4:
        bgra = np.ascontiguousarray(array[:, :, :4])
        rgba = bgra[:, :, [2, 1, 0, 3]].copy()
        qimage = QImage(
            rgba.data,
            width,
            height,
            int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        )
        return qimage.copy()

    return None


def network_debug_image_snapshot(panel) -> np.ndarray | None:
    """Prévia e Copiar imagem usam o mesmo conteúdo do evento atual."""
    if _multilight_event_matches(panel):
        return multilight_copy_image_snapshot(panel)
    return current_copy_image_snapshot(panel)


def _sync_capture_image_preview(panel) -> None:
    """Atualiza somente a prévia visual, sem participar do ciclo produtivo."""
    preview = getattr(panel, "lbl_capture_evidence_preview", None)
    if preview is None:
        return

    if not bool(getattr(panel, "_inspection_images_visible", True)):
        try:
            if hasattr(preview, "clear_source_image"):
                preview.clear_source_image("Aguardando captura")
            else:
                preview.clear()
                preview.setText("Aguardando captura")
        except Exception:
            pass
        return

    image = network_debug_image_snapshot(panel)
    if image is None:
        try:
            if hasattr(preview, "clear_source_image"):
                preview.clear_source_image("Aguardando captura")
            else:
                preview.clear()
                preview.setText("Aguardando captura")
        except Exception:
            pass
        return

    qimage = _qimage_from_bgr(image)
    if qimage is None or qimage.isNull():
        return

    try:
        if hasattr(preview, "set_source_image"):
            preview.set_source_image(qimage)
        else:
            from PyQt6.QtGui import QPixmap

            preview.setPixmap(QPixmap.fromImage(qimage))
    except Exception:
        pass


def copy_network_image_to_clipboard(panel) -> bool:
    if _multilight_event_matches(panel):
        image = multilight_copy_image_snapshot(panel)
    else:
        image = network_debug_image_snapshot(panel)
    if image is None:
        return False

    qimage = _qimage_from_bgr(image)
    if qimage is None or qimage.isNull():
        return False

    from PyQt6.QtWidgets import QApplication

    QApplication.clipboard().setImage(qimage)
    composite = _multilight_event_matches(panel)
    _set_button_feedback(
        getattr(panel, "btn_copy_network_image", None),
        "3 imagens copiadas" if composite else "Imagem copiada",
        "Copiar imagens SIDE/TOP/MID" if composite else "Copiar imagem",
    )
    return True


def sync_network_debug_controls(panel) -> None:
    record = _debug_record(panel)
    debug_available = bool(isinstance(record, dict) and record)
    image_available = network_debug_image_available(panel)
    _sync_capture_image_preview(panel)

    debug_button = getattr(panel, "btn_copy_network_debug", None)
    image_button = getattr(panel, "btn_copy_network_image", None)
    state_label = getattr(panel, "lbl_network_debug_state", None)

    if debug_button is not None:
        try:
            debug_button.setEnabled(debug_available)
        except Exception:
            pass

    if image_button is not None:
        try:
            composite_ready = (
                _multilight_event_matches(panel)
                and multilight_copy_image_snapshot(panel) is not None
            )
            image_button.setText(
                "Copiar imagens SIDE/TOP/MID" if composite_ready
                else "Copiar imagem"
            )
            image_button.setToolTip(
                "Copia uma única imagem com os três frames completos SIDE/TOP/MID."
                if composite_ready else
                "Copia a captura do evento atual; para multilight, aguarde "
                "as três iluminações."
            )
            image_button.setEnabled(image_available)
        except Exception:
            pass

    if state_label is not None:
        try:
            validation = record.get("validation", {}) if isinstance(record, dict) else {}
            reason = str(validation.get("reason", "") or "")
            valid = validation.get("valid", None) if isinstance(validation, dict) else None
            source_ip = str(record.get("source_ip", "") or "") if isinstance(record, dict) else ""
            source = str(record.get("source", "") or "").strip().lower() if isinstance(record, dict) else ""

            multilight_match = _multilight_event_matches(panel)
            analysis_count, frame_count = _multilight_counts(panel)

            if multilight_match:
                if analysis_count == 3 and frame_count == 3:
                    state_label.setText(
                        "Multilight completo • 3 análises + "
                        "imagem composta SIDE/TOP/MID"
                    )
                    state_label.setProperty("state", "ready")
                else:
                    state_label.setText(
                        "Multilight em coleta • "
                        f"análises {analysis_count}/3 • imagens {frame_count}/3"
                    )
                    state_label.setProperty("state", "partial")
            elif (
                debug_available
                and image_available
                and source == "local_mss"
            ):
                state_label.setText(
                    "Captura local MSS analisada • relatório + imagem"
                )
                state_label.setProperty("state", "ready")
            elif debug_available and image_available and valid is False:
                suffix = f" • {reason}" if reason else ""
                ip_text = f" • {source_ip}" if source_ip else ""
                state_label.setText(
                    f"Último frame XP REJEITADO{suffix}{ip_text} • imagem preservada"
                )
                state_label.setProperty("state", "rejected")
            elif debug_available and image_available and valid is True:
                ip_text = f" • {source_ip}" if source_ip else ""
                state_label.setText(
                    f"Último frame XP validado{ip_text} • relatório + imagem"
                )
                state_label.setProperty("state", "ready")
            elif debug_available and image_available:
                ip_text = f" • {source_ip}" if source_ip else ""
                state_label.setText(
                    f"Frame XP recebido{ip_text} • aguardando validação"
                )
                state_label.setProperty("state", "partial")
            elif debug_available:
                state_label.setText(
                    "Relatório disponível • imagem do evento não está preservada"
                )
                state_label.setProperty("state", "partial")
            else:
                state_label.setText("Aguardando a primeira captura analisada")
                state_label.setProperty("state", "idle")

            style = state_label.style()
            style.unpolish(state_label)
            style.polish(state_label)
            state_label.update()
        except Exception:
            pass


def set_network_debug_available(panel, available: bool = True) -> None:
    """Compatibilidade com o filtro anterior; a UI deriva o estado real."""
    if not available:
        button = getattr(panel, "btn_copy_network_debug", None)
        if button is not None:
            try:
                button.setEnabled(False)
            except Exception:
                pass
    sync_network_debug_controls(panel)


__all__ = [
    "DEBUG_SCHEMA",
    "MULTILIGHT_DEBUG_ORDER",
    "build_multilight_composite",
    "copy_network_debug_to_clipboard",
    "copy_network_image_to_clipboard",
    "format_multilight_debug_report",
    "format_network_debug_report",
    "multilight_copy_image_snapshot",
    "network_debug_image_available",
    "network_debug_image_snapshot",
    "set_network_debug_available",
    "sync_network_debug_controls",
]
