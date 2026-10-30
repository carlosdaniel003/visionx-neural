"""Roteamento SIDE: memória humana qualificada precede a CNN por similaridade.

95% isolados NÃO equivalem a uma identidade física: exige >=97% em A e B
para o mesmo PNG, confirmação persistida >=97,5% com margem >=4pp e
ausência de contraevidência NG significativa.

Aplica-se às cinco categorias reconhecidas pela AOI. A sonda independente
é um VETO TRANSVERSAL DE AUSÊNCIA, não um classificador de inversão,
deslocamento, tombstone ou adesivo. Outras manifestações NG semelhantes
a OK continuam sendo risco, exigindo replay físico antes da produção.

A rota altera o julgamento do evento NOVO, não os pesos da CNN nem a
classificação histórica. O comando XP continua sob o gate de Produção.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import PurePosixPath

from src.core.strict_category_memory import canonical_memory_category
from src.core.physical_missing_veto import physical_absence_probe
from src.utils.text_normalizer import CATEGORIES

RULE = "side_operator_memory_first_97pct_v2"
MEMORY_SCHEMA = "visionx.side_memory_first.v2"
MIN_KNN = 0.97
MIN_HUMAN = 0.975
MIN_MARGIN = 0.04
KNOWN_MEMORY_CATEGORIES = frozenset(
    canonical_memory_category(value) for value in CATEGORIES
)
PHYSICAL_PROBE_SCOPE = "CROSS_CATEGORY_ABSENCE_VETO_ONLY"


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if isfinite(number) and 0.0 <= number <= 1.0 else None


def _candidate(analysis: dict | None, category: str) -> dict | None:
    """Qualifica a memória em SIDE; rótulo AOI é escopo de busca, não prova."""
    if not isinstance(analysis, dict) or analysis.get("lighting_mode") != "SIDE":
        return None
    d = analysis.get("detail")
    if not isinstance(d, dict):
        return None
    scope = canonical_memory_category(category)
    if scope not in KNOWN_MEMORY_CATEGORIES:
        # Categoria desconhecida/erro de OCR nunca pode receber atalho.
        return None
    if (
        d.get("recognition_route") not in {"NEW_CNN", "NEW_EXPERTS"}
        or d.get("recognition_memory_verified") is True
        or d.get("recognition_match") == "EXACT_PAIR"
        or d.get("memory_conflict") is True
        or d.get("hard_missing_evidence") is True
        or analysis.get("hard_missing_evidence") is True
        or d.get("multilight_physical_disagreement") is True
        or d.get("capture_error") is True
    ):
        return None
    prior = d.get("physical_missing_veto")
    if isinstance(prior, dict) and prior.get("status") in {
        "SUSPECT_ABSENCE", "UNAVAILABLE", "NG_CORROBORATED",
    }:
        return None

    human = d.get("confirmed_visual_recall")
    knn = d.get("knn_c")
    if not (isinstance(human, dict) and isinstance(knn, dict)):
        return None
    if not (
        human.get("schema") == "visionx.human_visual_recall.v1"
        and human.get("source") == "PERSISTENT_HUMAN_LABELS"
        and human.get("diagnostic_only") is True
        and human.get("auto_ok_authorized") is False
        and human.get("category") == scope
        and knn.get("schema") == "visionx.knn_c_live_review_gated.v1"
        and knn.get("bank_status") == "READY"
        and knn.get("diagnostic_only") is True
        and knn.get("auto_decision_authorized") is False
        and knn.get("uses_aoi_category") is False
    ):
        return None
    label = (
        "OK" if human.get("status") == "HUMAN_OK_VISUAL_CANDIDATE"
        else "NG" if human.get("status") == "HUMAN_NG_VISUAL_CANDIDATE"
        else None
    )
    if label is None or canonical_memory_category(d.get("cnn_v2_aoi_category", scope)) != scope:
        return None
    other = "NG" if label == "OK" else "OK"
    human_score = _number(human.get("best_" + label.lower() + "_similarity"))
    opposed = _number(human.get("best_" + other.lower() + "_similarity"))
    human_source = human.get("best_" + label.lower() + "_source")
    if (
        human_score is None or human_score < MIN_HUMAN
        or opposed is None or human_score - opposed < MIN_MARGIN
        or not isinstance(human_source, dict)
        or human_source.get("label") != label
        or human_source.get("source") != "OPERATOR_CONFIRMED"
        or not isinstance(human_source.get("record_id"), str)
        or len(human_source["record_id"]) != 64
    ):
        return None
    # Estratégia primária: os dois descritores originais concordam num PNG.
    witnesses = []
    rows_by_method = []
    for key in ("neighbors_A", "neighbors_B"):
        rows = knn.get(key)
        if not isinstance(rows, list) or not rows:
            return None
        if any(
            not isinstance(row, dict)
            or row.get("label") not in {"OK", "NG"}
            or _number(row.get("similarity")) is None
            for row in rows[:5]
        ):
            return None
        rows_by_method.append(rows)
        top = rows[0]
        score = _number(top.get("similarity"))
        path = top.get("path")
        digest = top.get("source_pixel_sha256")
        witnesses.append((top.get("label"), score, path, digest))
    prefix = "public/" + ("ok_archive" if label == "OK" else "ng_archive") + "/"
    a, b = witnesses
    original_agrees = bool(
        a[0] == b[0] == label
        and a[1] >= MIN_KNN and b[1] >= MIN_KNN
        and a[2:] == b[2:]
        and isinstance(a[2], str) and a[2].startswith(prefix)
        and ".." not in PurePosixPath(a[2]).parts
        and isinstance(a[3], str) and len(a[3]) == 64
        and knn.get("vote_A") == knn.get("vote_B") == label
        and all(
            a_score - _number(row["similarity"]) >= MIN_MARGIN
            for a_score, rows in ((a[1], rows_by_method[0]), (b[1], rows_by_method[1]))
            for row in rows if row["label"] == other
        )
    )

    # Uma SEGUNDA rota compara JSON humano e JSON archive na MESMA métrica:
    # os scores brutos de foco/geometria A/B permanecem inalterados.
    # Esta rota só existe com fonte PNG original + fonte humana verificadas
    # e é vetada se A/B detectarem adversário forte (>=90%).
    aligned = knn.get("aligned_archive")
    source = aligned.get("archive_source") if isinstance(aligned, dict) else None
    aligned_ok = bool(
        not original_agrees
        and isinstance(aligned, dict)
        and aligned.get("schema") == "visionx.knn_c_archive_alignment.v1"
        and aligned.get("source") == "VERIFIED_ARCHIVE_JSON"
        and aligned.get("diagnostic_only") is True
        and aligned.get("auto_decision_authorized") is False
        and aligned.get("status") == "RECONCILED_" + label
        and aligned.get("label") == label
        and aligned.get("category") == scope
        and aligned.get("human_record_id") == human_source["record_id"]
        and _number(aligned.get("archive_similarity")) is not None
        and _number(aligned.get("archive_similarity")) >= MIN_KNN
        and _number(aligned.get("human_archive_similarity")) is not None
        and _number(aligned.get("human_archive_similarity")) >= MIN_KNN
        and _number(aligned.get("best_opposite_archive_similarity")) is not None
        and _number(aligned.get("archive_similarity"))
            - _number(aligned.get("best_opposite_archive_similarity")) >= MIN_MARGIN
        and isinstance(source, dict)
        and source.get("label") == label
        and isinstance(source.get("path"), str)
        and source["path"].startswith(prefix)
        and ".." not in PurePosixPath(source["path"]).parts
        and isinstance(source.get("source_sha256"), str)
        and len(source["source_sha256"]) == 64
        and all(
            _number(row["similarity"]) < 0.90
            for rows in rows_by_method for row in rows[:5]
            if row["label"] == other
        )
    )
    if not original_agrees and not aligned_ok:
        return None

    mode = "KNN_AB_SAME_PNG" if original_agrees else "ALIGNED_HUMAN_ARCHIVE_JSON"
    evidence = {
        "schema": MEMORY_SCHEMA, "rule": RULE, "label": label,
        "human_source": deepcopy(human_source),
        "human_similarity": human_score,
        "opposite_human_similarity": opposed,
        "knn_a_similarity": a[1],
        "knn_b_similarity": b[1],
        "knn_agreement_mode": mode,
        "archive_source": a[2] if original_agrees else source["path"],
        "archive_pixel_sha256": a[3] if original_agrees else None,
        "archive_file_sha256": None if original_agrees else source["source_sha256"],
        "aligned_archive": deepcopy(aligned) if aligned_ok else None,
        "min_similarity": MIN_KNN,
        "min_human_similarity": MIN_HUMAN,
        "min_opposite_margin": MIN_MARGIN,
        "approximate": True,
        "exact_pair_claimed": False,
    }
    return evidence


def _verified_probe(probe: dict, label: str) -> bool:
    if not isinstance(probe, dict) or probe.get("reason") != "TWO_SCALE_ROI_CHECK":
        return False
    scales = probe.get("scales")
    geometry = probe.get("geometry")
    if not (
        isinstance(scales, list) and len(scales) == 2
        and isinstance(geometry, dict)
        and geometry.get("method") in {"exact", "center_trim_no_rescale"}
        and all(
            isinstance(row, dict)
            and type(row.get("suspected_absence")) is bool
            for row in scales
        )
    ):
        return False
    if label == "NG":
        # Se a ROI sugere ausência, isso não veta rejeição NG.
        # Verificar integridade da medição antes de permitir 1/NG.
        if probe.get("status") == "SUSPECT_ABSENCE":
            return any(row.get("suspected_absence") is True for row in scales)
        return probe.get("status") in {"NO_STRONG_EVIDENCE", "GEOMETRY_REVIEW"}
    if any(row["suspected_absence"] for row in scales):
        return False
    if probe.get("status") == "NO_STRONG_EVIDENCE":
        return geometry.get("method") == "exact" and geometry.get("requires_review") is False
    # Aparagem central mínima produzida pela própria AOI: duas ROIs
    # conferidas sem ausência e desvio máximo de 3 px / 1% por eixo.
    # Nada de resize para mascarar uma região ausente.
    if probe.get("status") != "GEOMETRY_REVIEW":
        return False
    if geometry.get("method") != "center_trim_no_rescale" or geometry.get("requires_review") is not True:
        return False
    shape_a, shape_b = geometry.get("reference_hw"), geometry.get("test_hw")
    if not (
        isinstance(shape_a, list) and isinstance(shape_b, list)
        and len(shape_a) == len(shape_b) == 2
        and all(type(v) is int and v >= 200 for v in shape_a + shape_b)
    ):
        return False
    return all(
        abs(a - b) <= 3 and abs(a - b) / min(a, b) <= 0.01
        for a, b in zip(shape_a, shape_b)
    )


def build_side_memory_first(
    analysis: dict | None, category: str, reference, test, *, probe=None,
) -> dict | None:
    """Prioriza memória em uma luz e consulta veto físico de ausência.

    Para outras categorias, a sonda NÃO descarta erros físicos de posição,
    orientação e quantidade de adesivo. O gate do XP permanece responsável
    por autorizar a decisão e o histórico deve ser validado com NGs difíceis.
    """
    evidence = _candidate(analysis, category)
    if evidence is None:
        return None
    checker = physical_absence_probe if probe is None else probe
    try:
        physical = checker(reference, test)
    except Exception:
        return None
    if not _verified_probe(physical, evidence["label"]):
        return None
    final = deepcopy(analysis)
    label = evidence["label"]
    final["verdict"] = "FALHA FALSA" if label == "OK" else "DEFEITO REAL"
    final["is_defect"] = label == "NG"
    final["confidence"] = min(
        evidence["human_similarity"],
        evidence["knn_a_similarity"], evidence["knn_b_similarity"],
    )
    final["lighting_mode"] = "SIDE"
    final["multilight_final"] = True
    final["multilight_category"] = canonical_memory_category(category)
    final["eligible_for_final_decision"] = True
    final["production_review_required"] = False
    final["reason"] = (
        "SIDE • memória humana previamente julgada " + label
        + "; KNN A/B concordam >=97% com a mesma fonte, "
        "ou JSON de arquivo e julgamento humano na mesma métrica, "
        "margem >4pp; sonda de ausência física aplicada como veto, "
        "não como validação de toda categoria; TOP/MID dispensadas. "
        "Similaridade não é probabilidade."
    )
    d = final["detail"]
    d["side_memory_first"] = {
        **evidence,
        "physical_probe": deepcopy(physical),
        "physical_probe_scope": PHYSICAL_PROBE_SCOPE,
        "original_verdict": analysis.get("verdict"),
        "original_is_defect": analysis.get("is_defect"),
    }
    d["side_early_exit"] = {
        "rule": RULE,
        "threshold": MIN_KNN,
        "verdict": final["verdict"],
        "analyzed_lights": ["SIDE"],
        "skipped_lights": ["TOP", "MID"],
        "production_auto_allowed": True,
        "operator_review_required": False,
    }
    d["fusion_rule"] = RULE
    d["multilight_final"] = True
    d["multilight_category"] = final["multilight_category"]
    d["eligible_for_final_decision"] = True
    d["operator_review_required"] = False
    trace = d.get("decision_trace")
    if isinstance(trace, dict):
        trace["operator_review_required"] = False
        trace["fusion_rule"] = RULE
    d["side_memory_first"]["geometry_small_trim_authorized"] = bool(
        physical.get("status") == "GEOMETRY_REVIEW"
    )
    return final


def memory_first_side_decision(analysis: dict | None) -> str | None:
    """Gate da Produção: reconstrói evidência em vez de confiar no marker."""
    if not isinstance(analysis, dict):
        return None
    d = analysis.get("detail")
    if not isinstance(d, dict):
        return None
    marker = d.get("side_memory_first")
    exit_data = d.get("side_early_exit")
    if not (isinstance(marker, dict) and isinstance(exit_data, dict)):
        return None
    scope = analysis.get("multilight_category")
    original = {
        **analysis,
        "verdict": marker.get("original_verdict"),
        "is_defect": marker.get("original_is_defect"),
        "detail": d,
    }
    expected = _candidate(original, scope)
    label = expected.get("label") if isinstance(expected, dict) else None
    if (
        expected is None or label not in {"OK", "NG"}
        or any(marker.get(key) != value for key, value in expected.items())
        or marker.get("geometry_small_trim_authorized") is not (
            marker.get("physical_probe", {}).get("status") == "GEOMETRY_REVIEW"
        )
        or marker.get("physical_probe_scope") != PHYSICAL_PROBE_SCOPE
        or not _verified_probe(marker.get("physical_probe"), label)
        or analysis.get("lighting_mode") != "SIDE"
        or analysis.get("multilight_final") is not True
        or analysis.get("eligible_for_final_decision") is not True
        or analysis.get("production_review_required") is not False
        or d.get("multilight_final") is not True
        or d.get("eligible_for_final_decision") is not True
        or d.get("operator_review_required") is not False
        or d.get("fusion_rule") != RULE
        or d.get("multilight_category") != scope
        or analysis.get("verdict") != ("FALHA FALSA" if label == "OK" else "DEFEITO REAL")
        or analysis.get("is_defect") is not (label == "NG")
        or d.get("hard_missing_evidence") is True
        or analysis.get("hard_missing_evidence") is True
        or d.get("memory_conflict") is True
        or exit_data != {
            "rule": RULE,
            "threshold": MIN_KNN,
            "verdict": analysis["verdict"],
            "analyzed_lights": ["SIDE"],
            "skipped_lights": ["TOP", "MID"],
            "production_auto_allowed": True,
            "operator_review_required": False,
        }
    ):
        return None
    trace = d.get("decision_trace")
    if not isinstance(trace, dict) or trace.get("operator_review_required") is not False:
        return None
    return label


__all__ = [
    "RULE", "KNOWN_MEMORY_CATEGORIES", "PHYSICAL_PROBE_SCOPE",
    "build_side_memory_first", "memory_first_side_decision",
]
