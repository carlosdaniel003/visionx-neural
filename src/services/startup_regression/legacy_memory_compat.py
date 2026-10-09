"""Inspeção READ-ONLY de memórias antigas para simular compatibilidade exata.

Não altera schema, não cria ou assina registros, não carrega KNN, não
retreina CNN e não habilita a rota operacional. Em particular,
LEGACY_EXACT_CANDIDATE é *hipótese de compatibilidade*, nunca KNOWN.

É propositalmente conservadora: rejeita registro sem decisão humana,
sem assinatura de anomalia, sem par de PNGs ou com hash divergente.
Os dados e rótulos não são obtidos do acervo OK/NG nessa etapa.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from src.core.anomaly_signature import valid_anomaly_signature
from src.core.strict_category_memory import (
    canonical_memory_category, canonical_memory_lighting,
)
from src.core.verified_memory_router import (
    VerifiedKNNMemory, _clean, _file_within, _fingerprint_png,
)

LEGACY_COMPAT_SCHEMA = "visionx.legacy_memory_compat.v1"
HUMAN_SOURCES = frozenset({
    "button", "xp_keyboard", "keyboard", "manual",
    "operator", "physical_keyboard", "human",
})
COMPATIBLE = "LEGADO_PAR_AUDITAVEL_SIMULADO"
UNKNOWN_SCHEMA = "SEM_SCHEMA"
MAX_EXAMPLES_PER_SCHEMA = 3


def _safe_text(value, limit: int = 120) -> str:
    return str(value or "")[:limit]


def _schema_id(value) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()[:100]
    return UNKNOWN_SCHEMA


def _valid_signature(data: dict) -> bool:
    analysis = data.get("analysis")
    analysis = analysis if isinstance(analysis, dict) else {}
    return any(valid_anomaly_signature(value) for value in (
        analysis.get("anomaly_memory"),
        analysis.get("anomaly_signature"),
        data.get("anomaly_memory"),
    ))


def inspect_legacy_record(json_path: Path, folder_label: str, data: dict) -> dict:
    """Audita um JSON pré-v3 e devolve a chave exata *simulada* ou motivo.

    A chave é derivada apenas de metadados próprios e das imagens
    efetivamente presentes no diretório do JSON. Não há normalização
    artificial de OCR, parsing de nome de arquivo ou adivinhação de rótulo.
    """
    if not isinstance(data, dict):
        raise TypeError("JSON de memória deve ser um objeto")
    if data.get("schema") == "visionx.memory.v3":
        raise ValueError("Este adaptador só examina formatos não-v3")
    result = {
        "schema": _schema_id(data.get("schema")),
        "status": "NAO_AVALIADO",
        "would_be_key": None,
        "human_confirmed": False,
        "signature_present": _valid_signature(data),
        "reference_png_present": False,
        "test_png_present": False,
        "reference_fingerprint": None,
        "test_fingerprint": None,
        "declared_test_fingerprint_matches": None,
        "has_dedup_pointer": False,
        "confidence_note": "SIMULACAO_SEM_ESCRITA_NAO_E_KNN_KNOWN",
    }
    decision = data.get("decision")
    decision = decision if isinstance(decision, dict) else {}
    info = data.get("aoi_info")
    info = info if isinstance(info, dict) else {}
    storage = data.get("storage")
    storage = storage if isinstance(storage, dict) else {}

    label = _safe_text(data.get("label")).upper()
    operator_label = _safe_text(decision.get("operator_label")).upper()
    source = _safe_text(decision.get("source")).lower()
    result["human_confirmed"] = (
        label in {"OK", "NG"}
        and operator_label == label == str(folder_label).upper()
        and (source in HUMAN_SOURCES or source.startswith("operator_"))
    )
    result["has_dedup_pointer"] = bool(
        str(storage.get("duplicate_visual_of_json", "") or "").strip()
    )
    ref_path = _file_within(
        json_path.parent, storage.get("reference_image_file", "")
    )
    test_path = _file_within(
        json_path.parent, storage.get("test_image_file", "")
    )
    result["reference_png_present"] = ref_path is not None
    result["test_png_present"] = test_path is not None

    # Realizar diagnóstico mesmo quando a procedência humana falhar,
    # mas nunca admitir a simulação nesse caso.
    if not result["human_confirmed"]:
        result["status"] = "LEGADO_SEM_CONFIRMacao_HUMANA".upper()
        return result
    if not result["signature_present"]:
        result["status"] = "LEGADO_SEM_ASSINATURA_ANOMALIA"
        return result
    if not ref_path or not test_path:
        result["status"] = (
            "LEGADO_DEDUP_SEM_PAR" if result["has_dedup_pointer"]
            else "LEGADO_SEM_PAR_PNG"
        )
        return result
    if not all((
        canonical_memory_category(info.get("category")),
        _clean(info.get("board")),
        _clean(info.get("parts")),
        _clean(info.get("value")),
    )):
        result["status"] = "LEGADO_OCR_METADADOS_INCOMPLETOS"
        return result

    try:
        reference_hash = _fingerprint_png(ref_path)
        test_hash = _fingerprint_png(test_path)
    except (OSError, ValueError, TypeError) as exc:
        result["status"] = "LEGADO_PNG_ILEGIVEL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    result["reference_fingerprint"] = reference_hash or None
    result["test_fingerprint"] = test_hash or None
    if not reference_hash or not test_hash:
        result["status"] = "LEGADO_PNG_ILEGIVEL"
        return result

    declared = str(storage.get("test_image_fingerprint", "") or "").strip()
    if declared:
        result["declared_test_fingerprint_matches"] = declared == test_hash
        if declared != test_hash:
            result["status"] = "LEGADO_HASH_TESTE_DIVERGENTE"
            return result
    info_for_key = {
        "category": info.get("category"),
        "board": info.get("board"),
        "parts": info.get("parts"),
        "value": info.get("value"),
        "lighting_mode": canonical_memory_lighting(info.get("lighting_mode")),
    }
    key = VerifiedKNNMemory._key(info_for_key, reference_hash, test_hash)
    if key is None:
        result["status"] = "LEGADO_CHAVE_INVALIDA"
        return result
    result["would_be_key"] = key
    result["status"] = COMPATIBLE
    return result


def summarize_legacy_profiles(rows: list[dict]) -> dict:
    """Distribui schemas e causas sem filtrar qualquer arquivo antigo."""
    schemes = Counter()
    issues = Counter()
    compatible = Counter()
    samples = {}
    for row in rows:
        profile = row.get("legacy_profile")
        if not isinstance(profile, dict):
            continue
        schema = profile.get("schema") or UNKNOWN_SCHEMA
        status = profile.get("status") or "DESCONHECIDO"
        schemes[schema] += 1
        issues[status] += 1
        if status == COMPATIBLE:
            compatible[schema] += 1
        if len(samples.setdefault(schema, [])) < MAX_EXAMPLES_PER_SCHEMA:
            samples[schema].append(row.get("path"))
    return {
        "schema": LEGACY_COMPAT_SCHEMA,
        "records_examined": sum(schemes.values()),
        "schema_distribution": dict(sorted(schemes.items())),
        "eligibility_reasons": dict(sorted(issues.items())),
        "compatible_in_simulation": sum(compatible.values()),
        "compatible_by_schema": dict(sorted(compatible.items())),
        "sample_record_paths": dict(sorted(samples.items())),
        "production_knn_coverage_changed": False,
        "note": (
            "Compatibilidade simulada não altera a KNN de produção. "
            "Somente pares com fonte humana, assinatura, PNGs e hashes "
            "verificáveis podem ser candidatos; NÃO são aprovações reais."
        ),
    }


__all__ = [
    "COMPATIBLE", "UNKNOWN_SCHEMA", "LEGACY_COMPAT_SCHEMA",
    "inspect_legacy_record", "summarize_legacy_profiles",
]
