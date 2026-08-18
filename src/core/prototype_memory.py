"""Memória condensada por protótipos com proteção absoluta dos NG.

A compactação não altera a assinatura visual nem a política de melhor
correspondência. Registros OK praticamente equivalentes são representados por
um único protótipo; o número de ocorrências é apenas estatístico. Registros NG
nunca são fundidos automaticamente.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.config.settings import settings
from src.core.anomaly_signature import valid_anomaly_signature
from src.core.dual_scale_memory import valid_context_signature
from src.core.strict_category_memory import canonical_memory_category


PROTOTYPE_SCHEMA = "visionx.prototype.v1"
OK_PROTOTYPE_MERGE_SIMILARITY = 0.985
OK_PROTOTYPE_EPICENTER_MIN = 0.985
OK_PROTOTYPE_CONTEXT_MIN = 0.970
OK_LEGACY_MEMORY_MERGE_SIMILARITY = 0.995


def _clean_key(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _record_scope(record: dict) -> tuple[str, str, str, str]:
    signature = record.get("anomaly_signature") or {}
    scale = (
        "dual"
        if valid_context_signature(signature.get("context_signature", {}))
        else "legacy"
    )
    return (
        canonical_memory_category(record.get("category", "")),
        _clean_key(record.get("board", "")),
        _clean_key(record.get("part", "")),
        scale,
    )


def _json_scope(data: dict) -> tuple[str, str, str]:
    info = data.get("aoi_info", {}) if isinstance(data, dict) else {}
    return (
        canonical_memory_category(info.get("category", "")),
        _clean_key(info.get("board", "")),
        _clean_key(info.get("parts", "")),
    )


def _extract_signature(data: dict) -> dict | None:
    analysis = data.get("analysis", {}) if isinstance(data, dict) else {}
    candidates = (
        analysis.get("anomaly_memory") if isinstance(analysis, dict) else None,
        analysis.get("anomaly_signature") if isinstance(analysis, dict) else None,
        data.get("anomaly_memory") if isinstance(data, dict) else None,
    )
    for candidate in candidates:
        if valid_anomaly_signature(candidate):
            return candidate
    return None


def _resolved_label(data: dict, fallback: str = "") -> str:
    explicit = str(data.get("label", "")).strip().upper()
    if explicit in {"OK", "NG"}:
        return explicit
    analysis = data.get("analysis", {}) if isinstance(data, dict) else {}
    operator = str((analysis or {}).get("operator_label", "")).strip().upper()
    if operator in {"OK", "NG"}:
        return operator
    fallback = str(fallback or "").strip().upper()
    return fallback if fallback in {"OK", "NG"} else ""


def _read_json(path: str | Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _atomic_write_json(path: str | Path, data: dict) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    temporary.replace(target)


def _record_occurrences(record: dict) -> int:
    try:
        return max(1, int(record.get("prototype_occurrences", 1)))
    except (TypeError, ValueError):
        return 1


def _augment_record_from_json(record: dict) -> dict:
    output = dict(record)
    data = _read_json(output.get("json_path", ""))
    if not data:
        output.setdefault("board", "")
        output.setdefault("prototype_occurrences", 1)
        return output

    info = data.get("aoi_info", {}) or {}
    prototype = data.get("prototype", {}) or {}
    output["board"] = _clean_key(info.get("board", ""))
    output["prototype_occurrences"] = max(
        1,
        int(prototype.get("occurrences", 1) or 1),
    )
    output["prototype_persisted"] = bool(prototype)
    output["prototype_protected"] = bool(prototype.get("protected", False))
    return output


def _compare(
    comparator: Callable[[dict, dict], tuple[float, dict]],
    first: dict,
    second: dict,
) -> tuple[float, dict]:
    try:
        similarity, breakdown = comparator(first, second)
        return float(np.clip(float(similarity), 0.0, 1.0)), (
            breakdown if isinstance(breakdown, dict) else {}
        )
    except Exception:
        return 0.0, {}


def _ok_merge_eligible(
    first: dict,
    second: dict,
    comparator: Callable[[dict, dict], tuple[float, dict]],
    *,
    allow_legacy: bool,
) -> tuple[bool, float, dict]:
    if not valid_anomaly_signature(first) or not valid_anomaly_signature(second):
        return False, 0.0, {}

    similarity, breakdown = _compare(comparator, first, second)
    first_dual = valid_context_signature(first.get("context_signature", {}))
    second_dual = valid_context_signature(second.get("context_signature", {}))

    if first_dual and second_dual:
        epicenter = float(breakdown.get("epicenter_similarity", similarity) or 0.0)
        context = float(breakdown.get("context_similarity", 0.0) or 0.0)
        eligible = bool(
            similarity >= OK_PROTOTYPE_MERGE_SIMILARITY
            and epicenter >= OK_PROTOTYPE_EPICENTER_MIN
            and context >= OK_PROTOTYPE_CONTEXT_MIN
        )
        return eligible, similarity, breakdown

    if allow_legacy and not first_dual and not second_dual:
        return (
            similarity >= OK_LEGACY_MEMORY_MERGE_SIMILARITY,
            similarity,
            breakdown,
        )

    # Nunca funde persistentemente uma memória antiga sem contexto com uma nova
    # memória dual-scale: o conhecimento contextual recém-adquirido é preservado.
    return False, similarity, breakdown


def _choose_medoid(
    members: list[dict],
    comparator: Callable[[dict, dict], tuple[float, dict]],
) -> tuple[dict, float]:
    if len(members) <= 1:
        return members[0], 1.0

    best_record = members[0]
    best_average = -1.0
    for candidate in members:
        candidate_signature = candidate.get("anomaly_signature") or {}
        scores = []
        for other in members:
            if other is candidate:
                continue
            score, _ = _compare(
                comparator,
                candidate_signature,
                other.get("anomaly_signature") or {},
            )
            scores.append(score)
        average = float(np.mean(scores)) if scores else 1.0
        if average > best_average:
            best_average = average
            best_record = candidate
    return best_record, max(0.0, best_average)


def condense_ok_records(
    records: list[dict],
    comparator: Callable[[dict, dict], tuple[float, dict]],
) -> list[dict]:
    """Cria protótipos OK em memória sem apagar os JSONs existentes."""
    augmented = [_augment_record_from_json(record) for record in records]
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for record in augmented:
        grouped.setdefault(_record_scope(record), []).append(record)

    prototypes: list[dict] = []
    prototype_index = 0
    for scope in sorted(grouped):
        clusters: list[list[dict]] = []
        for record in sorted(
            grouped[scope],
            key=lambda item: str(item.get("json_path", item.get("path", ""))),
        ):
            signature = record.get("anomaly_signature") or {}
            best_cluster = None
            best_similarity = -1.0
            for cluster in clusters:
                representative = cluster[0].get("anomaly_signature") or {}
                eligible, similarity, _ = _ok_merge_eligible(
                    signature,
                    representative,
                    comparator,
                    allow_legacy=True,
                )
                if eligible and similarity > best_similarity:
                    best_cluster = cluster
                    best_similarity = similarity
            if best_cluster is None:
                clusters.append([record])
            else:
                best_cluster.append(record)

        for cluster in clusters:
            representative, centrality = _choose_medoid(cluster, comparator)
            prototype = dict(representative)
            prototype_index += 1
            prototype["prototype_schema"] = PROTOTYPE_SCHEMA
            prototype["prototype_id"] = f"OK-{prototype_index:05d}"
            prototype["prototype_label"] = "OK"
            prototype["prototype_protected"] = False
            prototype["prototype_occurrences"] = int(
                sum(_record_occurrences(item) for item in cluster)
            )
            prototype["prototype_member_jsons"] = int(len(cluster))
            prototype["prototype_centrality"] = float(centrality)
            prototype["quantity_influence"] = False
            prototypes.append(prototype)
    return prototypes


def protect_ng_records(records: list[dict]) -> list[dict]:
    """Mantém cada NG individualmente, mesmo quando são visualmente idênticos."""
    protected = []
    for index, record in enumerate(records, start=1):
        item = _augment_record_from_json(record)
        item["prototype_schema"] = PROTOTYPE_SCHEMA
        item["prototype_id"] = f"NG-{index:05d}"
        item["prototype_label"] = "NG"
        item["prototype_protected"] = True
        item["prototype_member_jsons"] = 1
        item["quantity_influence"] = False
        protected.append(item)
    return protected


def _focus_box(analysis: dict | None) -> Any:
    payload = analysis if isinstance(analysis, dict) else {}
    detail = payload.get("detail", {}) if isinstance(payload.get("detail", {}), dict) else {}
    return (
        detail.get("semantic_focus_box")
        or detail.get("adhesive_roi_box")
        or detail.get("missing_roi_box")
        or detail.get("inverted_roi_box")
        or detail.get("roi_box")
        or payload.get("bounding_box")
    )


def _query_signature(
    dataset_manager_module,
    ng_image,
    sample_image,
    aoi_info,
    analysis,
) -> dict | None:
    detail = (analysis or {}).get("detail", {}) if isinstance(analysis, dict) else {}
    signature = (
        detail.get("anomaly_signature")
        or detail.get("query_anomaly_signature")
        or {}
    )
    if valid_anomaly_signature(signature):
        return signature
    try:
        signature = dataset_manager_module.build_anomaly_signature(
            sample_image,
            ng_image,
            detail,
            aoi_info,
            _focus_box(analysis),
        )
    except Exception:
        return None
    return signature if valid_anomaly_signature(signature) else None


def _find_persistent_ok_prototype(
    signature: dict,
    aoi_info: dict | None,
    comparator: Callable[[dict, dict], tuple[float, dict]],
):
    info = aoi_info if isinstance(aoi_info, dict) else {}
    target_scope = (
        canonical_memory_category(info.get("category", "")),
        _clean_key(info.get("board", "")),
        _clean_key(info.get("parts", "")),
    )
    if not settings.NORMAL_DIR.exists():
        return None

    best = None
    best_similarity = -1.0
    for path in settings.NORMAL_DIR.rglob("*.json"):
        data = _read_json(path)
        if not data or _resolved_label(data, "OK") != "OK":
            continue
        if _json_scope(data) != target_scope:
            continue
        stored = _extract_signature(data)
        if not valid_anomaly_signature(stored):
            continue
        eligible, similarity, breakdown = _ok_merge_eligible(
            signature,
            stored,
            comparator,
            allow_legacy=False,
        )
        if eligible and similarity > best_similarity:
            best = (path, data, similarity, breakdown)
            best_similarity = similarity
    return best


def _save_duplicate_audit_images(
    json_path: Path,
    ng_image,
    sample_image,
    timestamp: str,
) -> tuple[str, str]:
    audit_dir = json_path.parent / "prototype_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    stem = f"observation_{timestamp.replace(':', '').replace('-', '').replace('.', '_')}"
    test_path = audit_dir / f"{stem}_test.png"
    ref_path = audit_dir / f"{stem}_reference.png"
    test_file = ""
    ref_file = ""
    if isinstance(ng_image, np.ndarray) and ng_image.size > 0:
        if cv2.imwrite(str(test_path), ng_image):
            test_file = str(test_path.relative_to(json_path.parent))
    if isinstance(sample_image, np.ndarray) and sample_image.size > 0:
        if cv2.imwrite(str(ref_path), sample_image):
            ref_file = str(ref_path.relative_to(json_path.parent))
    return test_file, ref_file


def _update_prototype_metadata(
    path: str | Path,
    *,
    label: str,
    protected: bool,
    increment: bool,
    similarity: float | None = None,
    source: str = "",
    ai_decision: str = "",
    save_images: bool = False,
    ng_image=None,
    sample_image=None,
) -> None:
    data = _read_json(path)
    if not data:
        return
    now = datetime.now().isoformat()
    current = data.get("prototype", {}) if isinstance(data.get("prototype"), dict) else {}
    previous_occurrences = max(1, int(current.get("occurrences", 1) or 1))
    occurrences = previous_occurrences + 1 if increment else previous_occurrences
    disagreement = bool(ai_decision and str(ai_decision).upper() != label)

    metadata = {
        "schema": PROTOTYPE_SCHEMA,
        "label": label,
        "protected": bool(protected),
        "quantity_influence": False,
        "occurrences": int(occurrences),
        "first_observed_at": current.get("first_observed_at") or data.get("timestamp", now),
        "last_observed_at": now,
        "compaction_policy": "never_merge_ng" if protected else "merge_redundant_ok",
        "merge_similarity_threshold": (
            None if protected else OK_PROTOTYPE_MERGE_SIMILARITY
        ),
        "last_similarity": (
            float(similarity) if similarity is not None else current.get("last_similarity")
        ),
        "last_source": str(source or ""),
        "last_ai_label": str(ai_decision or ""),
        "disagreement_occurrences": int(current.get("disagreement_occurrences", 0) or 0)
        + (1 if increment and disagreement else 0),
        "audit_image_occurrences": int(current.get("audit_image_occurrences", 0) or 0),
    }

    if increment and save_images:
        test_file, ref_file = _save_duplicate_audit_images(
            Path(path), ng_image, sample_image, now
        )
        if test_file or ref_file:
            metadata["audit_image_occurrences"] += 1
            metadata["last_audit_test_file"] = test_file
            metadata["last_audit_reference_file"] = ref_file
    else:
        for key in ("last_audit_test_file", "last_audit_reference_file"):
            if key in current:
                metadata[key] = current[key]

    data["prototype"] = metadata
    if label == "OK":
        data["status_treinamento"] = "prototipo_ok_ativo"
    elif label == "NG":
        data["status_treinamento"] = "memoria_ng_protegida"
    _atomic_write_json(path, data)


def install_prototype_memory(
    knn_expert_cls,
    dataset_manager_cls,
    dataset_manager_module,
    best_match_module,
) -> None:
    """Instala compactação OK e proteção NG sem mudar o score de similaridade."""
    if getattr(knn_expert_cls, "_prototype_memory_installed", False):
        return

    original_load_all = knn_expert_cls._load_all
    original_analyze = knn_expert_cls.analyze
    original_save_sample = dataset_manager_cls.save_sample

    def comparator(first: dict, second: dict):
        return best_match_module.compare_anomaly_signatures(first, second)

    def load_all_with_prototypes(self):
        original_load_all(self)
        raw_ok = list(self.signatures_ok)
        raw_ng = list(self.signatures_ng)
        self.signatures_ok = condense_ok_records(raw_ok, comparator)
        self.signatures_ng = protect_ng_records(raw_ng)
        self.memory_prototype_stats = {
            "schema": PROTOTYPE_SCHEMA,
            "raw_ok_jsons": int(len(raw_ok)),
            "ok_prototypes": int(len(self.signatures_ok)),
            "raw_ng_jsons": int(len(raw_ng)),
            "protected_ng_prototypes": int(len(self.signatures_ng)),
            "ok_observations": int(
                sum(_record_occurrences(item) for item in self.signatures_ok)
            ),
            "quantity_influence": False,
        }

    def analyze_with_prototype_stats(self, *args, **kwargs):
        result = original_analyze(self, *args, **kwargs)
        if isinstance(result, dict):
            result["memory_prototype_stats"] = dict(
                getattr(self, "memory_prototype_stats", {})
            )
            result["memory_prototype_policy"] = "ok_condensed_ng_protected"
            result["memory_quantity_influence"] = False
        return result

    def save_sample_with_prototypes(
        ng_image,
        label,
        sample_image=None,
        aoi_info=None,
        analysis=None,
        save_images=False,
        source="",
        ai_decision="",
    ):
        normalized = str(label or "").strip().upper()
        signature = _query_signature(
            dataset_manager_module,
            ng_image,
            sample_image,
            aoi_info,
            analysis,
        )

        if normalized == "OK" and valid_anomaly_signature(signature):
            match = _find_persistent_ok_prototype(
                signature,
                aoi_info,
                comparator,
            )
            if match is not None:
                path, _data, similarity, _breakdown = match
                _update_prototype_metadata(
                    path,
                    label="OK",
                    protected=False,
                    increment=True,
                    similarity=similarity,
                    source=source,
                    ai_decision=ai_decision,
                    save_images=bool(save_images),
                    ng_image=ng_image,
                    sample_image=sample_image,
                )
                return str(path)

        path = original_save_sample(
            ng_image=ng_image,
            label=normalized,
            sample_image=sample_image,
            aoi_info=aoi_info,
            analysis=analysis,
            save_images=save_images,
            source=source,
            ai_decision=ai_decision,
        )
        if path and normalized in {"OK", "NG"}:
            _update_prototype_metadata(
                path,
                label=normalized,
                protected=(normalized == "NG"),
                increment=False,
                source=source,
                ai_decision=ai_decision,
            )
        return path

    knn_expert_cls._load_all = load_all_with_prototypes
    knn_expert_cls.analyze = analyze_with_prototype_stats
    dataset_manager_cls.save_sample = staticmethod(save_sample_with_prototypes)
    knn_expert_cls._prototype_memory_installed = True
    dataset_manager_cls._prototype_memory_installed = True


__all__ = [
    "OK_LEGACY_MEMORY_MERGE_SIMILARITY",
    "OK_PROTOTYPE_CONTEXT_MIN",
    "OK_PROTOTYPE_EPICENTER_MIN",
    "OK_PROTOTYPE_MERGE_SIMILARITY",
    "PROTOTYPE_SCHEMA",
    "condense_ok_records",
    "install_prototype_memory",
    "protect_ng_records",
]
