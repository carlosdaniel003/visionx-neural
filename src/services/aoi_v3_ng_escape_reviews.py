"""Revisões humanas dos NG que a CNN V3 classificou como OK (05/11).

Somente offline e com opt-in. A referência é o SHA da captura original,
não a categoria AOI, nem a previsão da CNN. A revisão não treina pesos
por si só; o treinamento escolhe pesos exclusivamente pela validação.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

REVIEW_PATH = Path("docs/aoi_v3_operator_ng_escape_review_20261105.json")
REVIEW_SCHEMA = "visionx.aoi_v3_operator_ng_escapes.v1"
BASELINE_CHECKPOINT_SHA256 = (
    "fee6e062f98c8e7ddca39f4a16f8ebbfead1365b6510a23739e7cb0045c80fce"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
OBSERVATIONS = {
    "POLARITY_ORIENTATION_MISMATCH",
    "DARK_ADHESIVE_ON_PAD",
    "NG_CONFIRMED_TYPE_NOT_SPECIFIED",
}


def load_v3_ng_escape_reviews(store) -> tuple[dict[str, dict], str]:
    """Exige as 25 fontes originais íntegras do manifesto, ou falha.

    Não exige correspondência entre categoria AOI e o defeito físico.
    Um PNG removido pelo operador jamais é reintroduzido no dataset.
    """
    root = Path(store.root).resolve()
    path = root / REVIEW_PATH
    if path.is_symlink() or not path.is_file():
        raise ValueError("Revisão dos NG V3 ausente ou symlink")
    raw = path.read_bytes()
    doc = json.loads(raw)
    if (
        not isinstance(doc, dict)
        or doc.get("schema") != REVIEW_SCHEMA
        or doc.get("reviewed_on") != "2026-11-05"
        or doc.get("review_scope") != "ALL_26_NG_TO_OK_FROM_1111_PNG_AUDIT"
        or doc.get("reviewed_by") != "AOI_OPERATOR"
        or doc.get("source_audit_schema") != "visionx.aoi_v3_full_archive_audit.v1"
        or doc.get("source_audit_checkpoint_sha256") != BASELINE_CHECKPOINT_SHA256
        or doc.get("source_audit_totals") != {
            "png": 1111, "NG_to_OK": 26,
            "ignored_deleted": 1, "NG_confirmed": 25,
        }
        or doc.get("deleted_excluded_source") !=
            "public/ng_archive/2026-10-06_1615_MUITO_ADESIVO.png"
        or doc.get("labels_physically_confirmed_for_sources") is not True
        or doc.get("quality_not_aoi_category") is not True
        or doc.get("physical_event_association_verified") is not False
        or doc.get("allowed_for_opt_in_offline_training") is not True
        or doc.get("approved_for_production") is not False
        or doc.get("automatic_xp_authorized") is not False
    ):
        raise ValueError("Proveniência dos 25 NG revisados inválida")
    entries = doc.get("entries")
    if not isinstance(entries, list) or len(entries) != 25:
        raise ValueError("Esperados exatamente 25 NG confirmados")
    result = {}
    cases = set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("Revisão NG não é objeto")
        source = item.get("source_path")
        digest = item.get("source_sha256")
        n = item.get("case_number")
        if (
            type(n) is not int or n not in range(2, 27) or n in cases
            or not isinstance(source, str)
            or source in result
            or not SHA256.fullmatch(str(digest))
            or item.get("quality_label") != "NG"
            or item.get("physical_observation") not in OBSERVATIONS
        ):
            raise ValueError("Identidade/rótulo de NG revisado inválido")
        cases.add(n)
        relative = Path(source)
        if (
            relative.is_absolute()
            or len(relative.parts) != 3
            or relative.parts[:2] != ("public", "ng_archive")
            or relative.suffix.lower() != ".png"
            or ".." in relative.parts
            or source == doc["deleted_excluded_source"]
        ):
            raise ValueError("Origem NG revisada fora do acervo permitido")
        record = store.by_path.get(source)
        if record is None or record.get("status") != "EXTRACTED_PENDING_REVIEW":
            raise ValueError("NG confirmado não está no manifesto: " + source)
        if record.get("source_sha256") != digest:
            raise ValueError("SHA do NG confirmado diverge do manifesto: " + source)
        # _validate_source verifica origem e recortes derivados no staging.
        store._validate_source(source)
        original = root / source
        if (original.is_symlink() or not original.is_file()
                or sha256(original.read_bytes()).hexdigest() != digest):
            raise ValueError("PNG NG original ausente ou modificado: " + source)
        result[source] = dict(item)
    if cases != set(range(2, 27)):
        raise ValueError("Conjunto de casos NG incompleto")
    if Counter(x["quality_label"] for x in result.values()) != {"NG": 25}:
        raise ValueError("As 25 decisões humanas NG estão incompletas")
    return result, sha256(raw).hexdigest()


__all__ = [
    "load_v3_ng_escape_reviews", "REVIEW_PATH", "REVIEW_SCHEMA",
    "BASELINE_CHECKPOINT_SHA256",
]
