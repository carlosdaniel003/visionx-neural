"""Revisão humana de QUALIDADE (OK/NG), exclusivamente para CNN V3 offline.

A categoria textual da AOI não é consultada. O nome do PNG funciona somente
como identificador para encontrar a captura; o SHA garante a identidade.
A imagem ambígua é excluída mesmo se o manifesto antigo ainda a contiver.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

REVIEW_PATH = Path("docs/aoi_v3_operator_image_quality_20261104.json")
REVIEW_SCHEMA = "visionx.aoi_v3_operator_quality_by_image.v1"
SHA256 = re.compile(r"^[a-f0-9]{64}$")
OBSERVATIONS = {
    "NONE", "ADHESIVE_ON_TERMINAL", "UPSIDE_DOWN_COMPONENT", "UNCERTAIN",
}


def load_v3_image_reviews(store) -> dict[str, dict]:
    """Verifica 15 decisões por manifesto + SHA da captura; não muda arquivos.

    A revisão é transportável para um manifesto posterior quando cada PNG
    confirmado ainda existe no inventário, tem mesmo SHA e é legível. O
    IGNORE pode já ter sido apagado do acervo, sem invalidar a auditoria.
    """
    root = store.root.resolve()
    evidence = root / REVIEW_PATH
    if evidence.is_symlink() or not evidence.is_file():
        raise ValueError("Revisão humana de 15 imagens V3 ausente/symlink")
    doc = json.loads(evidence.read_text(encoding="utf-8"))
    if (
        not isinstance(doc, dict)
        or doc.get("schema") != REVIEW_SCHEMA
        or doc.get("reviewed_on") != "2026-11-04"
        or doc.get("review_scope") != "FIRST_V3_15_ERRORS"
        or doc.get("quality_not_aoi_category") is not True
        or doc.get("allowed_for_offline_training") is not True
        or doc.get("approved_for_production") is not False
        or doc.get("physical_event_association_verified") is not False
        or not SHA256.fullmatch(str(doc.get("source_manifest_sha256", "")))
        or not SHA256.fullmatch(str(
            doc.get("source_training_report_candidate_sha256", "")
        ))
    ):
        raise ValueError("Proveniência da revisão humana V3 inválida")
    entries = doc.get("entries")
    if not isinstance(entries, list) or len(entries) != 15:
        raise ValueError("Esperados exatamente 14 rótulos e 1 exclusão V3")
    result: dict[str, dict] = {}
    cases = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Revisão de imagem V3 inválida")
        source = entry.get("source_path")
        label = entry.get("quality_label")
        case = entry.get("case_number")
        digest = entry.get("source_sha256")
        issue = entry.get("physical_observation")
        if (
            type(case) is not int or not 1 <= case <= 15
            or case in cases or not isinstance(source, str)
            or source in result or not SHA256.fullmatch(str(digest))
            or label not in {"OK", "NG", "IGNORE"}
            or issue not in OBSERVATIONS
            or (label == "OK" and issue != "NONE")
            or (label == "IGNORE" and issue != "UNCERTAIN")
            or (label == "NG" and issue not in {
                "ADHESIVE_ON_TERMINAL", "UPSIDE_DOWN_COMPONENT",
            })
        ):
            raise ValueError("Rótulo humano V3 inválido/repetido")
        cases.add(case)
        parts = Path(source)
        if (
            parts.is_absolute() or len(parts.parts) != 3
            or parts.parts[0] != "public"
            or parts.parts[1] not in ("ok_archive", "ng_archive")
            or parts.suffix.lower() != ".png"
            or ".." in parts.parts
        ):
            raise ValueError("Caminho fonte inválido da revisão V3")
        row = store.by_path.get(source)
        if row is None:
            if label == "IGNORE":
                result[source] = entry
                continue
            raise ValueError("PNG confirmado ausente do manifesto V3: " + source)
        if (
            row.get("status") != "EXTRACTED_PENDING_REVIEW"
            or row.get("source_sha256") != digest
        ):
            raise ValueError("SHA/extração divergente na revisão V3: " + source)
        if label == "IGNORE":
            origin = store.files[source][0]
            if origin.is_symlink():
                raise ValueError("Fonte IGNORE via symlink: " + source)
            if origin.exists() and sha256(origin.read_bytes()).hexdigest() != digest:
                raise ValueError("SHA da imagem ignorada foi alterado: " + source)
        else:
            # Única garantia material do julgamento: o PNG original
            # validado; não há categoria AOI no veredito.
            store._validate_source(source)
        result[source] = entry
    if (
        cases != set(range(1, 16))
        or Counter(row["quality_label"] for row in result.values())
        != Counter({"NG": 9, "OK": 5, "IGNORE": 1})
        or Counter(row["physical_observation"] for row in result.values())
        != Counter({
            "ADHESIVE_ON_TERMINAL": 6,
            "UPSIDE_DOWN_COMPONENT": 3, "NONE": 5, "UNCERTAIN": 1,
        })
    ):
        raise ValueError("As 15 confirmações humanas estão incompletas")
    return result


__all__ = ["load_v3_image_reviews", "REVIEW_PATH", "REVIEW_SCHEMA"]
