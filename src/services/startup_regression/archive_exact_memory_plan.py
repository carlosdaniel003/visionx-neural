"""Proposta SOMENTE LEITURA de índice visual KNN para screenshots OK/NG.

Usa OCR e recortes AOI reais, hashes de referência/teste e o contrato
VerifiedKNNMemory._key. O rótulo do arquivo OK/NG é apenas expectativa;
não é confirmação humana. Este módulo NUNCA salva uma memória, nem a
cria em public/dataset ou altera o fluxo de operação.

Uma coincidência com uma entrada v3 humana já válida indica EXISTENTE;
uma v2 auditável indica apenas simulação; demais casos ficam pendentes.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.core.strict_category_memory import canonical_memory_category
from src.core.verified_memory_router import VerifiedKNNMemory
from src.services.faltando_cross_category_audit import _archive_image
from src.services.image_archive_dedup import image_fingerprint
from src.services.startup_regression.archive_inventory import inventory_archives
from src.services.startup_regression.archive_reconciler import (
    scan_memory_dataset, _index_records,
)
from src.utils.text_normalizer import normalize_aoi_text

SCHEMA = "visionx.archive_exact_memory_plan.v1"


def plan_archive_exact_memory(
    root: Path, *, inventory: dict | None = None,
    extractor: Callable | None = None,
    memory_rows: list | None = None,
    progress: Callable | None = None,
) -> dict:
    root = Path(root).expanduser().resolve()
    inv = inventory if inventory is not None else inventory_archives(root)
    if inv.get("schema") != "visionx.archive_inventory.v1":
        raise ValueError("Inventário de arquivo incompatível")
    items = inv.get("images")
    if (not isinstance(items, list) or not items
            or len(items) != inv.get("summary", {}).get("png_count")):
        raise ValueError("Inventário sem cobertura completa")
    if extractor is None:
        from src.services.faltando_neural_dataset import AOIPairExtractor
        extractor = AOIPairExtractor()
    if memory_rows is None:
        memory_rows = scan_memory_dataset(root)["rows"]
    indexes = _index_records(memory_rows)
    crossed = {
        p for conflict in inv.get("cross_label_conflicts", [])
        for p in conflict.get("paths", [])
    }
    cases = []
    keys = {}
    for i, item in enumerate(items, 1):
        path = item.get("path", "")
        label = item.get("expected_label")
        row = {
            "source_path": path,
            "source_file_sha256": item.get("file_sha256"),
            "expected_archive_label": label,
            "archive_label_human_verified": False,
            "category_hint": item.get("category_hint"),
            "lighting_mode": item.get("lighting_mode"),
            "lighting_source": item.get("lighting_source"),
            "event_id": item.get("event_id"),
            "manifest_links": item.get("manifest_links", []),
            "status": "NAO_ANALISADO",
            "observed_ocr": None,
            "reference_pixel_sha256": None,
            "test_pixel_sha256": None,
            "verified_memory_json_paths": [],
            "legacy_simulated_json_paths": [],
            "migration_ready": False,
            "error": None,
        }
        try:
            if item.get("status") != "VALID_PNG" or path in crossed:
                row["status"] = "PNG_OU_ROTULOS_INVALIDOS"
            else:
                frame = _archive_image(root, item)
                reference, test, info = extractor(frame)
                if not isinstance(info, dict):
                    raise ValueError("Extrator AOI não devolveu OCR")
                category, value = normalize_aoi_text(info.get("value", ""))
                row["observed_ocr"] = {
                    "board": str(info.get("board", "") or ""),
                    "parts": str(info.get("parts", "") or ""),
                    "value": str(value or ""),
                    "category": str(category or ""),
                }
                normalized_category = canonical_memory_category(category)
                hint = item.get("category_hint")
                if normalized_category in ("", "UNKNOWN") or not all(
                    row["observed_ocr"][name].strip()
                    for name in ("board", "parts", "value")
                ):
                    row["status"] = "OCR_INVALIDO"
                elif hint not in ("UNKNOWN", None, "") and (
                    canonical_memory_category(hint) != normalized_category
                ):
                    row["status"] = "CATEGORIA_OCR_DIVERGENTE"
                else:
                    key = VerifiedKNNMemory._key(
                        {
                            **info, "category": category, "value": value,
                            "lighting_mode": item.get("lighting_mode", "SIDE"),
                        },
                        image_fingerprint(reference),
                        image_fingerprint(test),
                    )
                    if key is None:
                        row["status"] = "IDENTIDADE_VISUAL_INCOMPLETA"
                    else:
                        row["reference_pixel_sha256"] = key[-2]
                        row["test_pixel_sha256"] = key[-1]
                        verified = indexes["verified"].get(key, [])
                        legacy = indexes["legacy_simulated_exact"].get(key, [])
                        row["verified_memory_json_paths"] = sorted(
                            r["path"] for r in verified
                        )
                        row["legacy_simulated_json_paths"] = sorted(
                            r["path"] for r in legacy
                        )
                        labels = {r["label"] for r in verified + legacy}
                        if len(labels) > 1 or (labels and label not in labels):
                            row["status"] = "CONFLITO_COM_MEMORIA_EXISTENTE"
                        elif verified:
                            row["status"] = "JA_VERIFICADO_EXATO_V3"
                        elif legacy:
                            row["status"] = "PAR_LEGADO_EXATO_PENDENTE"
                        elif item.get("lighting_source") == "LEGACY_DEFAULT":
                            row["status"] = "PENDENTE_ORIGEM_HUMANA_E_LUZ"
                        else:
                            row["status"] = "PENDENTE_ORIGEM_HUMANA"
                        keys.setdefault(key, []).append(row)
        except Exception as exc:
            row["status"] = "EXTRACAO_OU_LEITURA_INVALIDA"
            row["error"] = f"{type(exc).__name__}: {exc}"
        cases.append(row)
        if progress is not None:
            progress(i, len(items), path)

    # Mesmo com PNGs de screenshot diferentes, uma identidade de par
    # ref/test repetida entre OK e NG impede qualquer futura proposta.
    for entries in keys.values():
        if len({r["expected_archive_label"] for r in entries}) > 1:
            for row in entries:
                row["status"] = "CONFLITO_DE_PAR_VISUAL_NO_ACERVO"

    counts = Counter(row["status"] for row in cases)
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": "READ_ONLY_PLAN_NOT_A_MEMORY",
        "png_count": len(cases),
        "status_counts": dict(sorted(counts.items())),
        "by_label": {
            label: dict(sorted(Counter(
                r["status"] for r in cases
                if r["expected_archive_label"] == label
            ).items()))
            for label in ("OK", "NG")
        },
        "verified_existing": counts["JA_VERIFICADO_EXATO_V3"],
        "legacy_exact_pending": counts["PAR_LEGADO_EXATO_PENDENTE"],
        "pending_human_evidence": sum(
            counts[s] for s in (
                "PENDENTE_ORIGEM_HUMANA",
                "PENDENTE_ORIGEM_HUMANA_E_LUZ",
            )
        ),
        "ready_for_import": 0,
        "archive_label_used_as_human_approval": False,
        "writes_to_dataset": False,
        "trains": False,
        "startup_gate_enabled": False,
        "cases": cases,
        "note": (
            "Arquivo OK/NG não confirma decisão humana; nome SIDE não "
            "confirma evento. Nenhum arquivo foi importado, e "
            "0 casos foram autorizados para migração automática."
        ),
    }


__all__ = ["SCHEMA", "plan_archive_exact_memory"]
