"""Preparação offline DESLOCADO — mesmos recortes AOI usados pela CNN FALTANDO.

Lê OK/NG archive por inventário, extrai gabarito/teste e agrupa apenas
candidatos SIDE/TOP/MID pelo nome. Não move originais, não treina, não
autoriza Produção nem inventa defeitos NG que não existem.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any
import json

import cv2
import numpy as np

from src.services.faltando_neural_dataset import (
    AOIPairExtractor, LIGHTS, _group_hint, _read_png, _write_png,
)
from src.services.startup_regression.archive_inventory import inventory_archives

SCHEMA = "visionx.deslocado_neural_preparation.v1"
CATEGORY = "DESLOCADO"


def prepare_deslocado(
    root: Path, output_base: Path | None = None, *,
    extractor: Callable[[np.ndarray], tuple] | None = None,
    inventory: dict | None = None,
) -> tuple[dict[str, Any], Path]:
    root = Path(root).expanduser().resolve()
    reports_root = root / "reports"
    base = Path(output_base).expanduser().resolve() if output_base else (
        reports_root / "deslocado_neural"
    )
    if base == reports_root or reports_root not in base.parents:
        raise ValueError("Saída DESLOCADO deve ficar exclusivamente sob reports/")
    original = inventory if inventory is not None else inventory_archives(root)
    selected = sorted((
        item for item in original.get("images", [])
        if item.get("category_hint") == CATEGORY
        and item.get("expected_label") in ("OK", "NG")
    ), key=lambda x: x["path"])
    now = datetime.now(timezone.utc)
    folder = base / ("run_" + now.strftime("%Y%m%dT%H%M%S_%fZ"))
    folder.mkdir(parents=True, exist_ok=False)
    records = []
    groups = defaultdict(lambda: defaultdict(list))
    real_ng = 0

    for row in selected:
        source = root / row["path"]
        label = row["expected_label"]
        if label == "NG":
            real_ng += 1
        info = {
            "source_path": row["path"],
            "source_sha256": row.get("file_sha256", ""),
            "expected_label_from_archive": label,
            "category_hint": CATEGORY,
            "lighting_mode": row["lighting_mode"],
            "lighting_source": row.get("lighting_source"),
            "event_id": row.get("event_id"),
            "status": "INVALID_SOURCE",
            "reference_path": None,
            "test_path": None,
            "training_ready": False,
            "review_required": ["VERIFY_SOURCE_LABEL", "VERIFY_PAIR",
                                "VERIFY_INDEPENDENT_EVENT"],
        }
        records.append(info)
        if row.get("status") != "VALID_PNG":
            info["error"] = "Imagem de origem inválida"
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(root/"public"):
            info["error"] = "Origem fora do diretório público"
            continue
        hint = _group_hint(row["path"])
        if hint:
            groups[(label, hint)][row["lighting_mode"]].append(row["path"])
        try:
            frame = _read_png(source)
            if extractor is None:
                extractor = AOIPairExtractor()
            ref, test, ocr = extractor(frame)
            if not isinstance(ocr, dict) or any(
                not isinstance(im, np.ndarray) or im.ndim != 3
                or min(im.shape[:2]) < 12 for im in (ref, test)
            ):
                raise ValueError("Gabarito/teste ou metadados AOI inválidos")
            unique = label.lower()+"_"+row["file_sha256"]
            base_pair = Path("pairs") / unique
            _write_png(folder/base_pair/"reference.png", ref)
            _write_png(folder/base_pair/"test.png", test)
            info.update({
                "status": "EXTRACTED_PENDING_REVIEW",
                "reference_path": (base_pair/"reference.png").as_posix(),
                "test_path": (base_pair/"test.png").as_posix(),
                "reference_size": [ref.shape[1], ref.shape[0]],
                "test_size": [test.shape[1], test.shape[0]],
                "ocr_observed": {
                    k: str(ocr.get(k, "") or "")
                    for k in ("board", "parts", "value", "category")
                },
            })
            reported = str(ocr.get("category", "") or "").strip().upper()
            if reported and reported != CATEGORY:
                info["review_required"].append("OCR_CATEGORY_MISMATCH")
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            info["status"] = "EXTRACTION_FAILED"
            info["error"] = f"{type(exc).__name__}: {exc}"

    triples = []
    for (label, hint), per_light in sorted(groups.items()):
        if set(per_light) != set(LIGHTS) or any(
            len(per_light[mode]) != 1 for mode in LIGHTS
        ):
            continue
        triples.append({
            "id": hint,
            "label": label,
            "paths": {mode: per_light[mode][0] for mode in LIGHTS},
            "status": "NAME_ONLY_NEEDS_EVENT_VALIDATION",
        })
    statuses = Counter(r["status"] for r in records)
    by_label = dict(Counter(r["expected_label_from_archive"] for r in records))
    summary = {
        "total": len(records),
        "by_label": by_label,
        "by_lighting": dict(Counter(r["lighting_mode"] for r in records)),
        "extracted": statuses["EXTRACTED_PENDING_REVIEW"],
        "unusable": len(records)-statuses["EXTRACTED_PENDING_REVIEW"],
        "name_only_triplets": len(triples),
        "real_ng_count": real_ng,
        "has_real_ng_for_training": bool(real_ng),
        "real_ng_validation_ready": False,
    }
    report = {
        "schema": SCHEMA,
        "root": str(root),
        "generated_at_utc": now.isoformat(),
        "output_dir": str(folder),
        "stage": "DESLOCADO_DATA_PREPARATION_NO_REAL_NG_VALIDATION",
        "training_performed": False,
        "production_modified": False,
        "production_approved": False,
        "samples": records,
        "name_only_triplet_candidates": triples,
        "summary": summary,
    }
    (folder/"manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "CNN DESLOCADO — INVENTÁRIO / EXTRAÇÃO OFFLINE",
        f"Imagens: {summary['total']} | pares: {summary['extracted']}",
        f"OK: {by_label.get('OK', 0)} | NG reais: {real_ng}",
        f"SIDE/TOP/MID trincas candidatas: {len(triples)}",
        f"Falhas de extração: {summary['unusable']}",
        "Treino supervisionado OK/NG real: "
        + ("APENAS APÓS QUALIFICAÇÃO" if real_ng else "IMPOSSÍVEL SEM NG REAL"),
        "O motor físico DESLOCADO continua em produção.",
        "Não inferir event_id apenas pelo nome do screenshot.",
        "PNG de origem não alterado.",
    ]
    for row in records:
        if row["status"] != "EXTRACTED_PENDING_REVIEW":
            lines.append(
                f"{row['status']}: {row['source_path']} — {row.get('error', '')}"
            )
    (folder/"summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report, folder


def latest_deslocado_manifest(root: Path) -> Path:
    staging = Path(root).resolve()/"reports"/"deslocado_neural"
    paths = sorted(
        staging.glob("run_*/manifest.json"), reverse=True
    )
    if not paths:
        raise FileNotFoundError(
            "Nenhum dataset DESLOCADO. Execute a preparação primeiro."
        )
    return paths[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Extrair dataset CNN DESLOCADO sem alterar o ODIN"
    )
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    report, location = prepare_deslocado(args.root)
    print((location/"summary.txt").read_text(encoding="utf-8"))
    print("Manifesto:", location/"manifest.json")
    return 0 if report["summary"]["unusable"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
