"""SHA-pinned operator quality corrections: offline only, never mutates archives.

The six already visually reviewed AOI pairs may be registered in the existing
QualificationStore; the other decisions are valid quality evidence but do not
manufacture pair inspection or physical-event linkage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from src.services.faltando_neural_qualification import QualificationStore

SCHEMA = "visionx.operator_confirmed_quality.v1"
EVIDENCE = Path("docs/odin_operator_confirmed_quality_20261026.json")
HEX = re.compile(r"^[a-f0-9]{64}$")


def verified_operator_labels(root: Path) -> dict[str, dict]:
    root = Path(root).resolve()
    evidence_path = root / EVIDENCE
    if evidence_path.is_symlink() or not evidence_path.is_file():
        raise ValueError("Arquivo de decisões humanas ausente ou symlink")
    doc = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (doc.get("schema") != SCHEMA
            or doc.get("approved_for_production") is not False
            or doc.get("physical_event_groups_verified") is not False
            or doc.get("reviewed_on") != "2026-10-26"):
        raise ValueError("Proveniência da revisão humana incompatível")
    entries = doc.get("labels")
    if not isinstance(entries, list) or len(entries) != 15:
        raise ValueError("Revisão humana deve conter os 15 PNGs confirmados")
    verified: dict[str, dict] = {}
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("Registro humano inválido")
        source = row.get("source_path")
        digest = row.get("source_sha256")
        label = row.get("quality_label")
        if (not isinstance(source, str) or source in verified
                or not HEX.fullmatch(str(digest))
                or label not in ("OK", "NG")
                or type(row.get("pair_inspected_by_operator")) is not bool):
            raise ValueError("Registro de classe/hash repetido ou inválido")
        rel = Path(source)
        if (rel.is_absolute() or ".." in rel.parts
                or len(rel.parts) != 3
                or tuple(rel.parts[:2]) not in (
                    ("public", "ok_archive"), ("public", "ng_archive"))
                or rel.suffix.lower() != ".png"):
            raise ValueError("PNG fora dos arquivos protegidos: " + source)
        original = root / rel
        if original.is_symlink() or not original.is_file():
            raise ValueError("PNG não encontrado ou symlink: " + source)
        if not original.resolve().is_relative_to(root / "public"):
            raise ValueError("Origem fora do acervo")
        if sha256(original.read_bytes()).hexdigest() != digest:
            raise ValueError("PNG divergente da confirmação humana: " + source)
        verified[source] = row
    counts = Counter(row["quality_label"] for row in verified.values())
    if counts != Counter({"NG": 12, "OK": 3}):
        raise ValueError("Contagens de decisões humanas incompatíveis")
    conflict = verified.get("public/ng_archive/2026-10-21_1414_FALTANDO.png")
    if not conflict or conflict["quality_label"] != "OK":
        raise ValueError("Correção crítica do caso 7 ausente")
    return verified


def qualify_inspected_pairs(root: Path, manifest: Path, *, apply: bool) -> dict:
    root = Path(root).resolve()
    labels = verified_operator_labels(root)
    target = Path(manifest)
    target = target if target.is_absolute() else root / target
    store = QualificationStore(target)
    if store.root != root:
        raise ValueError("Manifesto pertence a outra estação/raiz")
    ready, without_pair = [], []
    for source, row in labels.items():
        sample = store.by_path.get(source)
        if not sample or not row["pair_inspected_by_operator"]:
            without_pair.append(source)
            continue
        if sample["source_sha256"] != row["source_sha256"]:
            raise ValueError("SHA do par divergente: " + source)
        store._validate_source(source)
        target_status = "CONFIRMED_" + row["quality_label"]
        previous = store.case_reviews.get(source)
        if previous and previous.get("status") != target_status:
            raise ValueError("Revisão anterior contraditória: " + source)
        if previous and previous.get("pair_inspected") is not True:
            raise ValueError("Revisão anterior sem inspeção do par: " + source)
        ready.append((source, target_status, row))
    if len(ready) != 6:
        raise ValueError(
            "Manifesto precisa conter seis pares humanos extraídos e íntegros; "
            f"encontrados {len(ready)}"
        )
    changed = 0
    if apply:
        for source, status, row in ready:
            if store.case_status(source) == status:
                continue
            store.review_case(
                source, status,
                "Revisão humana explícita 26/10/2026; defeito físico: "
                + row["physical_type"] + "; não confirma identidade multilight.",
            )
            changed += 1
    return {
        "status": "QUALIFICATION_UPDATED" if apply else "AUDIT_ONLY",
        "confirmed_source_images": len(labels),
        "ready_pair_reviews": len(ready),
        "pair_reviews_written": changed,
        "images_without_qualified_pairs": len(without_pair),
        "case_7_corrected_quality": labels[
            "public/ng_archive/2026-10-21_1414_FALTANDO.png"
        ]["quality_label"],
        "group_reviews_created": 0,
        "original_archive_modified": False,
        "training_performed": False,
        "production_approved": False,
        "xp_commands_sent": 0,
        "qualification_file": str(store.output) if apply else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Importar revisão humana CNN sem treinar")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("reports/faltando_neural/run_20261016T154441_358349Z/manifest.json"),
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(
        qualify_inspected_pairs(args.root, args.manifest, apply=args.apply),
        indent=2, ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
