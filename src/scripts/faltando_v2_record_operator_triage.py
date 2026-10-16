"""Persist operator contact-sheet triage without rewriting the FALTANDO archive.

This is a *read-only verification* of an explicit user classification of 14
diagnostic PNGs. INVERTIDO images are quarantined from *future FALTANDO-only*
development, never relabelled OK. The operator classification does not
approve all 420 examples, does not change any historical metric, and does
not create QualificationStore's separately validated qualification.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

from src.services.faltando_neural_qualification import QualificationStore

SCHEMA = "visionx.faltando_v2_operator_contact_sheet_triage.v1"
AUDIT_SCHEMA = "visionx.faltando_v2_failed_pair_audit.v1"
CATEGORIES = frozenset(("FALTANDO", "INVERTIDO", "IGNORAR"))
AUDIT_SECTIONS = ("historico", "treino", "holdout")
REVIEW_RELATIVE = Path("docs/faltando_v2_operator_triage_20261016.json")


def _load_bounded_json(root: Path, filename: Path, parent: Path, name: str) -> dict:
    given = filename if filename.is_absolute() else root / filename
    if given.is_symlink() or not given.is_file():
        raise ValueError("Arquivo ausente ou symlink: " + str(filename))
    resolved = given.resolve()
    if resolved.name != name or not resolved.is_relative_to(parent.resolve()):
        raise ValueError("Arquivo fora da localização permitida: " + str(filename))
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON não é objeto")
    return data


def verify_operator_triage(
    root: Path, manifest: Path, audit: Path,
    review: Path = REVIEW_RELATIVE,
) -> dict:
    """Validate every annotated source against original PNG SHA and audit."""
    root = Path(root).resolve()
    manifest = Path(manifest)
    manifest = manifest if manifest.is_absolute() else root / manifest
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("Manifesto ausente ou symlink")
    store = QualificationStore(manifest)
    if store.root != root:
        raise ValueError("Manifesto pertence a outra raiz")
    manifest_sha = sha256(manifest.read_bytes()).hexdigest()
    review_obj = _load_bounded_json(
        root, Path(review), root / "docs", REVIEW_RELATIVE.name,
    )
    audit_obj = _load_bounded_json(
        root, Path(audit),
        root / "reports/faltando_neural/failed_pair_audits",
        "audit_index.json",
    )
    if (
        review_obj.get("schema") != SCHEMA
        or review_obj.get("manifest_sha256") != manifest_sha
        or review_obj.get("individual_qualification_file_created") is not False
        or review_obj.get("production_training_approved") is not False
        or review_obj.get("bulk_archive_labels_modified") is not False
    ):
        raise ValueError("Revisão do operador incompatível com o manifesto")
    if (
        audit_obj.get("schema") != AUDIT_SCHEMA
        or audit_obj.get("manifest_sha256") != manifest_sha
        or audit_obj.get("source_candidate_sha256")
        != review_obj.get("audit_candidate_sha256")
        or audit_obj.get("read_only") is not True
    ):
        raise ValueError("Auditoria de origem inválida/incompatível")
    cases = review_obj.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Não há casos rotulados pelo operador")
    review_map = {}
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("Entrada de triagem não é objeto")
        src, kind = row.get("source"), row.get("visual_category")
        if (not isinstance(src, str) or src not in store.by_path
                or src in review_map or kind not in CATEGORIES
                or not isinstance(row.get("note"), str)
                or not row["note"].strip()):
            raise ValueError("Revisão inexistente, duplicada ou inválida: " + str(src))
        if (
            row.get("source_sha256") != store.by_path[src]["source_sha256"]
            or (kind in ("FALTANDO", "INVERTIDO")
                and store.expected_labels[src] != "NG")
        ):
            raise ValueError("SHA da fonte ou rótulo da pasta divergente: " + src)
        store._validate_source(src)
        review_map[src] = row

    audit_sources = {}
    presence = {}
    diagnostic_counts = Counter()
    for section in AUDIT_SECTIONS:
        records = audit_obj.get("cases", {}).get(section)
        if not isinstance(records, list):
            raise ValueError("Seção da auditoria ausente: " + section)
        for row in records:
            source = row.get("source")
            if source not in store.by_path or source not in review_map:
                raise ValueError("Caso na auditoria não existe na revisão: " + str(source))
            expected_sha = store.by_path[source]["source_sha256"]
            if row.get("source_sha256") != expected_sha or (
                source in audit_sources and audit_sources[source] != expected_sha
            ):
                raise ValueError("Hash auditado incompatível: " + source)
            if row.get("label_from_folder") != store.expected_labels[source]:
                raise ValueError("Rótulo da pasta no relatório foi alterado: " + source)
            audit_sources[source] = expected_sha
            presence.setdefault(source, []).append(section)
            diagnostic_counts[section] += 1

    if set(review_map) != set(audit_sources):
        raise ValueError("Revisão e auditoria não contêm exatamente as mesmas fontes")
    if audit_obj.get("unique_source_count") != len(audit_sources):
        raise ValueError("Contagem de fontes originais da auditoria divergente")
    if (len(review_map) != 14 and
            review_obj.get("operator_review_scope") == "ONLY_14_DIAGNOSTIC_SOURCE_PNGS"):
        raise ValueError("A triagem registrada não tem 14 fontes")

    categories = Counter(row["visual_category"] for row in cases)
    # For the user-specific annotation this is an invariant, not a target metric.
    if review_obj.get("operator_review_scope") == "ONLY_14_DIAGNOSTIC_SOURCE_PNGS" and (
        categories != Counter({"FALTANDO": 9, "INVERTIDO": 4, "IGNORAR": 1})
    ):
        raise ValueError("Composição da triagem dos 14 exemplos divergiu")

    quarantined = sorted(
        source for source, row in review_map.items()
        if row["visual_category"] in ("INVERTIDO", "IGNORAR")
    )
    confirmed_missing = sorted(
        source for source, row in review_map.items()
        if row["visual_category"] == "FALTANDO"
    )
    return {
        "schema": "visionx.faltando_v2_operator_triage_report.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_sha,
        "source_audit_candidate_sha256": audit_obj["source_candidate_sha256"],
        "annotation_source": review_obj["source"],
        "source_count": len(review_map),
        "visual_categories": dict(categories),
        "audit_occurrences": dict(diagnostic_counts),
        "reviewed_missing_from_contact_sheets": confirmed_missing,
        "quarantine_for_future_faltando_experiments": quarantined,
        "inverted_not_relabelled_ok": True,
        "remaining_archive_pairs_unreviewed": True,
        "historical_metrics_unchanged": True,
        "holdout_previously_exposed": True,
        "recompute_accuracy_on_filtered_cases": False,
        "train_performed": False,
        "live_checkpoint_changed": False,
        "qualification_file_changed": False,
        "original_pngs_changed": False,
        "cross_category_auto_training_allowed": False,
        "production_approval_eligible": False,
        "cases": [
            {**review_map[source], "audit_sections": presence[source]}
            for source in sorted(review_map)
        ],
    }


def record_triage(
    root: Path, manifest: Path, audit: Path,
    review: Path = REVIEW_RELATIVE,
) -> Path:
    root = Path(root).resolve()
    result = verify_operator_triage(root, manifest, audit, review)
    destination = root / "reports/faltando_neural/operator_triage"
    directory = destination / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + "_" + uuid4().hex[:6]
    )
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "operator_triage_report.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print("TRIAGEM CONFERIDA:", result["visual_categories"], flush=True)
    print("Para futura FALTANDO, quarentena:", len(
        result["quarantine_for_future_faltando_experiments"]), flush=True)
    print("Treinamento/promoção: NÃO", flush=True)
    print("Relatório:", path, flush=True)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Registra triagem humana de 14 PNGs sem alterar o acervo"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-index", type=Path, required=True)
    args = parser.parse_args(argv)
    record_triage(args.root, args.manifest, args.audit_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
