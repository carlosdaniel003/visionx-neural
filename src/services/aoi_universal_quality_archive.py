"""OFFLINE quality labels across every AOI routing category.

AOI category is NOT a physical defect label. Archive-folder OK/NG is
provisional and only authorizes candidate experiments with an explicit flag.
The 14 human contact-sheet assessments override *quality interpretation*
for exactly matching SHA-256 source files; IGNORAR is excluded.

No qualification.json, data file, KNN, active checkpoint or production route
is ever written by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import re

import numpy as np

from src.services.faltando_neural_qualification import QualificationStore
from src.scripts.train_faltando_cnn_v2_verified_archive import (
    _digest, _visual_key,
)

TRIAGE_RELATIVE = Path("docs/faltando_v2_operator_triage_20261016.json")
TRIAGE_SCHEMA = "visionx.faltando_v2_operator_contact_sheet_triage.v1"
MANIFEST_STAGE = "NEURAL_CROSS_CATEGORY_AOI_PREPARATION_ONLY"
CATEGORIES = ("FALTANDO", "INVERTIDO", "EMBORCADO", "DESLOCADO", "MUITO ADESIVO")


def _timestamp_bucket(source: str) -> str:
    """Fence same-minute images across AOI categories and illumination.

    A fence is NOT a physical event association. We keep the whole minute
    together only to reduce possible source leakage, and explicitly report
    that physical-piece independence remains unverified.
    """
    name = Path(source).name
    m = re.match(r"^(\d{4}-\d{2}-\d{2})_(\d{4}|\d{2}-\d{2})", name)
    if m:
        return m.group(1) + "_" + m.group(2).replace("-", "")
    return "legacy:" + name


def _load_review(root: Path) -> dict[str, dict]:
    path = root / TRIAGE_RELATIVE
    if path.is_symlink() or not path.is_file():
        raise ValueError("Arquivo obrigatório de triagem dos 14 casos ausente")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("schema") != TRIAGE_SCHEMA
            or doc.get("production_training_approved") is not False
            or doc.get("individual_qualification_file_created") is not False
            or doc.get("bulk_archive_labels_modified") is not False):
        raise ValueError("Triagem humana incompleta ou não corresponde ao contrato")
    cases = doc.get("cases", [])
    if len(cases) != 14:
        raise ValueError("Triagem não contém os 14 casos revisados")
    rows = {}
    for row in cases:
        source = row.get("source")
        if (not isinstance(source, str) or source in rows
                or row.get("visual_category") not in
                    ("FALTANDO", "INVERTIDO", "IGNORAR")
                or not re.fullmatch(r"[a-f0-9]{64}",
                                    str(row.get("source_sha256", "")))):
            raise ValueError("Entrada de revisão inválida")
        rows[source] = row
    if Counter(x["visual_category"] for x in rows.values()) != Counter({
        "FALTANDO": 9, "INVERTIDO": 4, "IGNORAR": 1,
    }):
        raise ValueError("Contagem da revisão humana divergente")
    return rows


def universal_archive_events(manifest_path: Path) -> tuple[list[dict], dict]:
    """Ingest all categories as SINGLE-VIEW provisional quality decisions."""
    store = QualificationStore(Path(manifest_path))
    doc = store.manifest
    if (doc.get("stage") != MANIFEST_STAGE
            or doc.get("all_aoi_categories_preparation_only") is not True
            or doc.get("aoi_category_not_visual_defect_type") is not True
            or doc.get("training_performed") is not False
            or doc.get("model_or_production_modified") is not False
            or doc.get("dataset_ready") is not False):
        raise ValueError("Manifesto multcategoria de preparação offline inválido")
    annotated = _load_review(store.root)
    for source, review in annotated.items():
        if source not in store.by_path:
            raise ValueError("Fonte da triagem ausente do manifesto: " + source)
        if store.by_path[source]["source_sha256"] != review["source_sha256"]:
            raise ValueError("SHA da triagem diverge da captura original: " + source)
        expected = store.expected_labels[source]
        kind = review["visual_category"]
        if kind != "IGNORAR" and expected != "NG":
            raise ValueError("Categoria física NG contradiz arquivo OK: " + source)

    recorded, duplicates, skipped = [], 0, 0
    seen = {}
    labels = Counter()
    categories = Counter()
    for source, item in sorted(store.by_path.items()):
        if item.get("status") != "EXTRACTED_PENDING_REVIEW":
            raise ValueError("Par inválido: " + source)
        if item.get("visual_defect_type") is not None:
            raise ValueError("Tipo físico não pode vir da categoria AOI: " + source)
        store._validate_source(source)
        label = store.expected_labels[source]
        if tuple(Path(source).parts[:2]) != (
            "public", label.lower() + "_archive",
        ):
            raise ValueError("Rótulo de pasta contradiz origem: " + source)
        review = annotated.get(source)
        if review and review["visual_category"] == "IGNORAR":
            skipped += 1
            continue
        category = item.get("category_hint")
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Categoria AOI ausente")
        origin, reference, test = store.files[source]
        if reference.is_symlink() or test.is_symlink():
            raise ValueError("Pares derivados com symlink não permitidos")
        signature = (_digest(reference), _digest(test))
        prior = seen.get(signature)
        if prior is not None:
            if prior != label:
                raise ValueError("Mesmo par visual consta simultaneamente OK e NG")
            duplicates += 1
            continue
        seen[signature] = label
        physical = review["visual_category"] if review else None
        row = {
            "id": "universal_provisional:" + source,
            "label": label, "category_hint": category,
            "visual_defect_type": physical,
            "label_provenance": (
                "HUMAN_CONTACT_SHEET_NG" if physical else "ARCHIVE_FOLDER_PROVISIONAL"
            ),
            "views": {item["lighting_mode"]: {
                "reference": reference, "test": test,
            }},
            "sources": [source],
            "source_sha256": item["source_sha256"],
            "split_group": "AOI_MINUTE:" + _timestamp_bucket(source),
            "event_link_verified": False,
        }
        recorded.append(row)
        labels[label] += 1
        categories[(category, label)] += 1
    if not recorded or labels["NG"] < 3 or labels["OK"] < 3:
        raise ValueError("Faltam rótulos OK/NG provisórios para experimentação")
    return recorded, {
        "source_manifest": str(store.manifest_path),
        "manifest_sha256": sha256(store.manifest_path.read_bytes()).hexdigest(),
        "source_pairs": len(store.by_path),
        "pairs_after_exact_deduplication": len(recorded),
        "exact_duplicates_skipped": duplicates,
        "human_ignored_sources": skipped,
        "human_reviewed_general_ng": sum(
            x["label_provenance"] == "HUMAN_CONTACT_SHEET_NG" for x in recorded
        ),
        "quality_source": "ARCHIVE_FOLDER_PROVISIONAL_WITH_SHA_PINNED_TRIAGE",
        "aoi_category_not_defect_type": True,
        "physical_event_association_verified": False,
        "human_reviewed_all_labels": False,
        "labels": dict(labels),
        "by_category_and_label": {
            category: {label: categories[(category, label)] for label in ("OK", "NG")}
            for category in sorted({c for c, _ in categories})
        },
        "source_pngs_unchanged": True,
        "production_training_approved": False,
    }


def split_universal_events(
    records: list[dict], *, seed: int = 42,
    test_fraction: float = .20, validation_fraction: float = .20,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Three non-overlapping splits with minute and visual duplication fences.

    Validation_fraction is of all examples. Approximate visual matching is
    conservative: it may merge distinct pieces; never silently split them.
    Strictly do not optimize on test data or claim physical-event independence.
    """
    if (isinstance(seed, bool) or not isinstance(seed, int)
            or not .10 <= test_fraction <= .30
            or not .10 <= validation_fraction <= .30
            or test_fraction + validation_fraction > .50):
        raise ValueError("Partição de 3 conjuntos inválida")
    if len({x["id"] for x in records}) != len(records):
        raise ValueError("IDs repetidos na base universal")
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        i, j = find(a), find(b)
        if i != j:
            parents[j] = i

    time_groups, exact, visual = {}, {}, defaultdict(list)
    for index, item in enumerate(records):
        if item.get("label") not in ("OK", "NG") or not item.get("views"):
            raise ValueError("Rótulo de qualidade ou vista inválida")
        minute = item.get("split_group")
        if not isinstance(minute, str) or not minute:
            raise ValueError("Timestamp de segurança ausente")
        if minute in time_groups:
            union(index, time_groups[minute])
        else:
            time_groups[minute] = index
        for light, pair in item["views"].items():
            key = (light, _digest(pair["reference"]), _digest(pair["test"]))
            if key in exact:
                union(index, exact[key])
            else:
                exact[key] = index
            rhash, rthumb = _visual_key(pair["reference"])
            thash, tthumb = _visual_key(pair["test"])
            visual[light].append((index, rhash, thash, rthumb, tthumb))
    for light, rows in visual.items():
        for ix, (i, ra, ta, rimg, timg) in enumerate(rows):
            for j, rb, tb, rimg2, timg2 in rows[ix+1:]:
                if (sum((a ^ b).bit_count() for a, b in zip(ra, rb)) <= 4
                        and sum((a ^ b).bit_count() for a, b in zip(ta, tb)) <= 4
                        and float(np.mean(np.abs(rimg.astype(np.int16) -
                                                 rimg2.astype(np.int16)))) <= 6
                        and float(np.mean(np.abs(timg.astype(np.int16) -
                                                 timg2.astype(np.int16)))) <= 6):
                    union(i, j)
    buckets = defaultdict(list)
    for i, record in enumerate(records):
        buckets[find(i)].append(record)
    if len(buckets) < 5:
        raise ValueError("Poucos grupos visualmente independentes")
    groups = list(buckets.values())
    counts = Counter((r["category_hint"], r["label"]) for r in records)
    targets = {"train": 1-test_fraction-validation_fraction,
               "validation": validation_fraction, "test": test_fraction}
    allocated = {fold: [] for fold in targets}
    acc = {fold: Counter() for fold in targets}
    rng = random.Random(seed)
    rng.shuffle(groups)
    # Allocate large/rare groups first; optimize categorical quality balance.
    groups.sort(key=lambda group: (
        -max(len(group) / max(1, counts[(r["category_hint"], r["label"])])
             for r in group),
        -len(group),
    ))
    for group in groups:
        here = Counter((r["category_hint"], r["label"]) for r in group)
        scores = {}
        for fold, ratio in targets.items():
            score = 0.
            for f, target_fraction in targets.items():
                for key, total in counts.items():
                    value = acc[f][key] + (here[key] if f == fold else 0)
                    goal = total * target_fraction
                    score += ((value-goal)**2) / max(1., total)
                total_count = sum(acc[f].values()) + (len(group) if f == fold else 0)
                score += ((total_count-len(records)*target_fraction)**2) / max(
                    1., len(records),
                )
            scores[fold] = score
        chosen = min(targets, key=lambda name: (scores[name], name))
        allocated[chosen].extend(group)
        acc[chosen].update(here)
    for fold, rows in allocated.items():
        if not rows or set(r["label"] for r in rows) != {"OK", "NG"}:
            raise ValueError("Fold " + fold + " sem OK e NG; não produzir métrica")
    names = {f: [r["id"] for r in rows] for f, rows in allocated.items()}
    if len(set(sum(names.values(), []))) != len(records):
        raise ValueError("Vazamento ou exclusão durante separação")
    return (
        allocated["train"], allocated["validation"], allocated["test"],
        {
            "seed": seed, "test_fraction_requested": test_fraction,
            "validation_fraction_requested": validation_fraction,
            "visual_near_duplicates_kept_together": True,
            "aoi_minute_all_categories_kept_together": True,
            "physical_event_association_verified": False,
            "group_count": len(groups),
            "folds": {
                fold: {
                    "records": len(rows),
                    "by_label": dict(Counter(r["label"] for r in rows)),
                    "by_aoi_category_and_label": {
                        cat: {
                            label: sum(r["category_hint"] == cat and r["label"] == label
                                       for r in rows)
                            for label in ("OK", "NG")
                        } for cat in sorted(set(r["category_hint"] for r in records))
                    },
                    "source_ids": names[fold],
                } for fold, rows in allocated.items()
            },
        },
    )
