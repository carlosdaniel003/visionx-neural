"""Opt-in offline dataset for CNN FALTANDO V2 experiments.

Folder names supply *provisional* labels, never fabricated pair-by-pair reviews.
This is deliberately separate from qualified_archive_events and cannot promote
a candidate to production or publish live_active.json.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path

from src.services.faltando_neural_dataset import _group_hint
from src.services.faltando_neural_qualification import QualificationStore


def archive_labeled_experimental_events(manifest_path: Path) -> tuple[list[dict], dict]:
    """Consume existing AOI pairs using operator-authorized folder labels.

    SIDE/TOP/MID filename matches are only *split groups* to avoid leakage.
    They never become verified multi-light events or supervision of a single
    physical component. Individually reviewed labels, if any, take precedence.
    """
    store = QualificationStore(Path(manifest_path))
    rows = []
    duplicates = 0
    human_cases = 0
    seen: dict[tuple[str, str], str] = {}
    counts = Counter()
    for source, item in sorted(store.by_path.items()):
        if item.get("status") != "EXTRACTED_PENDING_REVIEW":
            raise ValueError("Acervo experimental contém par não extraído: " + source)
        label = store.expected_labels[source]
        expected_prefix = ("public", label.lower() + "_archive")
        if tuple(Path(source).parts[:2]) != expected_prefix:
            raise ValueError("Pasta de origem contradiz rótulo do manifesto: " + source)
        if item.get("category_hint") != "FALTANDO":
            raise ValueError("Categoria de origem fora do escopo FALTANDO: " + source)
        store._validate_source(source)
        review = store.case_reviews.get(source)
        if review:
            state = review.get("status")
            if state == "REJECTED":
                raise ValueError("Par previamente rejeitado não pode ser aprovado em lote: " + source)
            if state in ("CONFIRMED_OK", "CONFIRMED_NG"):
                if (review.get("pair_inspected") is not True
                        or review.get("source_sha256") != item["source_sha256"]):
                    raise ValueError("Revisão individual inconsistente: " + source)
                label = state.split("_", 1)[1]
                human_cases += 1
        origin, reference, test = store.files[source]
        if reference.is_symlink() or test.is_symlink():
            raise ValueError("Recorte derivado não pode ser symlink: " + source)
        signature = (sha256(reference.read_bytes()).hexdigest(),
                     sha256(test.read_bytes()).hexdigest())
        previous = seen.get(signature)
        if previous is not None:
            if previous != label:
                raise ValueError("Conflito NG/OK: mesmo par visual no arquivo")
            duplicates += 1
            continue
        seen[signature] = label
        hint = _group_hint(source)
        event_id = item.get("event_id")
        split_group = (
            label + "::name::" + hint if hint
            else label + "::event::" + str(event_id) if event_id
            else None
        )
        rows.append({
            "id": "archive_provisional:" + source,
            "label": label,
            "views": {item["lighting_mode"]: {
                "reference": reference, "test": test,
            }},
            "sources": [source],
            "component": None,  # OCR Board/Parts is not a physical-piece ID.
            "split_group": split_group,
            "grouping": "PROVISIONAL_SINGLE_VIEW",
        })
        counts[label] += 1
    if not rows:
        raise ValueError("Nenhum par extraído utilizável para experimento")
    return rows, {
        "manifest": str(store.manifest_path),
        "manifest_sha256": sha256(store.manifest_path.read_bytes()).hexdigest(),
        "qualification_sha256": (
            sha256(store.output.read_bytes()).hexdigest() if store.output.exists()
            else None
        ),
        "label_source": "ARCHIVE_FOLDER_OPERATOR_BULK_AUTHORIZED",
        "individual_review_required_for_production": True,
        "individual_pair_review_complete": False,
        "operator_bulk_authorization": True,
        "human_reviewed_overrides": human_cases,
        "source_pairs": len(store.by_path),
        "provisional_single_view_events": len(rows),
        "exact_duplicates_skipped": duplicates,
        "labels_used": dict(counts),
        "name_only_triplets_not_verified": len(store.groups),
        "human_multilight_groups": 0,
        "original_sources_unchanged": True,
    }
