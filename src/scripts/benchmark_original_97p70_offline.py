"""Read-only head-to-head: first 97.70%-holdout CNN V2 vs universal AOI.

Evaluates the SHA-pinned original aggressive candidate on EXACTLY the 106
sources in the universal candidate's already-published test partition.
Universal predictions are reused verbatim (NOT rerun, tuned, or retrained).
Checks immutable source hashes, model SHA, report origin and exact split IDs.
NO new model/threshold selection and absolutely NO production activation.

87-image 97.70% is original experimental holdout evidence, not a guarantee
that general NG/OK works; its 19 historical false-NG must remain visible.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from uuid import uuid4

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.scripts.faltando_cnn_v2_blend_candidates import (
    _inside, _sha, _verified_candidate,
)
from src.scripts.train_aoi_universal_quality_offline import (
    REPORT_SCHEMA as UNIVERSAL_REPORT_SCHEMA, _metrics,
)
from src.scripts.train_faltando_cnn_v2_online import _assess
from src.services.aoi_universal_quality_archive import (
    split_universal_events, universal_archive_events,
)

PINNED_97_70_SHA256 = (
    "c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c"
)
SCHEMA = "visionx.aoi_head_to_head_97p70_readonly.v1"


def _universal_report(root: Path, report_path: Path, manifest_sha: str) -> dict:
    folder = root / "reports/aoi_universal_quality/experiments"
    path = _inside(root, report_path, folder)
    if path.name != "training_report.json":
        raise ValueError("Relatório universal deve ser training_report.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or (
        report.get("schema") != UNIVERSAL_REPORT_SCHEMA
        or report.get("candidate_only") is not True
        or report.get("promoted") is not False
        or report.get("production_approval_eligible") is not False
        or report.get("aoi_category_used_as_model_input") is not False
        or report.get("epoch_selection_uses_test") is not False
        or report.get("provenance", {}).get("manifest_sha256") != manifest_sha
    ):
        raise ValueError("Relatório universal incompatível ou não experimental")
    candidate = _inside(root, Path(report["source_candidate_relative_path"]), folder)
    if candidate.parent != path.parent or _sha(candidate) != report.get("candidate_sha256"):
        raise ValueError("Candidato universal não corresponde ao relatório")
    from src.scripts.train_aoi_universal_quality_offline import SCHEMA as UNIVERSAL_SCHEMA
    # Validate metadata without loading/executing the universal checkpoint.
    # Universal predictions remain those saved by its original test run.
    import torch
    chk = torch.load(candidate, map_location="cpu", weights_only=True)
    if not isinstance(chk, dict) or (
        chk.get("schema") != UNIVERSAL_SCHEMA
        or chk.get("candidate_only") is not True
        or chk.get("production_approved") is not False
        or chk.get("automatic_ok_authorized") is not False
        or chk.get("source_manifest_sha256") != manifest_sha
        or chk.get("selected_epoch") != report.get("selected_epoch")
    ):
        raise ValueError("Metadados do checkpoint universal incompatíveis")
    return report


def _operational_at_existing_thresholds(cases: list[dict]) -> dict:
    counts = Counter()
    for case in cases:
        value = case.get("ng_score")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0. <= value <= 1.:
            raise ValueError("Score fora de [0,1]")
        label = case.get("label")
        if label not in ("NG", "OK"):
            raise ValueError("Rótulo não binário")
        classification = "NG" if value >= .90 else ("OK" if value <= .10 else "REVIEW")
        counts[(label, classification)] += 1
    return {
        "thresholds_diagnostic_only": {"OK_max": .10, "NG_min": .90},
        "NG_conclusive": counts[("NG", "NG")],
        "NG_missed_as_OK": counts[("NG", "OK")],
        "NG_review": counts[("NG", "REVIEW")],
        "OK_conclusive": counts[("OK", "OK")],
        "OK_false_NG": counts[("OK", "NG")],
        "OK_review": counts[("OK", "REVIEW")],
        "no_threshold_changed": True,
    }


def benchmark_97p70(
    root: Path, manifest: Path, original_report: Path,
    universal_report: Path,
) -> dict:
    root = Path(root).resolve()
    if (root / "reports/neural_online/live_active.json").exists():
        raise ValueError("Ponteiro online ativo: auditoria manual necessária")
    manifest = Path(manifest)
    manifest = manifest if manifest.is_absolute() else root / manifest
    if (manifest.is_symlink() or not manifest.is_file()
            or not manifest.resolve().is_relative_to(root / "reports/faltando_neural")
            or manifest.name != "manifest.json"):
        raise ValueError("Manifesto fora da preparação AOI")
    manifest_sha = _sha(manifest)
    previous, chk = _verified_candidate(root, Path(original_report))
    if previous.get("candidate_sha256") != PINNED_97_70_SHA256:
        raise ValueError("Candidato não é o primeiro checkpoint de 97,70%")
    old_hold = previous.get("new_holdout", {})
    old_hist = previous.get("new_historical", {})
    if (
        old_hold.get("images") != 87 or old_hold.get("correct_images") != 85
        or old_hold.get("ng_false_ok") != 0
        or old_hist.get("images") != 117
        or old_hist.get("correct_images") != 98
    ):
        raise ValueError("As métricas originais 85/87 e 98/117 foram alteradas")
    size, focus = chk.get("image_size"), chk.get("focus_fraction")
    if (chk.get("schema") != MODEL_SCHEMA_V2
            or type(size) is not int or not 64 <= size <= 512
            or not isinstance(focus, (float, int)) or not .5 <= focus <= .9):
        raise ValueError("Pré-processamento original inválido")

    universal = _universal_report(root, Path(universal_report), manifest_sha)
    rows, provenance = universal_archive_events(manifest)
    if provenance["manifest_sha256"] != manifest_sha:
        raise ValueError("Proveniência da base universal não corresponde ao manifesto")
    split = universal.get("split", {})
    train, validation, test, recalculated = split_universal_events(
        rows, seed=universal["seed"],
        test_fraction=split["test_fraction_requested"],
        validation_fraction=split["validation_fraction_requested"],
    )
    # Confirm ALL fold identities (not just counts) against published run.
    for fold_name, fold in (("train", train), ("validation", validation), ("test", test)):
        expected = split["folds"][fold_name]["source_ids"]
        if [r["id"] for r in fold] != expected:
            raise ValueError("Split publicado mudou: " + fold_name)
    if len(test) != universal["test"]["total"]:
        raise ValueError("Total do teste original foi alterado")
    original_cases = universal.get("test_cases")
    if not isinstance(original_cases, list) or len(original_cases) != len(test):
        raise ValueError("Predições universais originais ausentes")
    for item, case in zip(test, original_cases):
        if (case.get("event_id_or_group_hint") != item["id"]
                or case.get("label") != item["label"]):
            raise ValueError("Predições universais fora de ordem ou com rótulo alterado")
        score = case.get("ng_score")
        if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Score universal não finito")
        if case.get("predicted") != ("NG" if score >= .5 else "OK"):
            raise ValueError("Predição universal inconsistente com score .50")
    original_metrics = _metrics(test, {"cases": original_cases})
    if (original_metrics["total"] != universal["test"]["total"]
            or original_metrics["correct"] != universal["test"]["correct"]
            or original_metrics["NG_false_OK"] != universal["test"]["NG_false_OK"]
            or original_metrics["OK_false_NG"] != universal["test"]["OK_false_NG"]):
        raise ValueError("Relatório universal diverge de suas predições salvas")

    model = FaltandoCNNV2()
    model.load_state_dict(chk["state_dict"], strict=True)
    measured = _assess(model, test, int(size), float(focus))
    old_metrics = _metrics(test, measured)
    old_by_id = {case["event_id_or_group_hint"]: case for case in measured["cases"]}
    if len(old_by_id) != len(test):
        raise ValueError("Predições da CNN 97,70 duplicadas")
    diffs = Counter()
    errors = []
    for item, universal_case in zip(test, original_cases):
        old = old_by_id[item["id"]]
        old_correct = old["correct"]
        new_correct = universal_case["correct"]
        if old_correct and new_correct:
            diffs["both_correct"] += 1
        elif old_correct:
            diffs["only_97p70_correct"] += 1
        elif new_correct:
            diffs["only_universal_correct"] += 1
        else:
            diffs["both_wrong"] += 1
        if not old_correct or not new_correct:
            errors.append({
                "source": item["sources"][0],
                "aoi_category": item["category_hint"],
                "operator_visual_defect_if_reviewed": item.get("visual_defect_type"),
                "expected_quality": item["label"],
                "original_97p70": {
                    "prediction": old["predicted"], "score_ng": old["ng_score"],
                },
                "universal": {
                    "prediction": universal_case["predicted"],
                    "score_ng": universal_case["ng_score"],
                },
            })
    # Same images can appear in BOTH experiments' earlier archive runs.
    prior_test_ids = set(previous.get("holdout_event_ids", ()))
    earlier_overlap = [
        item["id"] for item in test
        if item["id"].split(":", 1)[-1] in
        {s.split(":", 1)[-1] for s in prior_test_ids}
    ]
    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_only": True, "trained": False, "promoted": False,
        "candidate_only": True,
        "automatic_ok_authorized": False,
        "production_approval_eligible": False,
        "thresholds_changed": False, "active_pointer_changed": False,
        "dataset_changed": False, "labels_changed": False,
        "manifest_sha256": manifest_sha,
        "first_97p70_candidate_sha256": previous["candidate_sha256"],
        "universal_candidate_sha256": universal["candidate_sha256"],
        "first_experiment_original_holdout": {
            "correct": old_hold["correct_images"], "total": old_hold["images"],
            "historical_correct": old_hist["correct_images"],
            "historical_total": old_hist["images"],
            "historical_false_ng_count": (
                old_hist["images"] - old_hist["correct_images"]
            ),
            "note": "Original 87 images, a DIFFERENT evaluation set.",
        },
        "same_universal_test_total": len(test),
        "original_universal_report_scores_reused": True,
        "universal_test_not_reevaluated_or_tuned": True,
        "original_97p70_retested_on_universal_test": True,
        "test_split_exactly_matched_original": True,
        "identities_overlap_with_old_87_holdout": len(earlier_overlap),
        "previous_87_holdout_overlap_ids": earlier_overlap,
        "physically_independent_test_confirmed": False,
        "old_data_previously_used_for_experiments": True,
        "universal_test_metrics_saved": original_metrics,
        "original_97p70_on_same_test": old_metrics,
        "diagnostic_0_10_0_90": {
            "original_97p70": _operational_at_existing_thresholds(measured["cases"]),
            "universal": _operational_at_existing_thresholds(original_cases),
        },
        "pairwise_correctness": dict(diffs),
        "mistakes_in_either_model": errors,
        "decision": "NO_AUTO_SELECTION_OR_PRODUCTION_PROMOTION",
        "limitations": [
            "Comparação não cega: acervo parcialmente visto por CNN antiga.",
            "O 97,70% é de 85/87 imagens anteriores, NÃO deste teste de 106.",
            "Modelo antigo falhou em 19 imagens do histórico de 117.",
            "Teste universal não possui NG para todas as categorias.",
            "Classes da pasta são provisórias; 14 casos têm triagem visual.",
            "Nenhuma métrica substitui replay DESLOCADO e validação física independente.",
        ],
    }
    out = root / "reports/aoi_universal_quality/benchmarks" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + "_" + uuid4().hex[:6]
    )
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / "head_to_head_report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "CNN original 97,70% no NOVO conjunto:",
        f"{old_metrics['correct']}/{old_metrics['total']}",
        f"NG→OK={old_metrics['NG_false_OK']}, OK→NG={old_metrics['OK_false_NG']}",
        flush=True,
    )
    print(
        "CNN universal no MESMO conjunto:",
        f"{original_metrics['correct']}/{original_metrics['total']}",
        f"NG→OK={original_metrics['NG_false_OK']}, OK→NG={original_metrics['OK_false_NG']}",
        flush=True,
    )
    print(
        "Histórico original da CNN 97,70%:",
        f"{old_hist['correct_images']}/{old_hist['images']}",
        flush=True,
    )
    print("Apenas comparação offline — SEM TREINO, SEM PROMOÇÃO", flush=True)
    print("Relatório:", report_path, flush=True)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Benchmark OFFLINE da CNN original 97,70%")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--original-report", type=Path, required=True)
    p.add_argument("--universal-report", type=Path, required=True)
    args = p.parse_args(argv)
    benchmark_97p70(
        args.root, args.manifest, args.original_report,
        args.universal_report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
