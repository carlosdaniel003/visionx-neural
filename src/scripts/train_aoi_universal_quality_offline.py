"""Train AOI GENERAL OK/NG CNN candidate, *never* a production checkpoint.

Cross-category images share the same comparator architecture as CNN V2, but
candidate schema and output path are deliberately incompatible with production
loaders. Filename AOI categories are metadata, NOT input features or true
physical defect classes. Archive labels remain provisional.

A three-way, visually fenced split is made BEFORE optimization. Epoch choice
uses validation only. Test is evaluated exactly once after epoch selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, LIGHTS, MODEL_SCHEMA_V2
from src.core.neural.faltando_live import (
    PINNED_CHECKPOINT_SHA256, PINNED_RELATIVE_CHECKPOINT,
)
from src.scripts.deploy_faltando_97p70_trial import _read_trial_pointer
from src.scripts.train_faltando_cnn_v2_online import (
    PairDataset, _assess, _supervised_loss,
)
from src.services.aoi_universal_quality_archive import (
    split_universal_events, universal_archive_events,
)

SCHEMA = "visionx.aoi_general_quality_offline_candidate.v1"
REPORT_SCHEMA = "visionx.aoi_general_quality_experiment_report.v1"


def _trial_readonly_snapshot(root: Path, *, opt_in: bool) -> str | None:
    """Audit a known trial pointer without ever using it for optimization.

    The existing trial validator checks schema, authorized manual-review
    restrictions and SHA-256 of its referenced checkpoint. Unknown online
    pointers stay blocked. The fingerprint is checked again before saving.
    """
    pointer = root / "reports/neural_online/live_active.json"
    if not pointer.exists() and not pointer.is_symlink():
        if opt_in:
            raise ValueError("Nenhum ponteiro de ensaio ativo para auditar")
        return None
    if not opt_in:
        raise ValueError(
            "Ponteiro online ativo: usar "
            "--allow-verified-trial-pointer-readonly somente para o ensaio "
            "97,70% já validado, com ODIN fechado; nunca apagar o ponteiro"
        )
    _read_trial_pointer(root)
    return sha256(pointer.read_bytes()).hexdigest()


def _pinned_initialization(root: Path) -> tuple[Path, dict, str]:
    """Read ONLY the SHA-pinned original, not live_active.json or a candidate."""
    path = root / PINNED_RELATIVE_CHECKPOINT
    origin = path.resolve()
    expected_parent = (root / "reports/faltando_neural/models").resolve()
    if (path.is_symlink() or not origin.is_file()
            or not origin.is_relative_to(expected_parent)):
        raise ValueError("Checkpoint fixado ausente ou fora do diretório de modelos")
    digest = sha256(origin.read_bytes()).hexdigest()
    if digest != PINNED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint fixado alterado: SHA-256 incorreto")
    checkpoint = torch.load(origin, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("schema") != MODEL_SCHEMA_V2
            or checkpoint.get("experimental") is not True
            or checkpoint.get("production_approved") is not False
            or tuple(checkpoint.get("lights", ())) != LIGHTS):
        raise ValueError("Checkpoint fixado incompatível")
    return origin, checkpoint, digest


def _metrics(rows: list[dict], assessment: dict) -> dict:
    if len(assessment.get("cases", [])) != len(rows):
        raise ValueError("Avaliação por evento incompleta")
    cm = Counter()
    for event, result in zip(rows, assessment["cases"]):
        if result.get("event_id_or_group_hint") != event["id"]:
            raise ValueError("Avaliação do evento fora de ordem")
        predicted = result["predicted"]
        label = event["label"]
        if predicted not in ("OK", "NG"):
            raise ValueError("Predição binária inválida")
        cm[(event["category_hint"], label, predicted)] += 1
    cats = sorted({r["category_hint"] for r in rows})
    per_category = {}
    rates = []
    for cat in cats:
        tp = cm[(cat, "NG", "NG")]
        fn = cm[(cat, "NG", "OK")]
        tn = cm[(cat, "OK", "OK")]
        fp = cm[(cat, "OK", "NG")]
        ng, ok = tp + fn, tn + fp
        sensitivity = tp / ng if ng else None
        specificity = tn / ok if ok else None
        if sensitivity is not None and specificity is not None:
            rates.append((sensitivity + specificity) / 2)
        per_category[cat] = {
            "NG": ng, "OK": ok, "NG_correct": tp,
            "NG_false_OK": fn, "OK_correct": tn, "OK_false_NG": fp,
            "ng_recall": sensitivity,
            "ok_specificity": specificity,
            "both_labels_evaluable": bool(ng and ok),
        }
    tp = sum(v["NG_correct"] for v in per_category.values())
    ng = sum(v["NG"] for v in per_category.values())
    tn = sum(v["OK_correct"] for v in per_category.values())
    ok = sum(v["OK"] for v in per_category.values())
    return {
        "total": ng+ok,
        "correct": tp+tn,
        "accuracy": (tp+tn)/(ng+ok) if ng+ok else 0.,
        "NG_total": ng, "OK_total": ok,
        "NG_correct": tp, "NG_false_OK": ng-tp,
        "OK_correct": tn, "OK_false_NG": ok-tn,
        "ng_recall": tp/ng if ng else None,
        "ok_specificity": tn/ok if ok else None,
        "balanced_accuracy": .5 * (tp/ng + tn/ok) if ng and ok else None,
        "macro_category_balanced_accuracy": (
            sum(rates)/len(rates) if rates else None
        ),
        "categories_with_both_labels": len(rates),
        "category_count": len(cats),
        "by_aoi_category": per_category,
        "light_labels": "SINGLE_VIEW_ARCHIVE_PROVISIONAL",
        "threshold": .5,
    }


def _selection_key(metrics: dict) -> tuple:
    """Validation only: weight categories equally, not their AOI frequency."""
    macro = metrics["macro_category_balanced_accuracy"]
    if macro is None:
        raise ValueError("Validação sem ambas as classes por categoria")
    return (
        round(macro, 10),
        round(metrics["balanced_accuracy"], 10),
        -metrics["NG_false_OK"],
        -metrics["OK_false_NG"],
    )


def train_general_quality(
    root: Path, manifest: Path, *, epochs: int = 5, seed: int = 42,
    learning_rate: float = .00005, validation_fraction: float = .20,
    test_fraction: float = .20, image_size: int = 128,
    allow_provisional_archive_labels: bool = False,
    initialize_from_pinned_champion: bool = False,
    allow_verified_trial_pointer_readonly: bool = False,
) -> dict:
    """Offline candidate only; never reports training_ready or approval."""
    if not allow_provisional_archive_labels:
        raise ValueError(
            "Rótulos de pasta são provisórios: use "
            "--allow-provisional-archive-labels apenas para experimento offline"
        )
    if (type(epochs) is not int or not 1 <= epochs <= 20
            or type(seed) is not int
            or isinstance(seed, bool)
            or not 0. < learning_rate <= .0002
            or type(image_size) is not int
            or not 64 <= image_size <= 256):
        raise ValueError("Hiperparâmetros inválidos")
    root = Path(root).resolve()
    trial_pointer_digest = _trial_readonly_snapshot(
        root, opt_in=allow_verified_trial_pointer_readonly,
    )
    manifest = Path(manifest)
    manifest = manifest if manifest.is_absolute() else root/manifest
    manifest = manifest.resolve()
    if (manifest.name != "manifest.json"
            or not manifest.is_relative_to(root/"reports/faltando_neural")
            or not manifest.is_file()):
        raise ValueError("Manifesto não pertence à árvore de preparação local")
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    events, provenance = universal_archive_events(manifest)
    train, validation, test, split = split_universal_events(
        events, seed=seed, test_fraction=test_fraction,
        validation_fraction=validation_fraction,
    )
    # All source records are single-view; no fake multilight event links.
    if any(len(e["views"]) != 1 for e in events):
        raise ValueError("Um evento multilight não verificado entrou no treino")
    model = FaltandoCNNV2()
    source_base_sha = None
    if initialize_from_pinned_champion:
        _, base, source_base_sha = _pinned_initialization(root)
        model.load_state_dict(base["state_dict"], strict=True)

    # Inverse sqrt of class×AOI-category support, bounded against amplification
    # of extremely rare two-NG categories.
    frequencies = Counter((e["category_hint"], e["label"]) for e in train)
    weights = [
        min(5., 1. / (frequencies[(e["category_hint"], e["label"])] ** .5))
        for e in train
    ]
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double), num_samples=len(train),
        replacement=True, generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        PairDataset(train, image_size, .70), batch_size=4,
        sampler=sampler, num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=.01,
    )
    # Validation-only epoch selection includes epoch zero.
    baseline_val = _metrics(
        validation, _assess(model, validation, image_size, .70),
    )
    best_key = _selection_key(baseline_val)
    best_epoch = 0
    best_weights = deepcopy(model.state_dict())
    trajectory = [{
        "epoch": 0, "validation": baseline_val,
        "selected": True, "trained": False,
    }]
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.
        for ref, test_img, focus_ref, focus_test, mask, truth in loader:
            logits, per_light = model(
                ref, test_img, focus_ref, focus_test, mask,
            )
            # Each record has exactly one view. This light's provisional
            # quality label is the archive folder decision, not defect TYPE.
            loss = _supervised_loss(
                logits, per_light, mask, truth, light_weight=.15,
            )
            if not bool(torch.isfinite(loss).item()):
                raise ValueError("Loss não finita")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            total_loss += float(loss.item())
        validation_metrics = _metrics(
            validation, _assess(model, validation, image_size, .70),
        )
        score = _selection_key(validation_metrics)
        improved = score > best_key
        if improved:
            best_key, best_epoch = score, epoch
            best_weights = deepcopy(model.state_dict())
        trajectory.append({
            "epoch": epoch, "mean_loss": round(total_loss / len(loader), 6),
            "validation": validation_metrics, "selected": improved,
            "trained": True,
        })
        print(
            f"AOI UNIVERSAL {epoch}/{epochs} loss={total_loss/len(loader):.6f} "
            f"val={validation_metrics['correct']}/{validation_metrics['total']} "
            f"macro={validation_metrics['macro_category_balanced_accuracy']:.4f}",
            flush=True,
        )

    model.load_state_dict(best_weights, strict=True)
    best_val = _metrics(
        validation, _assess(model, validation, image_size, .70),
    )
    # The test partition is deliberately evaluated exactly ONCE, after
    # choosing the final epoch. Never select on historical holdout results.
    test_result = _assess(model, test, image_size, .70)
    test_metrics = _metrics(test, test_result)
    enough_coverage = all(
        cat["both_labels_evaluable"]
        for cat in test_metrics["by_aoi_category"].values()
    ) and len(test_metrics["by_aoi_category"]) == len(
        provenance["by_category_and_label"]
    )
    diagnostic_gates = {
        "holdout_accuracy_at_least_97_pct":
            test_metrics["accuracy"] >= .97,
        "goal_98_pct_reached": test_metrics["accuracy"] >= .98,
        "test_all_aoi_categories_with_both_labels": enough_coverage,
        "test_no_ng_false_ok": test_metrics["NG_false_OK"] == 0,
        "validation_has_both_classes": best_val["balanced_accuracy"] is not None,
        "independent_physical_events_confirmed": False,
        "all_labels_individually_verified": False,
        "baseline_other_defects_replay_qualified": False,
    }
    # Re-check even after CPU training: fail closed if the active trial changed.
    if _trial_readonly_snapshot(
        root, opt_in=allow_verified_trial_pointer_readonly,
    ) != trial_pointer_digest:
        raise ValueError("Ponteiro de ensaio mudou durante o treinamento")
    if initialize_from_pinned_champion:
        _, _, current_base_sha = _pinned_initialization(root)
        if current_base_sha != source_base_sha:
            raise ValueError("Checkpoint original mudou durante o treinamento")
    stamp = datetime.now(timezone.utc)
    out = root/"reports/aoi_universal_quality/experiments"/(
        stamp.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:6]
    )
    out.mkdir(parents=True, exist_ok=False)
    candidate = out/"aoi_general_quality_offline_candidate.pt"
    # Deliberately NOT MODEL_SCHEMA_V2: productive loader cannot accept it.
    torch.save({
        "schema": SCHEMA,
        "architecture_schema": MODEL_SCHEMA_V2,
        "state_dict": {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()},
        "lights": ("SIDE", "TOP", "MID"),
        "image_size": image_size, "focus_fraction": .70,
        "label_semantics": "PROVISIONAL_ARCHIVE_QUALITY_OK_NG_NOT_PHYSICAL_TYPE",
        "aoi_category_used_as_model_input": False,
        "trained_with_provisional_labels": True,
        "individual_pair_review_complete": False,
        "production_approved": False,
        "automatic_ok_authorized": False,
        "candidate_only": True,
        "source_manifest_sha256": provenance["manifest_sha256"],
        "source_champion_sha256": source_base_sha,
        "active_trial_pointer_sha256_readonly": trial_pointer_digest,
        "selected_epoch": best_epoch,
    }, candidate)
    report = {
        "schema": REPORT_SCHEMA, "created_at_utc": stamp.isoformat(),
        "candidate_only": True,
        "production_approval_eligible": False,
        "automatic_ok_authorized": False, "promoted": False,
        "training_performed": True, "trained_on_provisional_labels": True,
        "aoi_category_used_as_model_input": False,
        "physical_defect_type_predicted": False,
        "physically_independent_test_claim": False,
        "blind_test_claim": False,
        "legacy_historical_replay_qualified": False,
        "new_cross_category_production_regression_qualified": False,
        "old_faltando_holdout_may_overlap": True,
        "source_champion_sha256": source_base_sha,
        "active_trial_pointer_sha256_readonly": trial_pointer_digest,
        "active_checkpoint_used_for_training": False,
        "provenance": provenance, "split": split,
        "seed": seed, "epochs": epochs, "learning_rate": learning_rate,
        "image_size": image_size, "selected_epoch": best_epoch,
        "epoch_selection_uses_test": False,
        "epoch_history": trajectory,
        "validation": best_val, "test": test_metrics,
        "test_cases": test_result["cases"],
        "diagnostic_gates": diagnostic_gates,
        "source_candidate_relative_path": candidate.relative_to(root).as_posix(),
        "candidate_sha256": sha256(candidate.read_bytes()).hexdigest(),
        "limitations": [
            "Nomes das categorias da AOI são pistas falíveis, não defeitos físicos.",
            "NG/OK herdados das pastas ainda não são revisão humana individual.",
            "Trincas por nome e capturas no mesmo minuto NÃO provam evento físico.",
            "Imagens semelhantes são agrupadas conservadoramente para split.",
            "Arquivo já foi usado em experimentos anteriores; teste NÃO é cego.",
            "Novo teste independente por peça/evento e gates por categoria são obrigatórios.",
            "Este checkpoint não é aceito pelo carregador produtivo; nunca promover.",
        ],
    }
    out_report = out/"training_report.json"
    out_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Melhor época por validação (SEM teste): {best_epoch}", flush=True)
    print(
        f"Teste final: {test_metrics['correct']}/{test_metrics['total']} "
        f"({100*test_metrics['accuracy']:.2f}%), "
        f"falsos OK NG={test_metrics['NG_false_OK']}",
        flush=True,
    )
    print("CANDIDATO EXPERIMENTAL — PROIBIDO PROMOVER", flush=True)
    print("Relatório:", out_report, flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Treinador offline de qualidade AOI universal")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=.00005)
    parser.add_argument("--validation", type=float, default=.20)
    parser.add_argument("--test", type=float, default=.20)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--allow-provisional-archive-labels", action="store_true")
    parser.add_argument("--initialize-from-pinned-champion", action="store_true")
    parser.add_argument("--allow-verified-trial-pointer-readonly", action="store_true",
                        help="Audita ensaio 97,70%% conhecido sem modificar ponteiro")
    args = parser.parse_args(argv)
    train_general_quality(
        args.root, args.manifest, epochs=args.epochs, seed=args.seed,
        learning_rate=args.lr, validation_fraction=args.validation,
        test_fraction=args.test, image_size=args.image_size,
        allow_provisional_archive_labels=args.allow_provisional_archive_labels,
        initialize_from_pinned_champion=args.initialize_from_pinned_champion,
        allow_verified_trial_pointer_readonly=args.allow_verified_trial_pointer_readonly,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
