"""Supervised offline hard-case repair for the failed general-quality CNN.

Development ONLY. Replays SHA-confirmed 15 cases (6 inverted, one mislabeled
OK, additional NG/OK controls) alongside old train rows. Reassigns validation
minute-fences touched by reviewed cases to train, does not touch original test.
Known cases are training evidence, never an independent holdout.
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
from torch.utils.data import DataLoader

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2
from src.scripts.audit_aoi_universal_operator_confirmed import _build_records, _candidate, _original_manifest
from src.scripts.train_aoi_universal_quality_offline import _metrics, _selection_key
from src.scripts.train_faltando_cnn_v2_online import PairDataset, _assess, _supervised_loss
from src.services.aoi_universal_quality_archive import (
    _timestamp_bucket, split_universal_events, universal_archive_events,
)
from src.services.odin_operator_confirmed_labels import verified_operator_labels

REPAIR_REPORT = "visionx.aoi_universal_operator_hard_case_repair_report.v1"
REPAIR_SCHEMA = "visionx.aoi_universal_operator_hard_case_candidate.v1"


def _hard_metrics(assessment: dict) -> dict:
    cases = assessment.get("cases", [])
    if len(cases) != 15:
        raise ValueError("Replay humano deve conter exatamente 15 capturas")
    outcomes = Counter((x["label"], x["predicted"]) for x in cases)
    if (outcomes[("NG", "NG")] + outcomes[("NG", "OK")] != 12
            or outcomes[("OK", "NG")] + outcomes[("OK", "OK")] != 3):
        raise ValueError("Distribuição humana alterada")
    return {
        "total": 15,
        "NG_confirmed": 12,
        "NG_correct": outcomes["NG", "NG"],
        "NG_false_OK": outcomes["NG", "OK"],
        "OK_confirmed": 3,
        "OK_correct": outcomes["OK", "OK"],
        "OK_false_NG": outcomes["OK", "NG"],
        "errors": [
            {"source": row["event_id_or_group_hint"], "label": row["label"],
             "predicted": row["predicted"], "ng_score": row["ng_score"]}
            for row in cases if not row["correct"]
        ],
    }


def _protected_split(events: list[dict], confirmed: dict[str, dict],
                     *, seed: int) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Move *entire same-minute fences* from validation to training.

    Prior test remains untouched; any overlap with operator-reviewed minutes
    aborts rather than falsely reporting independent evaluation.
    """
    train, val, test, split = split_universal_events(events, seed=seed)
    reviewed = set(confirmed)
    fences = {"AOI_MINUTE:" + _timestamp_bucket(source) for source in reviewed}
    touched_test = [e["id"] for e in test
                    if e["split_group"] in fences
                    or any(s in reviewed for s in e["sources"])]
    if touched_test:
        raise ValueError("Revisão humana contamina teste antigo; novo split requerido")
    moved = [e for e in val if e["split_group"] in fences
             or any(s in reviewed for s in e["sources"])]
    remaining = [e for e in val if e not in moved]
    selected = train + moved
    # Exactly one supervised copy per reviewed PNG. The known 15 are loaded
    # separately, with SHA and human labels overriding any provisional folder.
    cleaned = [e for e in selected if not any(s in reviewed for s in e["sources"])]
    if not remaining or {e["label"] for e in remaining} != {"NG", "OK"}:
        raise ValueError("Validação restante sem duas classes")
    if len({e["id"] for e in cleaned + remaining + test}) != (
            len(cleaned) + len(remaining) + len(test)):
        raise ValueError("Fonte duplicada entre folds")
    return cleaned, remaining, test, {
        "original_split": split,
        "validation_rows_reassigned_to_train": len(moved),
        "confirmed_old_rows_removed_for_replacement":
            len(selected) - len(cleaned),
        "validation_records_after_fence": len(remaining),
        "old_test_untouched_records": len(test),
        "new_independent_holdout": False,
    }


def _operator_events(raw: list[dict]) -> list[dict]:
    items = []
    for event in raw:
        source = event["id"]
        category = None
        for key in ("FALTANDO", "INVERTIDO", "EMBORCADO", "DESLOCADO", "MUITO_ADESIVO"):
            if "_" + key in Path(source).stem:
                category = key.replace("_", " ")
                break
        if category is None:
            raise ValueError("Metadado AOI desconhecido (não é classe física)")
        items.append({
            **event, "id": "human_verified:" + source,
            "category_hint": category,
            "sources": [source],
            "label_provenance": "HUMAN_CONFIRMED_SHA256",
            "split_group": "AOI_MINUTE:" + _timestamp_bucket(source),
            "event_link_verified": False,
        })
    return items


def _safe_key(verified: dict, val: dict, baseline_val: dict) -> tuple | None:
    """Prefer detecting actual NG, but veto worsened unrelated validation."""
    if (val["NG_false_OK"] > baseline_val["NG_false_OK"]
            or val["OK_false_NG"] > baseline_val["OK_false_NG"] + 2):
        return None
    return (
        -verified["NG_false_OK"],
        -verified["OK_false_NG"],
        round(val["macro_category_balanced_accuracy"] or 0., 9),
        round(val["balanced_accuracy"] or 0., 9),
    )


def repair_universal_candidate(
    root: Path, training_report: Path, manifest: Path, *, epochs: int = 8,
    lr: float = .00004, seed: int = 42, repeat_every: int = 3,
) -> dict:
    if (type(epochs) is not int or not 1 <= epochs <= 15
            or not 0. < lr <= .0002
            or type(seed) is not int
            or type(repeat_every) is not int or not 1 <= repeat_every <= 10):
        raise ValueError("Hiperparâmetros inválidos")
    root = Path(root).resolve()
    human = verified_operator_labels(root)
    prior, source_path, checkpoint = _candidate(root, Path(training_report))
    manifest_path, manifest_doc = _original_manifest(
        root, Path(manifest), prior["provenance"]["manifest_sha256"],
    )
    pointer = root / "reports/neural_online/live_active.json"
    if pointer.is_symlink():
        raise ValueError("Ponteiro ativo não pode ser symlink")
    pointer_digest = sha256(pointer.read_bytes()).hexdigest() if pointer.exists() else None
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    full, provenance = universal_archive_events(manifest_path)
    base_train, validation, old_test, split = _protected_split(full, human, seed=seed)
    size, focus = checkpoint.get("image_size"), checkpoint.get("focus_fraction")
    if (type(size) is not int or not 64 <= size <= 256
            or type(focus) not in (float, int) or not .1 < focus < 1.):
        raise ValueError("Pré-processamento original incompatível")
    stamp = datetime.now(timezone.utc)
    output = root / "reports/aoi_universal_quality/hard_case_repairs" / (
        stamp.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:6]
    )
    output.mkdir(parents=True, exist_ok=False)
    evidence, pair_counts = _build_records(
        root, manifest_path, manifest_doc, human, output,
    )
    hard = _operator_events(evidence)
    if len(hard) != 15 or not base_train:
        raise ValueError("Base corrigida vazia ou sem 15 confirmações")
    model = FaltandoCNNV2()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    starting_validation = _metrics(validation, _assess(model, validation, size, focus))
    starting_hard = _hard_metrics(_assess(model, hard, size, focus))
    best_metric = _safe_key(starting_hard, starting_validation, starting_validation)
    best_epoch = 0
    best_weights = deepcopy(model.state_dict())
    checkpoints = [{
        "epoch": 0, "human_training": starting_hard,
        "unrelated_validation": starting_validation, "accepted": True,
    }]
    loader = DataLoader(
        PairDataset(base_train, size, focus), batch_size=4,
        shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    hard_loader = DataLoader(
        PairDataset(hard, size, focus), batch_size=4,
        shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.02)

    def update(batch):
        ref, test_img, focus_ref, focus_test, mask, truth = batch
        logits, per_light = model(ref, test_img, focus_ref, focus_test, mask)
        loss = _supervised_loss(
            logits, per_light, mask, truth, light_weight=.25,
        )
        if not bool(torch.isfinite(loss).item()):
            raise ValueError("Loss não finita")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
        optimizer.step()
        return float(loss.item())

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.
        hard_batches = iter(hard_loader)
        hard_seen = 0
        for batch_ix, batch in enumerate(loader):
            total_loss += update(batch)
            if (batch_ix + 1) % repeat_every == 0:
                try:
                    hb = next(hard_batches)
                except StopIteration:
                    hard_batches = iter(hard_loader)
                    hb = next(hard_batches)
                total_loss += update(hb)
                hard_seen += 1
        # Ensure at least one *complete* explicit pass over all 15 labels,
        # even with very small base training pools.
        for hb in hard_loader:
            total_loss += update(hb)
            hard_seen += 1
        vm = _metrics(validation, _assess(model, validation, size, focus))
        hm = _hard_metrics(_assess(model, hard, size, focus))
        key = _safe_key(hm, vm, starting_validation)
        picked = key is not None and key > best_metric
        if picked:
            best_metric, best_epoch = key, epoch
            best_weights = deepcopy(model.state_dict())
        checkpoints.append({
            "epoch": epoch,
            "mean_optimization_loss": round(total_loss / (len(loader) + hard_seen), 6),
            "supervised_hard_batches": hard_seen,
            "human_training": hm,
            "unrelated_validation": vm,
            "accepted": picked,
            "veto": key is None,
        })
        print(
            f"REPARO OFFLINE {epoch}/{epochs} "
            f"confirmados_NG_ignorados={hm['NG_false_OK']}/12 "
            f"OK_indevidos={hm['OK_false_NG']}/3 "
            f"val_NG_ignorados={vm['NG_false_OK']} "
            f"selecionado={best_epoch}", flush=True,
        )

    model.load_state_dict(best_weights, strict=True)
    # Old test was previously observed by developers. Regression only.
    baseline_old_test = _metrics(old_test, _assess(model, old_test, size, focus))
    chosen_hard = _hard_metrics(_assess(model, hard, size, focus))
    chosen_validation = _metrics(validation, _assess(model, validation, size, focus))
    if (sha256(source_path.read_bytes()).hexdigest() != prior["candidate_sha256"]
            or (sha256(pointer.read_bytes()).hexdigest() if pointer.exists() else None) != pointer_digest):
        raise ValueError("Checkpoint/ponteiro mudou durante treino offline")
    candidate_path = output / "aoi_operator_hard_case_candidate.pt"
    torch.save({
        "schema": REPAIR_SCHEMA,
        "architecture_schema": checkpoint["architecture_schema"],
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "image_size": size, "focus_fraction": focus,
        "source_candidate_sha256": prior["candidate_sha256"],
        "source_manifest_sha256": provenance["manifest_sha256"],
        "human_confirmed_sources": 15,
        "trained_with_provisional_labels": True,
        "candidate_only": True, "production_approved": False,
        "automatic_ok_authorized": False,
        "selected_epoch": best_epoch,
    }, candidate_path)
    report = {
        "schema": REPAIR_REPORT,
        "created_at_utc": stamp.isoformat(),
        "candidate_only": True,
        "production_approved": False,
        "automatic_ok_authorized": False,
        "xp_commands_sent": 0, "active_pointer_modified": False,
        "original_archive_modified": False,
        "source_candidate_sha256": prior["candidate_sha256"],
        "candidate_path": candidate_path.relative_to(root).as_posix(),
        "candidate_sha256": sha256(candidate_path.read_bytes()).hexdigest(),
        "operator_labels": {"NG": 12, "OK": 3},
        "operator_pair_provenance": pair_counts,
        "split": split,
        "epochs": epochs, "selected_epoch": best_epoch,
        "old_checkpoint_on_confirmed_examples": starting_hard,
        "selected_checkpoint_on_confirmed_examples": chosen_hard,
        "validation_start": starting_validation,
        "validation_selected": chosen_validation,
        "previously_observed_test_diagnostic": baseline_old_test,
        "epoch_history": checkpoints,
        "eligible_for_production": False,
        "blind_independent_test": False,
        "limitations": [
            "Os 15 exemplos humanos participaram do treino de reparo; acerto neles prova apenas ajuste.",
            "Capturas por minuto não representam eventos físicos comprovadamente distintos.",
            "O teste anterior já foi examinado e é somente diagnóstico de regressão.",
            "Rótulos gerais do acervo continuam provisórios; precisa de novo holdout físico independente.",
        ],
    }
    path = output / "hard_case_repair_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"REPARO EXPERIMENTAL; melhor época {best_epoch}; diagnóstico: {path}", flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CNN: reparo offline supervisionado dos 15 casos")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=.00004)
    args = parser.parse_args(argv)
    print(json.dumps({
        "report": repair_universal_candidate(
            args.root, args.training_report, args.manifest,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
        )["candidate_path"],
        "production_approved": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
