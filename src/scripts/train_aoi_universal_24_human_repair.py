"""OFFLINE second-stage CNN repair for 24 verified quality decisions.

Nine new operator judgments from the prior validation become training labels,
never an independently scored validation. Previously repaired 15 are retained.
Old test is retrospective only, and this script cannot touch production.
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

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.scripts.audit_aoi_universal_operator_confirmed import (
    _build_records, _original_manifest, _read_json,
)
from src.scripts.train_aoi_universal_hard_case_repair import (
    _operator_events, _protected_split, REPAIR_SCHEMA, REPAIR_REPORT,
)
from src.scripts.train_aoi_universal_quality_offline import _metrics
from src.scripts.train_faltando_cnn_v2_online import PairDataset, _assess, _supervised_loss
from src.services.aoi_universal_quality_archive import universal_archive_events
from src.scripts.train_faltando_cnn_v2_verified_archive import _visual_key
from src.services.faltando_neural_qualification import QualificationStore
from src.services.odin_operator_confirmed_labels import verified_operator_labels
from src.services.odin_operator_validation_labels import verified_validation_labels

SCHEMA = "visionx.aoi_universal_24_human_repair_candidate.v1"
REPORT_SCHEMA = "visionx.aoi_universal_24_human_repair_report.v1"


def _load_prior(root: Path, report_name: Path) -> tuple[dict, Path, dict]:
    report_file = report_name if report_name.is_absolute() else root / report_name
    report_file = report_file.resolve()
    parent = (root / "reports/aoi_universal_quality/hard_case_repairs").resolve()
    if (report_file.is_symlink() or report_file.parent.parent != parent
            or report_file.name != "hard_case_repair_report.json"):
        raise ValueError("Relatório anterior fora do experimento de reparo")
    report = _read_json(report_file)
    if (report.get("schema") != REPAIR_REPORT
            or report.get("candidate_only") is not True
            or report.get("automatic_ok_authorized") is not False
            or report.get("production_approved") is not False
            or report.get("selected_epoch") != 6):
        raise ValueError("Reparo anterior incompatível com auditoria confirmada")
    source = report.get("candidate_path")
    if (not isinstance(source, str) or Path(source).is_absolute()
            or ".." in Path(source).parts):
        raise ValueError("Caminho de checkpoint inválido")
    checkpoint_file = root / source
    if (checkpoint_file.is_symlink() or not checkpoint_file.is_file()
            or checkpoint_file.resolve().parent != report_file.parent
            or checkpoint_file.name != "aoi_operator_hard_case_candidate.pt"
            or sha256(checkpoint_file.read_bytes()).hexdigest() !=
               report.get("candidate_sha256")):
        raise ValueError("Checkpoint offline diverge do relatório")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("schema") != REPAIR_SCHEMA
            or checkpoint.get("architecture_schema") != MODEL_SCHEMA_V2
            or checkpoint.get("candidate_only") is not True
            or checkpoint.get("production_approved") is not False
            or checkpoint.get("automatic_ok_authorized") is not False
            or checkpoint.get("human_confirmed_sources") != 15):
        raise ValueError("Pesos anteriores não são o reparo isolado aprovado")
    return report, checkpoint_file, checkpoint


def _confirmed_training(all_events: list[dict], old: dict, new: dict,
                        *, root: Path, manifest: Path, manifest_doc: dict,
                        output: Path) -> tuple[list[dict], dict]:
    source_to_event = {
        row["sources"][0]: row for row in all_events
    }
    if len(source_to_event) != len(all_events):
        raise ValueError("Fonte duplicada nos eventos universais")
    previous, previous_pairs = _build_records(
        root, manifest, manifest_doc, old, output,
    )
    new_records = []
    for source, review in sorted(new.items()):
        event = source_to_event.get(source)
        if event is None or event["source_sha256"] != review["source_sha256"]:
            raise ValueError("Caso humano da validação ausente do manifesto: " + source)
        if (event["label"] != review["quality_label"]
                or event["label_provenance"] != "HUMAN_CONFIRMED_SHA256"):
            raise ValueError("Qualidade humana não prevaleceu sobre a pasta")
        view = event["views"]
        new_records.append({
            "id": source, "label": review["quality_label"],
            "views": view, "source_sha256": review["source_sha256"],
        })
    combined = _operator_events(previous + new_records)
    if (len(combined) != 24
            or Counter(x["label"] for x in combined) !=
               Counter({"NG": 15, "OK": 9})
            or len(set(s for event in combined for s in event["sources"])) != 24):
        raise ValueError("Total de imagens confirmadas difere de 15 NG e 9 OK")
    return combined, {
        "old_reviewed": len(old), "new_reviewed": len(new),
        "old_pairs": previous_pairs, "new_prepared_pairs": len(new_records),
    }


def _verified_metrics(outcome: dict, truth: list[dict]) -> dict:
    cases = outcome.get("cases", [])
    if len(cases) != len(truth):
        raise ValueError("Replay supervisionado incompleto")
    cm = Counter()
    misses = []
    for record, row in zip(truth, cases):
        if (record["id"] != row.get("event_id_or_group_hint")
                or record["label"] != row.get("label")):
            raise ValueError("Classificação não corresponde às imagens verificadas")
        cm[record["label"], row["predicted"]] += 1
        if record["label"] != row["predicted"]:
            misses.append({
                "source": record["sources"][0],
                "true": record["label"], "predicted": row["predicted"],
                "ng_score": row["ng_score"],
            })
    if (cm["NG", "NG"] + cm["NG", "OK"] != 15
            or cm["OK", "NG"] + cm["OK", "OK"] != 9):
        raise ValueError("Distribuição física supervisionada alterada")
    return {
        "NG_correct": cm["NG", "NG"],
        "NG_false_OK": cm["NG", "OK"],
        "OK_correct": cm["OK", "OK"],
        "OK_false_NG": cm["OK", "NG"],
        "total_correct": cm["NG", "NG"] + cm["OK", "OK"],
        "errors": misses,
    }


def _acceptable(hard: dict, validation: dict, baseline: dict) -> tuple | None:
    """No increase in provisional NG escapes OR unjustified OK rejections."""
    if (validation["NG_false_OK"] > baseline["NG_false_OK"]
            or validation["OK_false_NG"] > baseline["OK_false_NG"]):
        return None
    return (
        -hard["NG_false_OK"],
        -hard["OK_false_NG"],
        -validation["NG_false_OK"],
        -validation["OK_false_NG"],
        round(validation["balanced_accuracy"] or 0, 10),
    )


def _avoid_visual_replay_in_validation(
    validation: list[dict], human_training: list[dict],
) -> tuple[list[dict], int]:
    """Remove validation minute groups visually matching the supervised 24.

    These records would no longer be an independent validation after hard
    cases enter training. Same-light reference AND test similarity only.
    """
    def signature(record):
        return [
            (light, *_visual_key(pair["reference"]), *_visual_key(pair["test"]))
            for light, pair in record["views"].items()
        ]

    reviewed = [v for row in human_training for v in signature(row)]
    unsafe_groups = set()
    for row in validation:
        for light, rh, rthumb, th, tthumb in signature(row):
            if any(
                light == other_light
                and sum((a ^ b).bit_count() for a, b in zip(rh, other_rh)) <= 4
                and sum((a ^ b).bit_count() for a, b in zip(th, other_th)) <= 4
                and float(np.mean(np.abs(rthumb.astype(np.int16) - other_rthumb.astype(np.int16)))) <= 6
                and float(np.mean(np.abs(tthumb.astype(np.int16) - other_tthumb.astype(np.int16)))) <= 6
                for other_light, other_rh, other_rthumb, other_th, other_tthumb in reviewed
            ):
                unsafe_groups.add(row["split_group"])
                break
    retained = [row for row in validation if row["split_group"] not in unsafe_groups]
    if not retained or {x["label"] for x in retained} != {"OK", "NG"}:
        raise ValueError("Após cerca visual, validação independente perdeu uma classe")
    return retained, len(validation) - len(retained)


def train_24_reviewed(
    root: Path, previous_report: Path, manifest: Path,
    *, epochs: int = 6, lr: float = .00002, seed: int = 42,
) -> dict:
    if (type(epochs) is not int or not 1 <= epochs <= 15
            or not 0 < lr <= .0002 or type(seed) is not int):
        raise ValueError("Hiperparâmetros inválidos")
    root = Path(root).resolve()
    original, previous_file, checkpoint = _load_prior(root, Path(previous_report))
    manifest_file, manifest_doc = _original_manifest(
        root, Path(manifest), checkpoint["source_manifest_sha256"],
    )
    if original["candidate_sha256"] != sha256(previous_file.read_bytes()).hexdigest():
        raise ValueError("Candidato anterior alterado")
    store = QualificationStore(manifest_file)
    reviewed15 = verified_operator_labels(root)
    reviewed9 = verified_validation_labels(store)
    if set(reviewed15) & set(reviewed9):
        raise ValueError("Imagens humanas repetidas entre revisões")
    confirmed = {**reviewed15, **reviewed9}
    pointer = root / "reports/neural_online/live_active.json"
    if pointer.is_symlink():
        raise ValueError("Ponteiro ativo symlink inesperado")
    before = sha256(pointer.read_bytes()).hexdigest() if pointer.exists() else None
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    events, provenance = universal_archive_events(manifest_file)
    base, validation, past_test, partition = _protected_split(
        events, confirmed, seed=seed,
    )
    reviewed_set = set(confirmed)
    if (any(any(src in reviewed_set for src in e["sources"])
            for e in validation + past_test)
            or not base):
        raise ValueError("Revisões humanas vazaram para validação/teste")
    size, focus = checkpoint["image_size"], checkpoint["focus_fraction"]
    if (type(size) is not int or not 64 <= size <= 256
            or not isinstance(focus, (int, float)) or not .1 < focus < 1.):
        raise ValueError("Pré-processamento anterior inválido")

    output = root / "reports/aoi_universal_quality/validation_repairs" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + "_" + uuid4().hex[:6]
    )
    output.mkdir(parents=True, exist_ok=False)
    hard, human_source = _confirmed_training(
        events, reviewed15, reviewed9,
        root=root, manifest=manifest_file, manifest_doc=manifest_doc,
        output=output,
    )
    validation, removed_visual_neighbors = _avoid_visual_replay_in_validation(
        validation, hard,
    )
    partition["validation_visual_neighbors_excluded"] = removed_visual_neighbors
    partition["validation_records_after_visual_fence"] = len(validation)
    model = FaltandoCNNV2()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    original_validation = _metrics(
        validation, _assess(model, validation, size, focus),
    )
    original_hard = _verified_metrics(
        _assess(model, hard, size, focus), hard,
    )
    selected_epoch, selected_weights = 0, deepcopy(model.state_dict())
    best_key = _acceptable(original_hard, original_validation, original_validation)
    progress = [{
        "epoch": 0, "confirmed_train": original_hard,
        "unseen_provisional_validation": original_validation,
        "selected": True, "trained": False,
    }]
    loader = DataLoader(
        PairDataset(base, size, focus), batch_size=4,
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
        r, t, rf, tf, mask, truth = batch
        logits, light_logits = model(r, t, rf, tf, mask)
        loss = _supervised_loss(
            logits, light_logits, mask, truth, light_weight=.20,
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
        losses = []
        hard_iter = iter(hard_loader)
        hard_updates = 0
        for ix, batch in enumerate(loader, 1):
            losses.append(update(batch))
            if ix % 5 == 0:
                try:
                    supervised = next(hard_iter)
                except StopIteration:
                    hard_iter = iter(hard_loader)
                    supervised = next(hard_iter)
                losses.append(update(supervised))
                hard_updates += 1
        for supervised in hard_loader:
            losses.append(update(supervised))
            hard_updates += 1
        val = _metrics(validation, _assess(model, validation, size, focus))
        training = _verified_metrics(_assess(model, hard, size, focus), hard)
        key = _acceptable(training, val, original_validation)
        accepted = key is not None and key > best_key
        if accepted:
            best_key = key
            selected_epoch = epoch
            selected_weights = deepcopy(model.state_dict())
        progress.append({
            "epoch": epoch, "mean_loss": round(sum(losses) / len(losses), 8),
            "hard_batches": hard_updates, "confirmed_train": training,
            "unseen_provisional_validation": val,
            "selected": accepted, "regression_veto": key is None,
        })
        print(
            f"ODIN OFFLINE 24 {epoch}/{epochs} "
            f"NG_confirmados_ignorados={training['NG_false_OK']}/15 "
            f"OK_confirmados_rejeitados={training['OK_false_NG']}/9 "
            f"val_NG_ignorados={val['NG_false_OK']} "
            f"val_OK_rejeitados={val['OK_false_NG']} "
            f"melhor={selected_epoch}", flush=True,
        )
    model.load_state_dict(selected_weights, strict=True)
    final_hard = _verified_metrics(_assess(model, hard, size, focus), hard)
    final_val = _metrics(validation, _assess(model, validation, size, focus))
    # Diagnostic only: old test was already exposed and has provisional labels.
    old_test_metric = _metrics(past_test, _assess(model, past_test, size, focus))
    after = sha256(pointer.read_bytes()).hexdigest() if pointer.exists() else None
    if (before != after
            or sha256(previous_file.read_bytes()).hexdigest() != original["candidate_sha256"]
            or sha256(manifest_file.read_bytes()).hexdigest() != provenance["manifest_sha256"]):
        raise ValueError("Origem/pesos/ponteiro alterados durante a sessão")

    candidate = output / "aoi_24_operator_confirmed_candidate.pt"
    torch.save({
        "schema": SCHEMA, "architecture_schema": MODEL_SCHEMA_V2,
        "state_dict": {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()},
        "source_candidate_sha256": original["candidate_sha256"],
        "source_manifest_sha256": provenance["manifest_sha256"],
        "image_size": size, "focus_fraction": focus,
        "operator_supervised_captures": 24,
        "candidate_only": True, "production_approved": False,
        "automatic_ok_authorized": False,
        "selected_epoch": selected_epoch,
    }, candidate)
    report = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_only": True, "production_approved": False,
        "automatic_ok_authorized": False,
        "xp_commands_sent": 0, "live_active_modified": False,
        "archive_modified": False,
        "previous_candidate_sha256": original["candidate_sha256"],
        "candidate_path": candidate.relative_to(root).as_posix(),
        "candidate_sha256": sha256(candidate.read_bytes()).hexdigest(),
        "source_manifest_sha256": provenance["manifest_sha256"],
        "source_revision": human_source,
        "data_counts": {"operator_confirmed_NG": 15,
                        "operator_confirmed_OK": 9},
        "split": partition,
        "epochs": epochs, "selected_epoch": selected_epoch,
        "baseline_confirmed_training": original_hard,
        "selected_confirmed_training": final_hard,
        "baseline_remaining_validation": original_validation,
        "selected_remaining_validation": final_val,
        "historical_test_not_blind": old_test_metric,
        "history": progress,
        "eligible_for_production": False,
        "independent_test_performed": False,
        "limitations": [
            "24 casos humanos foram incluídos no treino: acertos são memorizações possíveis.",
            "Validação remanescente mantém etiquetas provisórias da AOI.",
            "O teste histórico já foi visto; não representa desempenho em peças novas.",
            "Sem evento físico confirmado e sem garantia de todas as categorias/iluminações.",
            "Não promover nem usar no XP sem outra aprovação explícita.",
        ],
    }
    destination = output / "24_operator_repair_report.json"
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print("CANDIDATO OFFLINE — NÃO PROMOVER")
    print("Relatório:", destination, flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ODIN: segunda correção offline dos 24 julgamentos")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--previous-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=.00002)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = train_24_reviewed(
        args.root, args.previous_report, args.manifest,
        epochs=args.epochs, lr=args.lr, seed=args.seed,
    )
    print(json.dumps({
        "selected_epoch": result["selected_epoch"],
        "operator_correct": result["selected_confirmed_training"]["total_correct"],
        "validation_NG_false_OK": result["selected_remaining_validation"]["NG_false_OK"],
        "validation_OK_false_NG": result["selected_remaining_validation"]["OK_false_NG"],
        "candidate_path": result["candidate_path"],
        "production_approved": False,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
