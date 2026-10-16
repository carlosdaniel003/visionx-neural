"""Corrective OFFLINE fine-tuning of FALTANDO V2 from a blend candidate.

The original champion is untouched. Supervise the hardest NG by light
(SIDE/TOP/MID) and protect difficult historical OK anchors using paired
balanced gradient updates. Candidate selection uses historical, online, and
archive TRAIN data only. Held-out pairs are evaluated ONCE after selection.

Bulk archive folder labels remain provisional. No production promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, LIGHTS, MODEL_SCHEMA_V2
from src.scripts.faltando_cnn_v2_blend_candidates import (
    _inside, _sha, _verified_candidate,
)
from src.scripts.train_faltando_cnn_v2_online import (
    PairDataset, _all_online, _assess, _base_model, _historical, _supervised_loss,
)
from src.scripts.train_faltando_cnn_v2_verified_archive import (
    _operational_light_summary, _selection_score, candidate_acceptance,
    split_verified_events,
)
from src.services.faltando_neural_archive_experiment import (
    archive_labeled_experimental_events,
)

SCHEMA = "visionx.faltando_v2_guarded_light_repair_experiment.v1"


def _load_start(root: Path, report_path: Path, manifest_sha: str,
                train: list[dict], hold: list[dict]) -> tuple[dict, dict]:
    """Pin blend candidate to audited two-source archive split and champion."""
    folder = root / "reports/faltando_neural/archive_blends"
    path = _inside(root, report_path, folder)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "visionx.faltando_v2_weight_blend_experiment.v1"
        or report.get("candidate_only") is not True
        or report.get("production_approval_eligible") is not False
        or report.get("promoted") is not False
        or report.get("holdout_used_for_selection") is not False
        or report.get("source_manifest_sha256") != manifest_sha
        or report.get("holdout_previously_examined_in_prior_experiments") is not True
    ):
        raise ValueError("Relatório blend experimental inválido")
    candidate = _inside(root, Path(report["candidate_relative_path"]), folder)
    if candidate.parent != path.parent or _sha(candidate) != report.get("candidate_sha256"):
        raise ValueError("Candidato blend adulterado ou fora da execução")
    ckpt = torch.load(candidate, map_location="cpu", weights_only=True)
    if (
        not isinstance(ckpt, dict)
        or ckpt.get("schema") != MODEL_SCHEMA_V2
        or ckpt.get("production_approved") is not False
        or ckpt.get("individual_pair_review_complete") is not False
        or ckpt.get("archive_labels_experimental") is not True
        or ckpt.get("weight_blend_offline") is not True
        or ckpt.get("source_champion_sha256") != report.get("base_checkpoint_sha256")
        or ckpt.get("provisional_archive_manifest_sha256") != manifest_sha
    ):
        raise ValueError("Checkpoint blend possui origem/flags inválidos")

    expected_train = [e["id"] for e in train]
    expected_hold = [e["id"] for e in hold]
    sources = report.get("source_candidate_sha256")
    if not isinstance(sources, dict) or set(sources) != {"conservative", "aggressive"}:
        raise ValueError("Fontes do blend não identificadas")
    archive = root / "reports/faltando_neural/archive_experiments"
    for role, expected_sha in sources.items():
        matches = []
        for source in archive.glob("*/training_report.json"):
            try:
                obj = json.loads(source.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if obj.get("candidate_sha256") == expected_sha:
                matches.append(source)
        if len(matches) != 1:
            raise ValueError("Fonte experimental não localizada de forma única: " + role)
        prior, prior_ckpt = _verified_candidate(root, matches[0])
        if (
            prior.get("base_checkpoint_sha256") != report["base_checkpoint_sha256"]
            or prior["provenance"]["manifest_sha256"] != manifest_sha
            or prior.get("train_event_ids") != expected_train
            or prior.get("holdout_event_ids") != expected_hold
            or prior_ckpt.get("image_size") != ckpt.get("image_size")
            or prior_ckpt.get("focus_fraction") != ckpt.get("focus_fraction")
            or prior_ckpt.get("lights") != ckpt.get("lights")
        ):
            raise ValueError("Fonte blend divergente do split original: " + role)
    return report, ckpt


def _select_hard_pairs(
    net: FaltandoCNNV2, historical: list[dict], online: list[dict],
    archive_train: list[dict], size: int, focus: float, *,
    per_light: int = 12,
) -> tuple[list[dict], dict]:
    """Pick hard NG per illumination and hardest OK historical anchors.

    This function has no access to the holdout. All selected images are
    contained in training, online human events or original fixed history.
    """
    if not 2 <= per_light <= 32:
        raise ValueError("Exposição por iluminação fora dos limites")
    development = online + archive_train
    dev_scores = _assess(net, development, size, focus)
    hist_scores = _assess(net, historical, size, focus)
    if len(dev_scores["cases"]) != len(development) or len(hist_scores["cases"]) != len(historical):
        raise ValueError("Avaliação de desenvolvimento incompleta")
    ng_items = list(zip(development, dev_scores["cases"]))
    chosen_ng, per_light_counts = [], {}
    for light in LIGHTS:
        eligible = [
            (event, case) for event, case in ng_items
            if event["label"] == "NG" and light in case["per_light"]
        ]
        if not eligible:
            raise ValueError("Sem exemplos NG para iluminação " + light)
        eligible.sort(key=lambda item: (
            float(item[1]["per_light"][light]["score"]), item[0]["id"]
        ))
        picked = [
            eligible[i % len(eligible)][0] for i in range(per_light)
        ]
        chosen_ng.extend(picked)
        per_light_counts[light] = len(picked)

    legacy_ok = [
        (event, case) for event, case in zip(historical, hist_scores["cases"])
        if event["label"] == "OK"
    ]
    ok_pool = [
        (event, case) for event, case in
        (list(zip(development, dev_scores["cases"])) + legacy_ok)
        if event["label"] == "OK"
    ]
    if not legacy_ok or not ok_pool:
        raise ValueError("Sem OK históricos para impedir esquecimento")

    def risk(entry):
        return max(float(view["score"]) for view in entry[1]["per_light"].values())

    legacy_ok.sort(key=lambda e: (-risk(e), e[0]["id"]))
    ok_pool.sort(key=lambda e: (-risk(e), e[0]["id"]))
    count = len(chosen_ng)
    # 1/3 of OK steps explicitly repeat hardest original historical examples.
    protected_count = max(1, count // 3)
    hardest_legacy = legacy_ok[:max(1, min(4, len(legacy_ok)))]
    chosen_ok = [
        hardest_legacy[i % len(hardest_legacy)][0]
        for i in range(protected_count)
    ]
    chosen_ok.extend(ok_pool[i % len(ok_pool)][0]
                     for i in range(count - protected_count))
    if len(chosen_ok) != len(chosen_ng):
        raise ValueError("Replay corretivo NG/OK não balanceado")
    paired = [
        event for ng, ok in zip(chosen_ng, chosen_ok) for event in (ng, ok)
    ]
    return paired, {
        "NG": count, "OK": count,
        "hard_ng_by_illumination": per_light_counts,
        "protected_original_ok_steps": protected_count,
    }


def repair_offline(
    root: Path, manifest: Path, start_report: Path, *,
    epochs: int = 4, lr: float = .00001, per_light: int = 12,
) -> dict:
    """Short, low-LR head-only supervised correction; no active model changes."""
    root = Path(root).resolve()
    if not (type(epochs) is int and 1 <= epochs <= 8
            and 0 < lr <= .00002 and 2 <= per_light <= 32):
        raise ValueError("Parâmetros de reparo fora dos limites")
    if (root / "reports/neural_online/live_active.json").exists():
        raise ValueError("Checkpoint online ativo: auditar antes de corrigir")
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    manifest = Path(manifest)
    manifest = manifest if manifest.is_absolute() else root / manifest
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("Manifesto ausente ou symlink")
    events, provenance = archive_labeled_experimental_events(manifest)
    train, hold, split = split_verified_events(events, seed=42, holdout=.25)
    base_path, base = _base_model(root)
    source, ckpt = _load_start(
        root, Path(start_report), provenance["manifest_sha256"], train, hold,
    )
    if _sha(base_path) != source["base_checkpoint_sha256"]:
        raise ValueError("Champion ativo não corresponde ao modelo inicial")
    for key in ("schema", "image_size", "focus_fraction", "lights"):
        if base.get(key) != ckpt.get(key):
            raise ValueError("Checkpoint incompatível com champion: " + key)
    size, focus = int(base["image_size"]), float(base["focus_fraction"])
    if not (64 <= size <= 512 and .5 <= focus <= .90):
        raise ValueError("Pré-processamento inválido")
    historical, online = _historical(root, base), _all_online(root)
    train_ids, hold_ids = {e["id"] for e in train}, {e["id"] for e in hold}
    if train_ids & hold_ids or hold_ids & {e["id"] for e in online + historical}:
        raise ValueError("Sobreposição de IDs entre treino e holdout")
    if split != source["split"]:
        raise ValueError("Mudança de partição desde o blend")

    champion = FaltandoCNNV2()
    champion.load_state_dict(base["state_dict"], strict=True)
    old_historical = _assess(champion, historical, size, focus)
    old_online = _assess(champion, online, size, focus)
    if not old_historical["all_correct"]:
        raise ValueError("Champion já erra o histórico")
    net = FaltandoCNNV2()
    net.load_state_dict(ckpt["state_dict"], strict=True)

    # Select among start model and trained epochs ONLY using development data.
    def development_assess():
        return (
            _assess(net, historical, size, focus),
            _assess(net, online, size, focus),
            _assess(net, train, size, focus),
        )

    current_hist, current_online, current_train = development_assess()
    best_score = _selection_score(current_hist, current_online, current_train)
    best_epoch = 0
    best_state = deepcopy(net.state_dict())

    # Holdout is intentionally absent from ALL training and selection steps.
    for parameter in net.encoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        net.head.parameters(), lr=lr, weight_decay=.005,
    )
    history = [{
        "epoch": 0, "phase": "original_blend",
        "historical_correct": current_hist["correct_images"],
        "historical_total": current_hist["images"],
        "online_correct": current_online["correct_images"],
        "online_ng_false_ok": current_online["ng_false_ok"],
        "train_correct": current_train["correct_images"],
    }]
    for epoch in range(1, epochs + 1):
        net.eval()
        replay, counts = _select_hard_pairs(
            net, historical, online, train, size, focus, per_light=per_light,
        )
        if {e["label"] for e in replay} != {"OK", "NG"}:
            raise ValueError("Replay sem ambas as classes")
        loader = DataLoader(
            PairDataset(replay, size, focus), batch_size=2,
            shuffle=False, num_workers=0,
        )
        running = 0.
        net.train()
        for r, t, rf, tf, mask, truth in loader:
            logits, by_light = net(r, t, rf, tf, mask)
            # Increase per-light gradient, especially on MID missed by event.
            loss = _supervised_loss(
                logits, by_light, mask, truth, light_weight=.5,
            )
            if not bool(torch.isfinite(loss)):
                raise ValueError("Loss de reparo não finita")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.head.parameters(), 2.)
            optimizer.step()
            running += float(loss.item())
        current_hist, current_online, current_train = development_assess()
        score = _selection_score(current_hist, current_online, current_train)
        if score < best_score:
            best_epoch, best_score = epoch, score
            best_state = deepcopy(net.state_dict())
        entry = {
            "epoch": epoch,
            "mean_loss": round(running / len(loader), 6),
            "replay": counts,
            "historical_correct": current_hist["correct_images"],
            "historical_total": current_hist["images"],
            "historical_all_correct": current_hist["all_correct"],
            "online_correct": current_online["correct_images"],
            "online_ng_false_ok": current_online["ng_false_ok"],
            "train_correct": current_train["correct_images"],
        }
        history.append(entry)
        print(
            f"REPAIR {epoch}/{epochs} loss={entry['mean_loss']:.6f} "
            f"historical={entry['historical_correct']}/{entry['historical_total']} "
            f"online={entry['online_correct']} "
            f"ng_replay={counts['NG']} ok_replay={counts['OK']}",
            flush=True,
        )

    net.load_state_dict(best_state, strict=True)
    new_historical, new_online, new_train = development_assess()
    # IMPORTANT: Only ONE holdout assessment, after all epoch choices.
    old_holdout = _assess(champion, hold, size, focus)
    new_holdout = _assess(net, hold, size, focus)
    acceptance = candidate_acceptance(
        old_historical, new_historical, old_online, new_online,
        old_holdout, new_holdout, minimum_holdout_accuracy=.97,
    )
    when = datetime.now(timezone.utc)
    out = root / "reports/faltando_neural/archive_repairs" / (
        when.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:6]
    )
    out.mkdir(parents=True, exist_ok=False)
    candidate = out / "faltando_v2_guarded_light_repair_candidate.pt"
    torch.save({
        **ckpt,
        "state_dict": {
            k: v.detach().cpu().clone() for k, v in net.state_dict().items()
        },
        "experimental": True, "production_approved": False,
        "archive_labels_experimental": True,
        "individual_pair_review_complete": False,
        "guarded_light_repair_offline": True,
        "selected_repair_epoch": best_epoch,
    }, candidate)
    result = {
        "schema": SCHEMA, "created_at_utc": when.isoformat(),
        "candidate_only": True, "promoted": False,
        "automatic_ok_authorized": False, "production_approval_eligible": False,
        "individual_pair_review_complete": False,
        "holdout_used_for_selection": False,
        "holdout_previously_examined_in_prior_experiments": True,
        "independent_validation_claim": False,
        "start_blend_sha256": source["candidate_sha256"],
        "base_checkpoint_sha256": _sha(base_path),
        "manifest_sha256": provenance["manifest_sha256"],
        "split": split, "training_only_head": True,
        "selected_epoch": best_epoch, "epochs": epochs,
        "learning_rate": lr, "per_light_supervision_weight": .5,
        "history": history,
        "candidate_relative_path": candidate.relative_to(root).as_posix(),
        "candidate_sha256": _sha(candidate),
        "old_historical": old_historical,
        "new_historical": new_historical,
        "old_online": old_online, "new_online": new_online,
        "old_holdout": old_holdout, "new_holdout": new_holdout,
        "new_train": new_train,
        "acceptance": acceptance,
        "operational_threshold_diagnostics": {
            "new_historical": _operational_light_summary(new_historical),
            "new_holdout": _operational_light_summary(new_holdout),
        },
        "limitations": [
            "Usa rótulos provisórios das pastas, não qualificação individual.",
            "Holdout de 87 imagens já visto: não é avaliação independente.",
            "Não substitui validação cruzada de DESLOCADO, INVERTIDO, EMBORCADO.",
            "Candidato SEMPRE bloqueado para uso em produção.",
        ],
    }
    report = out / "repair_report.json"
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("Época escolhida (sem holdout):", best_epoch, flush=True)
    print("Holdout final:", new_holdout["correct_images"], "/", new_holdout["images"], flush=True)
    print(
        "Gates:", "APROVADOS (SOMENTE OFFLINE)" if acceptance["passed"]
        else "REPROVADOS", "— SEM PROMOÇÃO", flush=True,
    )
    print("Relatório:", report, flush=True)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Reparo CNN V2 por iluminação, somente offline")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--start-report", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=.00001)
    p.add_argument("--per-light", type=int, default=12)
    args = p.parse_args(argv)
    repair_offline(
        args.root, args.manifest, args.start_report,
        epochs=args.epochs, lr=args.lr, per_light=args.per_light,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
