"""Fine-tuning REAL da CNN FALTANDO V2 com pares históricos qualificados.

IMPORTANTE: treino exclusivamente OFFLINE de candidato. Nunca altera
live_active.json, checkpoint original, journal, KNN ou comandos XP.
O rótulo da pasta OK/NG NUNCA é uma confirmação humana suficiente.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from uuid import uuid4

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, LIGHTS, MODEL_SCHEMA_V2
from src.scripts.train_faltando_cnn_v2_online import (
    PairDataset, _all_online, _assess, _base_model, _historical,
    _supervised_loss,
)
from src.services.faltando_neural_qualification import (
    QualificationStore, QUALIFICATION_SCHEMA,
)
from src.services.faltando_neural_archive_experiment import (
    archive_labeled_experimental_events,
)

SCHEMA = "visionx.faltando_verified_archive_finetuning.v1"
MIN_PER_LABEL_TRAIN = 2
MIN_PER_LABEL_HOLDOUT = 2


def _digest(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Arquivo de treinamento ausente ou link simbólico")
    return sha256(path.read_bytes()).hexdigest()


def _stamp(event: dict) -> tuple:
    """Hash exato por luz do par de imagens, não do nome do arquivo."""
    return tuple(
        (light, _digest(view["reference"]), _digest(view["test"]))
        for light, view in sorted(event["views"].items())
    )


def _visual_key(path: Path) -> tuple[bytes, np.ndarray]:
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or min(image.shape[:2]) < 12:
        raise ValueError("Recorte qualificado ilegível")
    reduced = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    fingerprint = np.packbits((reduced[:, 1:] > reduced[:, :-1]).reshape(-1)).tobytes()
    thumbnail = cv2.resize(image, (24, 24), interpolation=cv2.INTER_AREA)
    return fingerprint, thumbnail


def qualified_archive_events(manifest_path: Path) -> tuple[list[dict], dict]:
    """Só pares realmente conferidos no painel humano de qualificação.

    Não usa agrupamento heurístico por nome. Uma associação SIDE/TOP/MID
    depende de confirmação humana e dos hashes das três origens.
    """
    store = QualificationStore(Path(manifest_path))
    if not store.output.exists():
        raise ValueError("qualification.json ausente; revise os pares no painel")
    raw = json.loads(store.output.read_text(encoding="utf-8"))
    if (
        raw.get("schema") != QUALIFICATION_SCHEMA
        or raw.get("automated_training_allowed") is not False
        or raw.get("manifest_sha256") != _digest(store.manifest_path)
    ):
        raise ValueError("Qualificação inválida/incompatível com o manifesto atual")
    events, consumed = [], set()
    used = Counter()

    def confirmed(path: str) -> tuple[str, dict] | None:
        if path not in store.by_path:
            raise ValueError("Qualificação referencia caso ausente")
        review = store.case_reviews.get(path, {})
        state = review.get("status")
        if state not in {"CONFIRMED_OK", "CONFIRMED_NG"}:
            return None
        label = state.split("_", 1)[1]
        if (
            review.get("pair_inspected") is not True
            or review.get("reviewed_label") != label
            or review.get("archive_label") != store.expected_labels[path]
            or review.get("source_sha256") != store.by_path[path]["source_sha256"]
            or not review.get("reviewed_at_utc")
        ):
            raise ValueError("Rótulo humano sem prova de inspeção do par AOI")
        store._validate_source(path)
        _, ref, test = store.files[path]
        return label, {"reference": ref, "test": test}

    for name, group in sorted(store.groups.items()):
        human = store.group_reviews.get(name, {})
        if human.get("status") != "CONFIRMED_VISUAL_ASSOCIATION":
            continue
        if human.get("sources_sha256") != store._group_sha(name):
            raise ValueError("Grupo visual alterado desde revisão humana")
        if human.get("human_group_id") != name or not human.get("reviewed_at_utc"):
            raise ValueError("Grupo sem identificação de revisão humana")
        parts = [(light, group["paths"][light], confirmed(group["paths"][light]))
                 for light in LIGHTS]
        if any(pair is None for _, _, pair in parts):
            raise ValueError("Trinca confirmada contém par sem revisão")
        labels = {pair[0] for _, _, pair in parts}
        if len(labels) != 1:
            raise ValueError("Vínculo SIDE/TOP/MID com rótulos conflitantes")
        if any(path in consumed for _, path, _ in parts):
            raise ValueError("Par qualificado pertence a grupos sobrepostos")
        labels_value = labels.pop()
        events.append({
            "id": "reviewed_group:" + name, "label": labels_value,
            "views": {light: pair[1] for light, _, pair in parts},
            "sources": [path for _, path, _ in parts],
            "grouping": "HUMAN_VISUAL_ASSOCIATION",
            "component": tuple(
                str(store.by_path[parts[0][1]].get("ocr_observed", {}).get(k, "")).strip().upper()
                for k in ("board", "parts")
            ),
        })
        consumed.update(path for _, path, _ in parts)
        used[labels_value] += len(parts)
    for path in sorted(store.by_path):
        if path in consumed:
            continue
        answer = confirmed(path)
        if answer is None:
            continue
        label, view = answer
        light = store.by_path[path]["lighting_mode"]
        events.append({
            "id": "reviewed_single:" + path, "label": label,
            "views": {light: view}, "sources": [path],
            "grouping": "HUMAN_CONFIRMED_SINGLE",
            "component": tuple(
                str(store.by_path[path].get("ocr_observed", {}).get(k, "")).strip().upper()
                for k in ("board", "parts")
            ),
        })
        used[label] += 1

    if not events:
        raise ValueError("Nenhum par com rótulo humano confirmado")
    # Fotos duplicadas e fotos originalmente em pastas opostas podem
    # corresponder à mesma peça: recusar contradições ao invés de votar.
    signatures = {}
    unique = []
    for event in events:
        key = _stamp(event)
        if key in signatures:
            if signatures[key] != event["label"]:
                raise ValueError("Mesma imagem qualificada como NG e OK")
            continue
        signatures[key] = event["label"]
        unique.append(event)
    return unique, {
        "manifest": str(store.manifest_path),
        "manifest_sha256": _digest(store.manifest_path),
        "qualification_sha256": _digest(store.output),
        "confirmed_source_images": sum(len(e["views"]) for e in unique),
        "confirmed_labels": dict(Counter(e["label"] for e in unique)),
        "human_multilight_groups": sum(
            e["grouping"] == "HUMAN_VISUAL_ASSOCIATION" for e in unique
        ),
        "excluded_unconfirmed": len(store.by_path) - len(consumed)
                              - sum(len(e["sources"]) for e in unique
                                    if e["grouping"] == "HUMAN_CONFIRMED_SINGLE"),
    }


def split_verified_events(
    events: list[dict], *, seed: int = 42, holdout: float = .25,
    visual_links: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    """Holdout bloqueado por pares quase repetidos e OCR board/parts.

    Trincas confirmadas são uma unidade. Dois pares visualmente semelhantes
    (duas imagens, mesma luz) não podem atravessar treino/holdout.
    """
    if not .15 <= holdout <= .40:
        raise ValueError("holdout deve estar entre .15 e .40")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Seed inválida")
    if len({e["id"] for e in events}) != len(events):
        raise ValueError("IDs de eventos duplicados")
    parents = list(range(len(events)))

    def find(n):
        while n != parents[n]:
            parents[n] = parents[parents[n]]
            n = parents[n]
        return n

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[b] = a

    cached = {}
    components = {}
    split_groups = {}
    for index, event in enumerate(events):
        split_group = event.get("split_group")
        if split_group is not None:
            if not isinstance(split_group, str) or not split_group:
                raise ValueError("Grupo de separação inválido")
            if split_group in split_groups:
                union(index, split_groups[split_group])
            else:
                split_groups[split_group] = index
        component = event.get("component")
        if (isinstance(component, tuple) and len(component) == 2
                and all(component)):
            if component in components:
                union(index, components[component])
            else:
                components[component] = index
        if event.get("label") not in ("OK", "NG") or not event.get("views"):
            raise ValueError("Evento sem rótulo/par")
        for light, pair in event["views"].items():
            if light not in LIGHTS:
                raise ValueError("Iluminação não reconhecida")
            key = tuple(_digest(pair[p]) for p in ("reference", "test"))
            if (light, key) in cached:
                union(index, cached[(light, key)])
            else:
                cached[(light, key)] = index
    if visual_links:
        # Comparar minúsculos thumbnails somente OFFLINE. Para grandes acervos
        # o custo é localizado por luz, com fallback explícito sem vazamento.
        visual = defaultdict(list)
        for index, event in enumerate(events):
            for light, pair in event["views"].items():
                digest_r, thumbnail_r = _visual_key(pair["reference"])
                digest_t, thumbnail_t = _visual_key(pair["test"])
                visual[light].append((index, digest_r, digest_t,
                                      thumbnail_r, thumbnail_t))
        for light, entries in visual.items():
            for i, (ai, ar, at, aimg_r, aimg_t) in enumerate(entries):
                for bi, br, bt, bimg_r, bimg_t in entries[i+1:]:
                    # 4 bits de tolerância por imagem e 6 níveis MAE no
                    # thumbnail — evita vazamento de regravações similares.
                    if (
                        sum((x ^ y).bit_count() for x, y in zip(ar, br)) <= 4
                        and sum((x ^ y).bit_count() for x, y in zip(at, bt)) <= 4
                        and float(np.mean(np.abs(aimg_r.astype(np.int16)
                            - bimg_r.astype(np.int16)))) <= 6
                        and float(np.mean(np.abs(aimg_t.astype(np.int16)
                            - bimg_t.astype(np.int16)))) <= 6
                    ):
                        union(ai, bi)
    groups = defaultdict(list)
    for i in range(len(events)):
        groups[find(i)].append(events[i])
    for group in groups.values():
        if len({e["label"] for e in group}) != 1:
            raise ValueError("OK e NG visualmente duplicados: qualificação conflitante")

    by_label = defaultdict(list)
    for group in groups.values():
        by_label[group[0]["label"]].append(group)
    if set(by_label) != {"OK", "NG"}:
        raise ValueError("Treino histórico requer NG e OK humanos")
    rng = random.Random(seed)
    train, test = [], []
    for label in ("OK", "NG"):
        batches = sorted(by_label[label], key=lambda g: min(e["id"] for e in g))
        rng.shuffle(batches)
        if len(batches) < 2:
            raise ValueError("Poucos grupos independentes para holdout " + label)
        count = max(1, min(len(batches)-1, round(len(batches) * holdout)))
        test.extend(e for group in batches[:count] for e in group)
        train.extend(e for group in batches[count:] for e in group)
    ctrain, ctest = Counter(e["label"] for e in train), Counter(e["label"] for e in test)
    if any(ctrain[l] < MIN_PER_LABEL_TRAIN or ctest[l] < MIN_PER_LABEL_HOLDOUT
           for l in ("NG", "OK")):
        raise ValueError("Holdout não possui dois NG e dois OK independentes")
    return train, test, {
        "seed": seed, "holdout_requested": holdout,
        "groups": len(groups),
        "train_events": len(train), "holdout_events": len(test),
        "train_by_label": dict(ctrain), "holdout_by_label": dict(ctest),
        "no_shared_exact_pairs": True,
        "no_near_duplicates_cross_boundary": bool(visual_links),
    }


def candidate_acceptance(
    old_anchors: dict, new_anchors: dict, old_online: dict,
    new_online: dict, old_holdout: dict, new_holdout: dict,
    *, minimum_holdout_accuracy: float | None = None,
) -> dict:
    """Exige NG e OK melhores/iguais em holdout, sem perder histórico."""
    def errors(result: dict) -> tuple[int, int]:
        return (
            sum(c["label"] == "NG" and
                (not c["correct"] or any(not x["correct"]
                    for x in c["per_light"].values()))
                for c in result["cases"]),
            sum(c["label"] == "OK" and
                (not c["correct"] or any(not x["correct"]
                    for x in c["per_light"].values()))
                for c in result["cases"]),
        )
    if minimum_holdout_accuracy is not None and not (
        .5 < minimum_holdout_accuracy <= 1.
    ):
        raise ValueError("Meta mínima de acurácia fora dos limites")
    holdout_images = int(new_holdout.get("images", sum(
        len(case["per_light"]) for case in new_holdout["cases"]
    )))
    accuracy = (
        new_holdout["correct_images"] / holdout_images
        if holdout_images else 0.
    )
    old_ng, old_ok = errors(old_holdout)
    new_ng, new_ok = errors(new_holdout)
    gates = {
        "historical_replay_all_correct": new_anchors["all_correct"],
        "online_no_regression": (
            new_online["correct_images"] >= old_online["correct_images"]
            and new_online["ng_false_ok"] <= old_online["ng_false_ok"]
        ),
        "holdout_ng_not_worse": new_ng <= old_ng,
        "holdout_ok_not_worse": new_ok <= old_ok,
        "holdout_zero_new_false_ok": new_holdout["ng_false_ok"] <=
                                       old_holdout["ng_false_ok"],
        "holdout_both_labels_present": (
            {"OK", "NG"} <= {c["label"] for c in new_holdout["cases"]}
        ),
        "historical_no_new_false_ok": (
            new_anchors["ng_false_ok"] <= old_anchors["ng_false_ok"]
        ),
        "holdout_accuracy_minimum": (
            minimum_holdout_accuracy is None
            or accuracy >= minimum_holdout_accuracy - 1e-12
        ),
    }
    return {
        "gates": gates, "passed": all(gates.values()),
        "baseline_holdout_ng_errors": old_ng,
        "candidate_holdout_ng_errors": new_ng,
        "baseline_holdout_ok_errors": old_ok,
        "candidate_holdout_ok_errors": new_ok,
        "holdout_accuracy": round(accuracy, 6),
        "holdout_accuracy_minimum": minimum_holdout_accuracy,
        "holdout_accuracy_goal": .98 if minimum_holdout_accuracy else None,
        "holdout_goal_reached": (
            accuracy >= .98 if minimum_holdout_accuracy is not None else None
        ),
    }


def _label_light_errors(result: dict, label: str) -> int:
    """Count light-level misclassifications; never substitute event accuracy."""
    return sum(
        not view["correct"]
        for case in result["cases"] if case["label"] == label
        for view in case["per_light"].values()
    )


def _balanced_hard_replay(
    hard: list[dict], anchors: list[dict], model: FaltandoCNNV2,
    size: int, focus: float, *, seed: int, epoch: int, epochs: int,
) -> tuple[list[dict], dict]:
    """Balance real gradient steps NG/OK; never inspect the held-out cases.

    Rotate the errors of the original checkpoint through epochs. Each paired
    batch contains an NG and an OK; prioritize OK currently nearest to a false
    NG classification, including proven historical OK anchors.
    """
    by_label = {
        label: [x for x in hard if x["label"] == label]
        for label in ("NG", "OK")
    }
    if not any(by_label.values()):
        return [], {"NG": 0, "OK": 0, "paired_steps": 0}
    if not 1 <= epoch <= epochs:
        raise ValueError("Época fora do intervalo")
    count = min(48, max(8, math.ceil(max(map(len, by_label.values())) / epochs)))
    target = {}
    for label in ("NG", "OK"):
        pool = by_label[label]
        others = [e for e in anchors if e["label"] == label]
        if not others:
            raise ValueError("Sem âncoras OK/NG para reforço balanceado")
        if label == "OK":
            scored = _assess(model, others, size, focus)
            if len(scored["cases"]) != len(others):
                raise ValueError("Diagnóstico de OK incompleto no replay")
            # Worst individual illumination controls the risk of false NG.
            ranked = sorted(
                zip(others, scored["cases"]),
                key=lambda pair: (
                    -max(float(v["score"]) for v in pair[1]["per_light"].values()),
                    pair[0]["id"],
                ),
            )
            target[label] = [ranked[i % len(ranked)][0] for i in range(count)]
        else:
            pool = pool if pool else others
            indexes = list(range(len(pool)))
            random.Random(seed + 1009).shuffle(indexes)
            offset = (epoch - 1) * count
            target[label] = [
                pool[indexes[(offset + i) % len(indexes)]]
                for i in range(count)
            ]
    samples = [
        event
        for pair in zip(target["NG"], target["OK"])
        for event in pair
    ]
    if len(samples) != 2 * count:
        raise ValueError("Reforço com classes desequilibradas")
    return samples, {"NG": count, "OK": count, "paired_steps": count}


def _selection_score(historical: dict, online: dict, development: dict) -> tuple:
    """Choose an epoch from training/anchors ONLY, never from the holdout."""
    ok_errors = (_label_light_errors(historical, "OK")
                 + _label_light_errors(online, "OK"))
    old_ng_errors = _label_light_errors(historical, "NG")
    return (
        int(not historical["all_correct"]),
        ok_errors,
        old_ng_errors,
        online["ng_false_ok"],
        _label_light_errors(development, "NG"),
        -online["correct_images"],
        -development["correct_images"],
    )


def _operational_light_summary(result: dict) -> dict:
    """Diagnostics at real 0.10 / 0.90 review thresholds; no XP decisions."""
    summary = {
        light: {
            "total": 0, "correct_conclusive": 0, "unmeasured": 0,
            "review": 0, "ng_as_conclusive_ok": 0,
            "ok_as_conclusive_ng": 0,
        }
        for light in LIGHTS
    }
    for case in result["cases"]:
        for light, view in case["per_light"].items():
            entry = summary[light]
            entry["total"] += 1
            if "score" not in view:
                entry["unmeasured"] += 1
                continue
            score = float(view["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Score CNN inválido para limiares operacionais")
            if .10 < score < .90:
                entry["review"] += 1
            elif (score <= .10 and case["label"] == "NG"):
                entry["ng_as_conclusive_ok"] += 1
            elif (score >= .90 and case["label"] == "OK"):
                entry["ok_as_conclusive_ng"] += 1
            else:
                entry["correct_conclusive"] += 1
    return summary


def train_verified_archive(
    root: Path, manifest: Path, *, epochs: int = 5, seed: int = 42,
    holdout: float = .25, lr: float = .00002,
    archive_labels_experimental: bool = False,
) -> dict:
    """Produz SOMENTE checkpoint experimental sem ponteiro de ativação."""
    root = Path(root).resolve()
    if not 1 <= epochs <= 25 or not 0 < lr <= .0001:
        raise ValueError("Parâmetros de treino fora dos limites")
    if (root / "reports/neural_online/live_active.json").exists():
        raise ValueError("Checkpoint online ativo: auditar antes de novo experimento")
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Folder-label opt-in is an experiment, NOT individual human verification.
    verified, provenance = (
        archive_labeled_experimental_events(manifest)
        if archive_labels_experimental else qualified_archive_events(manifest)
    )
    if Path(provenance["manifest"]).resolve().parents[2] != root / "reports":
        raise ValueError("Dataset qualificado pertence a outra instalação")
    train, hold, split = split_verified_events(verified, seed=seed, holdout=holdout)
    baseline_path, checkpoint = _base_model(root)
    historical = _historical(root, checkpoint)
    online = _all_online(root)
    size = int(checkpoint["image_size"])
    focus = float(checkpoint["focus_fraction"])
    if not (64 <= size <= 512 and .5 <= focus <= .90):
        raise ValueError("Pré-processamento CNN inválido")
    # Evitar rótulos conflitantes e vazamento de pares já usados no replay
    # histórico (checkpoint pré-existente). O holdout é de fine-tuning,
    # não teste cego da experiência passada do champion.
    identity_labels = {}
    for event in historical + online + verified:
        key = _stamp(event)
        label = event["label"]
        if key in identity_labels and identity_labels[key] != label:
            raise ValueError("NG/OK contraditórios entre histórico, online e arquivo")
        identity_labels[key] = label

    champion = FaltandoCNNV2()
    champion.load_state_dict(checkpoint["state_dict"], strict=True)
    old_hist = _assess(champion, historical, size, focus)
    old_online = _assess(champion, online, size, focus)
    old_hold = _assess(champion, hold, size, focus)
    if not old_hist["all_correct"]:
        raise ValueError("Champion já falhou no histórico integral; treinar isoladamente")

    model = FaltandoCNNV2()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    examples = historical + online + train
    frequency = Counter(row["label"] for row in examples)
    weights = [1/frequency[row["label"]] for row in examples]
    loader = DataLoader(
        PairDataset(examples, size, focus), batch_size=4,
        sampler=WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double), num_samples=len(examples),
            replacement=True, generator=torch.Generator().manual_seed(seed),
        ),
        num_workers=0,
    )
    # Baseline errors are the hardest NG. Keep the held-out images entirely
    # outside both the ordinary sampler and the paired hard replay.
    baseline_train = _assess(champion, train, size, focus)
    forced = [
        event for event, case in zip(train, baseline_train["cases"])
        if not case["correct"] or any(
            not view["correct"] for view in case["per_light"].values()
        )
    ]
    training_ids = {event["id"] for event in examples}
    heldout_ids = {event["id"] for event in hold}
    if training_ids & heldout_ids:
        raise ValueError("Vazamento de holdout no replay")
    anchors = historical + online + train
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.02)
    history = []
    best_state = None
    best_score = None
    best_epoch = None
    # Never use the holdout for checkpoint/epoch selection.
    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for r, t, rf, tf, mask, truth in loader:
            logits, per_light = model(r, t, rf, tf, mask)
            loss = _supervised_loss(logits, per_light, mask, truth,
                                    light_weight=.15)
            if not torch.isfinite(loss):
                raise ValueError("Perda não finita")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            running += float(loss.item())

        if archive_labels_experimental:
            balanced, replay_counts = _balanced_hard_replay(
                forced, anchors, model, size, focus,
                seed=seed, epoch=epoch, epochs=epochs,
            )
        else:
            # Original strict individually verified route stays unchanged.
            balanced = forced
            replay_counts = dict(Counter(e["label"] for e in forced))
        if balanced:
            forced_loader = DataLoader(
                PairDataset(balanced, size, focus),
                batch_size=2 if archive_labels_experimental else 1,
                shuffle=False, num_workers=0,
            )
            model.train()
            for r, t, rf, tf, mask, truth in forced_loader:
                logits, per_light = model(r, t, rf, tf, mask)
                loss = _supervised_loss(logits, per_light, mask, truth,
                                        light_weight=.15)
                if not torch.isfinite(loss):
                    raise ValueError("Perda forçada não finita")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()

        epoch_entry = {
            "epoch": epoch,
            "training_loss": round(running/len(loader), 6),
            "forced_replay_by_label": {
                label: int(replay_counts.get(label, 0))
                for label in ("NG", "OK")
            },
        }
        if archive_labels_experimental:
            assessment_hist = _assess(model, historical, size, focus)
            assessment_online = _assess(model, online, size, focus)
            assessment_train = _assess(model, train, size, focus)
            score = _selection_score(
                assessment_hist, assessment_online, assessment_train,
            )
            epoch_entry["selection_without_holdout"] = {
                "historical_all_correct": assessment_hist["all_correct"],
                "historical_ok_light_errors": _label_light_errors(
                    assessment_hist, "OK"
                ),
                "online_ok_light_errors": _label_light_errors(
                    assessment_online, "OK"
                ),
                "online_ng_false_ok": assessment_online["ng_false_ok"],
                "train_ng_light_errors": _label_light_errors(
                    assessment_train, "NG"
                ),
            }
            if best_score is None or score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = deepcopy({
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                })
        history.append(epoch_entry)
        print(f"FALTANDO VERIFIED {epoch}/{epochs} "
              f"loss={running/len(loader):.6f} "
              f"replay_NG={replay_counts.get('NG', 0)} "
              f"replay_OK={replay_counts.get('OK', 0)}", flush=True)

    if archive_labels_experimental:
        if best_state is None:
            raise ValueError("Nenhuma época candidata foi selecionada")
        model.load_state_dict(best_state, strict=True)
        print(f"Melhor época por âncoras de desenvolvimento: {best_epoch} "
              "(sem consultar o holdout)", flush=True)
    new_hist = _assess(model, historical, size, focus)
    new_online = _assess(model, online, size, focus)
    new_hold = _assess(model, hold, size, focus)
    acceptance = candidate_acceptance(
        old_hist, new_hist, old_online, new_online, old_hold, new_hold,
        minimum_holdout_accuracy=.97 if archive_labels_experimental else None,
    )
    when = datetime.now(timezone.utc)
    destination = ("archive_experiments" if archive_labels_experimental
                   else "verified_candidates")
    out = (root / "reports/faltando_neural" / destination /
           (when.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:6]))
    out.mkdir(parents=True, exist_ok=False)
    checkpoint_out = out / (
        "faltando_v2_archive_experimental_candidate.pt"
        if archive_labels_experimental else "faltando_v2_verified_candidate.pt"
    )
    torch.save({
        **checkpoint,
        "state_dict": {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()},
        "verified_archive_training_schema": SCHEMA,
        "experimental": True, "production_approved": False,
        **({"provisional_archive_manifest_sha256": provenance["manifest_sha256"],
            "individual_pair_review_complete": False,
            "archive_labels_experimental": True}
           if archive_labels_experimental else {
            "verified_source_manifest_sha256": provenance["manifest_sha256"],
            "verified_qualification_sha256": provenance["qualification_sha256"],
           }),
        "source_champion_sha256": _digest(baseline_path),
    }, checkpoint_out)
    report = {
        "schema": SCHEMA, "created_at_utc": when.isoformat(),
        "candidate_only": True, "promoted": False,
        "automatic_ok_authorized": False,
        "production_approval_eligible": False,
        "qualification_mode": ("ARCHIVE_FOLDER_BULK_EXPERIMENT"
                               if archive_labels_experimental else "INDIVIDUALLY_VERIFIED"),
        "base_checkpoint_sha256": _digest(baseline_path),
        "candidate_relative_path": checkpoint_out.relative_to(root).as_posix(),
        "candidate_sha256": _digest(checkpoint_out),
        "provenance": provenance, "split": split,
        "train_event_ids": [e["id"] for e in train],
        "holdout_event_ids": [e["id"] for e in hold],
        "train_forced_misclassified": [e["id"] for e in forced],
        "epochs": epochs, "learning_rate": lr, "history": history,
        "selected_epoch": best_epoch if archive_labels_experimental else epochs,
        "epoch_selection_uses_holdout": False,
        "objective": (
            {"target_accuracy": .98, "minimum_acceptable_accuracy": .97,
             "evaluation_split": "holdout", "classification_threshold": .50}
            if archive_labels_experimental else None
        ),
        "operational_threshold_diagnostics": (
            {"old_holdout": _operational_light_summary(old_hold),
             "new_holdout": _operational_light_summary(new_hold),
             "old_historical": _operational_light_summary(old_hist),
             "new_historical": _operational_light_summary(new_hist)}
            if archive_labels_experimental else None
        ),
        "old_historical": old_hist, "new_historical": new_hist,
        "old_online": old_online, "new_online": new_online,
        "old_holdout": old_hold, "new_holdout": new_hold,
        "acceptance": acceptance,
        "limitations": [
            "Holdout separado do fine-tuning atual; checkpoint base pode ter "
            "visto imagens semelhantes no histórico antigo.",
            "Avaliação de acurácia offline, não é aprovação de produção.",
            "Sem promoção automática de pesos: requer replay completo do "
            "acervo e autorização humana posterior.",
            *(["Rótulos das pastas presumidos a pedido do operador; NÃO houve "
               "revisão visual individual das imagens.",
               "Não é possível usar este experimento como aprovação de "
               "produção nem validar independência física sem auditoria.",]
              if archive_labels_experimental else []),
        ],
    }
    (out / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Candidato treinado:", checkpoint_out, flush=True)
    print("Gates:", "APROVADOS" if acceptance["passed"] else "REPROVADOS",
          "— SEM PROMOÇÃO", flush=True)
    print("Relatório:", out / "training_report.json", flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tuning offline CNN V2 com arquivo AOI humano qualificado"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=float, default=.25)
    parser.add_argument("--lr", type=float, default=.00002)
    parser.add_argument(
        "--archive-labels-experimental", action="store_true",
        help=("Aceita explicitamente rótulos OK/NG das pastas SEM revisão "
              "individual, apenas para candidato OFFLINE NÃO PROMOVÍVEL"),
    )
    args = parser.parse_args(argv)
    train_verified_archive(
        args.root, args.manifest, epochs=args.epochs,
        seed=args.seed, holdout=args.holdout, lr=args.lr,
        archive_labels_experimental=args.archive_labels_experimental,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
