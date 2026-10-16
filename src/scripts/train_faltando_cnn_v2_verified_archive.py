"""Fine-tuning REAL da CNN FALTANDO V2 com pares históricos qualificados.

IMPORTANTE: treino exclusivamente OFFLINE de candidato. Nunca altera
live_active.json, checkpoint original, journal, KNN ou comandos XP.
O rótulo da pasta OK/NG NUNCA é uma confirmação humana suficiente.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    for index, event in enumerate(events):
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
    }
    return {
        "gates": gates, "passed": all(gates.values()),
        "baseline_holdout_ng_errors": old_ng,
        "candidate_holdout_ng_errors": new_ng,
        "baseline_holdout_ok_errors": old_ok,
        "candidate_holdout_ok_errors": new_ok,
    }


def train_verified_archive(
    root: Path, manifest: Path, *, epochs: int = 5, seed: int = 42,
    holdout: float = .25, lr: float = .00002,
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
    verified, provenance = qualified_archive_events(manifest)
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
    # Garantir gradiente dos NG/OK históricos com erros, sem expor holdout.
    baseline_train = _assess(champion, train, size, focus)
    forced = [
        event for event, case in zip(train, baseline_train["cases"])
        if not case["correct"] or any(
            not view["correct"] for view in case["per_light"].values()
        )
    ]
    # Não deixar centenas de fotos idênticas abafarem os NG difíceis:
    # cada evento selecionado recebe um passo adicional por época.
    forced_loader = (
        DataLoader(PairDataset(forced, size, focus), batch_size=1,
                   shuffle=False, num_workers=0) if forced else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.02)
    history = []
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
        if forced_loader is not None:
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
        history.append({
            "epoch": epoch, "training_loss": round(running/len(loader), 6)
        })
        print(f"FALTANDO VERIFIED {epoch}/{epochs} "
              f"loss={running/len(loader):.6f}", flush=True)

    new_hist = _assess(model, historical, size, focus)
    new_online = _assess(model, online, size, focus)
    new_hold = _assess(model, hold, size, focus)
    acceptance = candidate_acceptance(
        old_hist, new_hist, old_online, new_online, old_hold, new_hold,
    )
    when = datetime.now(timezone.utc)
    out = (root / "reports/faltando_neural/verified_candidates" /
           (when.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:6]))
    out.mkdir(parents=True, exist_ok=False)
    checkpoint_out = out / "faltando_v2_verified_candidate.pt"
    torch.save({
        **checkpoint,
        "state_dict": {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()},
        "verified_archive_training_schema": SCHEMA,
        "experimental": True, "production_approved": False,
        "verified_source_manifest_sha256": provenance["manifest_sha256"],
        "verified_qualification_sha256": provenance["qualification_sha256"],
        "source_champion_sha256": _digest(baseline_path),
    }, checkpoint_out)
    report = {
        "schema": SCHEMA, "created_at_utc": when.isoformat(),
        "candidate_only": True, "promoted": False,
        "automatic_ok_authorized": False,
        "base_checkpoint_sha256": _digest(baseline_path),
        "candidate_relative_path": checkpoint_out.relative_to(root).as_posix(),
        "candidate_sha256": _digest(checkpoint_out),
        "provenance": provenance, "split": split,
        "train_event_ids": [e["id"] for e in train],
        "holdout_event_ids": [e["id"] for e in hold],
        "train_forced_misclassified": [e["id"] for e in forced],
        "epochs": epochs, "learning_rate": lr, "history": history,
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
    args = parser.parse_args(argv)
    train_verified_archive(
        args.root, args.manifest, epochs=args.epochs,
        seed=args.seed, holdout=args.holdout, lr=args.lr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
