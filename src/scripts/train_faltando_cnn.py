"""Treinamento experimental local da CNN FALTANDO, sem habilitar Produção.

Uso: python -m src.scripts.train_faltando_cnn --epochs 25

Usa os pares extraídos pela AOI nos reports/faltando_neural/run_*/pairs.
Rótulos OK/NG de arquivos históricos são aceitos por decisão explícita do
operador; trincas por nome só são usadas como grupo de treinamento quando
board/parts/value OCR são coerentes. Não requer qualification.json.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import re

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.core.neural.faltando_cnn import FaltandoCNN, LIGHTS, MODEL_SCHEMA
from src.services.faltando_neural_qualification import latest_manifest

RUN_SCHEMA = "visionx.faltando_training_experiment.v1"


def _safe_file(parent: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Caminho inválido: " + str(relative))
    target = (parent / relative).resolve()
    if parent.resolve() not in target.parents or target.is_symlink() or not target.is_file():
        raise ValueError("Caminho ausente/fora do staging: " + relative)
    return target


def _text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def load_events(manifest_path: Path) -> tuple[list[dict], dict]:
    """Preserva trincas como 1 evento; sem depender do painel de revisão."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "visionx.faltando_neural_preparation.v1":
        raise ValueError("Manifesto de preparação FALTANDO incompatível")
    run = path.parent
    if run.parent.name != "faltando_neural" or run.parent.parent.name != "reports":
        raise ValueError("Treinamento requer staging em reports/faltando_neural")
    root = Path(manifest["root"]).resolve()
    if root / "reports" != run.parent.parent:
        raise ValueError("Raiz do manifesto diferente da pasta de preparação")
    sample_map = {}
    for item in manifest["samples"]:
        if item.get("status") != "EXTRACTED_PENDING_REVIEW":
            raise ValueError("Exemplo com falha de extração; não treinar")
        if item.get("expected_label_from_archive") not in ("OK", "NG"):
            raise ValueError("Rótulo inválido")
        if item.get("lighting_mode") not in LIGHTS:
            raise ValueError("Iluminação inválida")
        source = item["source_path"]
        if source in sample_map:
            raise ValueError("Origem repetida no manifesto")
        original = _safe_file(root / "public", str(Path(source).relative_to("public")))
        if original.parent.name not in ("ok_archive", "ng_archive"):
            raise ValueError("Origem fora dos archives")
        if sha256(original.read_bytes()).hexdigest() != item["source_sha256"]:
            raise ValueError("Origem mudou desde a preparação: " + source)
        for key in ("reference_path", "test_path"):
            target = _safe_file(run, item[key])
            if (run / "pairs").resolve() not in target.parents:
                raise ValueError("Recorte fora de pairs")
        sample_map[source] = item

    events = []
    included = set()
    warnings = []
    for group in manifest.get("name_only_triplet_candidates", []):
        paths = group.get("paths", {})
        if set(paths) != set(LIGHTS) or len(set(paths.values())) != 3:
            warnings.append(f"Trinca incompleta: {group.get('id')}")
            continue
        if any(p not in sample_map or p in included for p in paths.values()):
            warnings.append(f"Trinca ausente/reutilizada: {group.get('id')}")
            continue
        records = [sample_map[paths[mode]] for mode in LIGHTS]
        if len({x["expected_label_from_archive"] for x in records}) != 1:
            warnings.append(f"Rótulos contraditórios: {group.get('id')}")
            continue
        # São observações da mesma peça somente se a AOI também informou
        # o mesmo board, parts e value. Horário sozinho nunca é suficiente.
        signatures = {
            tuple(_text(x.get("ocr_observed", {}).get(field)) for field in
                  ("board", "parts", "value"))
            for x in records
        }
        if len(signatures) != 1 or not all(next(iter(signatures))[:2]):
            warnings.append(f"OCR incompatível: {group.get('id')}")
            continue
        for p in paths.values():
            included.add(p)
        events.append({
            "id": "name_coherent:" + group["id"],
            "label": records[0]["expected_label_from_archive"],
            "observations": {mode: paths[mode] for mode in LIGHTS},
            "association": "NAME_AND_OCR_COHERENT_NOT_PROVEN_EVENT_ID",
        })

    for source, item in sorted(sample_map.items()):
        if source not in included:
            events.append({
                "id": "single:" + source,
                "label": item["expected_label_from_archive"],
                "observations": {item["lighting_mode"]: source},
                "association": "SINGLE_ARCHIVE_FRAME",
            })

    for event in events:
        record = sample_map[next(iter(event["observations"].values()))]
        ocr = record.get("ocr_observed", {})
        board, part = _text(ocr.get("board")), _text(ocr.get("parts"))
        # Agrupar mesmo board+parts entre eventos para impedir vazamento.
        event["split_key"] = (
            ("component:" + board + "/" + part)
            if board and part else "event:" + event["id"]
        )

    return events, {"sample_map": sample_map, "warnings": warnings,
                    "run_dir": run, "root": root}


def _union_component_groups(events: list[dict], data: dict) -> list[list[int]]:
    """Agrupa board/parts e imagens perceptualmente quase repetidas no split."""
    parents = list(range(len(events)))

    def find(x):
        while x != parents[x]:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parents[j] = i

    by_component = {}
    for i, event in enumerate(events):
        key = event["split_key"]
        if key in by_component:
            union(i, by_component[key])
        else:
            by_component[key] = i

    # Prevent near-identical captures crossing the evaluation boundary.
    # dHash is only a conservative grouping hint, never a classifier.
    signatures = {}
    for i, event in enumerate(events):
        for light, source in event["observations"].items():
            sample = data["sample_map"][source]
            for filekey in ("reference_path", "test_path"):
                file = data["run_dir"] / sample[filekey]
                frame = cv2.imdecode(
                    np.frombuffer(file.read_bytes(), dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                )
                if frame is None:
                    raise ValueError("Recorte indecifrável: " + source)
                reduced = cv2.resize(frame, (9, 8), interpolation=cv2.INTER_AREA)
                signatures[(i, light, filekey)] = np.packbits(
                    (reduced[:, 1:] > reduced[:, :-1]).flatten()
                ).tobytes()
    for i, a in enumerate(events):
        for j in range(i+1, len(events)):
            b = events[j]
            common = set(a["observations"]) & set(b["observations"])
            for light in common:
                dist = []
                for filekey in ("reference_path", "test_path"):
                    h1 = int.from_bytes(signatures[i, light, filekey], "big")
                    h2 = int.from_bytes(signatures[j, light, filekey], "big")
                    dist.append((h1 ^ h2).bit_count())
                if max(dist) <= 4:
                    union(i, j)
                    break

    grouped = defaultdict(list)
    for i in range(len(events)):
        grouped[find(i)].append(i)
    return list(grouped.values())


def split_events(events: list[dict], data: dict, seed: int = 42,
                 holdout: float = .20) -> tuple[list[int], list[int], dict]:
    groups = _union_component_groups(events, data)
    rng = random.Random(seed)
    total = Counter(e["label"] for e in events)
    best = None
    # Search for a group-stratified split containing each class in both
    # sets. No record from the same group may cross the split.
    for _ in range(2000):
        shuffled = groups[:]
        rng.shuffle(shuffled)
        proposed = []
        running = Counter()
        for group in shuffled:
            labels = Counter(events[i]["label"] for i in group)
            if (
                sum(running.values()) < max(2, int(len(events) * holdout))
                and all(
                    running[label] + labels[label] <= total[label] - 1
                    for label in ("OK", "NG")
                )
            ):
                proposed.extend(group)
                running.update(labels)
        if not proposed:
            continue
        val = sorted(proposed)
        train = sorted(set(range(len(events))) - set(val))
        by_val = Counter(events[i]["label"] for i in val)
        by_train = Counter(events[i]["label"] for i in train)
        if any(by_val[label] < 1 or by_train[label] < 1 for label in ("OK", "NG")):
            continue
        deviation = sum(
            abs(by_val[label]/total[label] - holdout)
            for label in ("OK", "NG")
        )
        if best is None or deviation < best[0]:
            best = (deviation, train, val)
    if best is None:
        raise ValueError(
            "Não há grupos NG/OK independentes suficientes para holdout. "
            "Treino interrompido sem gerar modelo."
        )
    _, train, val = best
    overlap = set(events[i]["split_key"] for i in train) & set(
        events[i]["split_key"] for i in val
    )
    if overlap:
        raise AssertionError("Vazamento por componente no holdout")
    return train, val, {
        "groups": len(groups),
        "train": dict(Counter(events[i]["label"] for i in train)),
        "validation": dict(Counter(events[i]["label"] for i in val)),
        "strategy": "grouped_component_plus_near_duplicate",
    }


def _load_rgb(path: Path, size: int) -> torch.Tensor:
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None or image.size == 0:
        raise ValueError("Não foi possível decodificar: " + str(path))
    h, w = image.shape[:2]
    ratio = size / max(h, w)
    new_w, new_h = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Preencher com borda neutra evita esticar a geometria do componente.
    canvas = np.full((size, size, 3), 127, dtype=np.uint8)
    x, y = (size-new_w)//2, (size-new_h)//2
    canvas[y:y+new_h, x:x+new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.)


class EventDataset(Dataset):
    def __init__(self, events: list[dict], data: dict, size: int = 160,
                 augment: bool = False):
        self.events, self.data, self.size, self.augment = events, data, size, augment

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        event = self.events[index]
        reference = torch.zeros((3, 3, self.size, self.size))
        test = torch.zeros_like(reference)
        mask = torch.zeros(3, dtype=torch.float32)
        for index_light, light in enumerate(LIGHTS):
            if light not in event["observations"]:
                continue
            sample = self.data["sample_map"][event["observations"][light]]
            reference[index_light] = _load_rgb(
                self.data["run_dir"] / sample["reference_path"], self.size
            )
            test[index_light] = _load_rgb(
                self.data["run_dir"] / sample["test_path"], self.size
            )
            mask[index_light] = 1.
        if self.augment:
            # Uma operação espacial idêntica em gabarito e teste não
            # destrói a correspondência geométrica.
            if torch.rand(()) < .5:
                reference = reference.flip(-1)
                test = test.flip(-1)
            # Perturbar igualmente os dois frames evita pistas de rótulo
            # introduzidas pela variação fotométrica independente.
            gain = float(torch.empty(1).uniform_(.88, 1.12))
            offset = float(torch.empty(1).uniform_(-.035, .035))
            reference = (reference * gain + offset).clamp(0, 1)
            test = (test * gain + offset).clamp(0, 1)
        return reference, test, mask, torch.tensor(
            float(event["label"] == "NG"), dtype=torch.float32
        )


def _evaluate(model, loader, device):
    model.eval()
    tp = tn = fp = fn = 0
    loss_sum = 0.
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for ref, test, mask, truth in loader:
            ref, test, mask, truth = [
                item.to(device) for item in (ref, test, mask, truth)
            ]
            logits, _ = model(ref, test, mask)
            loss_sum += float(criterion(logits, truth).item()) * len(truth)
            predicted = (torch.sigmoid(logits) >= .5)
            positives = truth >= .5
            tp += int((predicted & positives).sum())
            tn += int((~predicted & ~positives).sum())
            fp += int((predicted & ~positives).sum())
            fn += int((~predicted & positives).sum())
    n = tp + tn + fp + fn
    return {
        "cases": n, "loss": round(loss_sum/max(n, 1), 6),
        "TP_NG": tp, "TN_OK": tn, "FP_OK_as_NG": fp,
        "FN_NG_as_OK": fn,
        "ng_recall": round(tp/(tp+fn), 4) if tp+fn else None,
        "ok_specificity": round(tn/(tn+fp), 4) if tn+fp else None,
        "accuracy": round((tp+tn)/n, 4) if n else None,
    }


def train(manifest: Path, *, epochs: int = 25, batch_size: int = 4,
          size: int = 160, seed: int = 42, holdout: float = .20,
          device: str = "cpu") -> tuple[dict, Path]:
    if not 1 <= epochs <= 500 or not 1 <= batch_size <= 128:
        raise ValueError("epochs/batch_size fora do intervalo")
    if size < 64 or size > 512 or size % 32:
        raise ValueError("size deve ser múltiplo de 32, entre 64 e 512")
    if not 0.1 <= holdout <= 0.4:
        raise ValueError("holdout deve ficar entre 0.1 e 0.4")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    events, data = load_events(manifest)
    train_indices, eval_indices, split = split_events(
        events, data, seed=seed, holdout=holdout
    )
    training = [events[i] for i in train_indices]
    evaluation = [events[i] for i in eval_indices]
    dl_train = DataLoader(
        EventDataset(training, data, size, augment=True),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    dl_eval = DataLoader(
        EventDataset(evaluation, data, size),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    model = FaltandoCNN().to(device)
    ng = sum(e["label"] == "NG" for e in training)
    ok = len(training) - ng
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(ok/max(ng, 1)), device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.
        for ref, test, mask, truth in dl_train:
            ref, test, mask, truth = [
                item.to(device) for item in (ref, test, mask, truth)
            ]
            logits, per_light = model(ref, test, mask)
            base = criterion(logits, truth)
            aux = nn.functional.binary_cross_entropy_with_logits(
                per_light, truth[:, None].expand_as(per_light),
                reduction="none",
            )
            aux = (aux * mask).sum()/mask.sum().clamp_min(1)
            loss = base + .20 * aux
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            loss_sum += float(loss.item()) * len(truth)
        validation = _evaluate(model, dl_eval, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(loss_sum/len(training), 6),
            "holdout": validation,
        })
        print(
            f"[{epoch:03d}/{epochs}] loss={history[-1]['train_loss']:.4f}"
            f" | holdout FN_NG_as_OK={validation['FN_NG_as_OK']}"
            f" FP_OK_as_NG={validation['FP_OK_as_NG']}", flush=True
        )

    now = datetime.now(timezone.utc)
    out = data["root"] / "reports" / "faltando_neural" / "models" / (
        "experiment_" + now.strftime("%Y%m%dT%H%M%S_%fZ")
    )
    out.mkdir(parents=True, exist_ok=False)
    metrics = {
        "schema": RUN_SCHEMA,
        "model_schema": MODEL_SCHEMA,
        "source_manifest": str(Path(manifest).resolve()),
        "source_manifest_sha256": sha256(Path(manifest).read_bytes()).hexdigest(),
        "trained_at_utc": now.isoformat(),
        "experimental": True,
        "production_approved": False,
        "knn_used": False,
        "manual_qualification_required": False,
        "labels_from_archive_accepted_by_operator": True,
        "event_association_policy": "name+OCR_consistent_or_single",
        "score_threshold_for_report_only": 0.5,
        "no_top_mid_ng_in_historical_set": not any(
            e["label"] == "NG" and set(e["observations"]) != {"SIDE"}
            for e in events
        ),
        "counts": {
            "frames": len(data["sample_map"]), "events": len(events),
            "labels": dict(Counter(e["label"] for e in events)),
            "by_mode": dict(Counter(
                light for e in events for light in e["observations"]
            )),
        },
        "split": split,
        "warnings": data["warnings"],
        "train_event_ids": [e["id"] for e in training],
        "holdout_event_ids": [e["id"] for e in evaluation],
        "configuration": {
            "epochs": epochs, "batch_size": batch_size, "image_size": size,
            "seed": seed, "holdout": holdout, "device": device
        },
        "final_holdout": history[-1]["holdout"],
        "history": history,
    }
    torch.save({
        "schema": MODEL_SCHEMA, "state_dict": model.cpu().state_dict(),
        "image_size": size, "lights": LIGHTS,
        "experimental": True, "production_approved": False,
        "source_manifest_sha256": metrics["source_manifest_sha256"],
    }, out / "faltando_cnn_candidate.pt")
    (out / "training_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "training_summary.txt").write_text(
        "ODIN CNN FALTANDO — TREINAMENTO EXPERIMENTAL\n"
        + f"Frames: {metrics['counts']['frames']}\n"
        + f"Eventos: {metrics['counts']['events']}\n"
        + f"Treino (OK/NG): {split['train']}\n"
        + f"Holdout (OK/NG): {split['validation']}\n"
        + f"Holdout: {metrics['final_holdout']}\n"
        + "CNN NÃO CONECTADA À PRODUÇÃO.\n"
        + "Sem NG TOP/MID reais; não interpretar números como certificação.\n",
        encoding="utf-8"
    )
    return metrics, out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Treina a CNN FALTANDO localmente sem modificar produção."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=float, default=.20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    manifest = args.manifest or latest_manifest(args.root)
    print(f"Fonte: {manifest}", flush=True)
    report, out = train(
        manifest, epochs=args.epochs, batch_size=args.batch_size,
        size=args.size, seed=args.seed, holdout=args.holdout,
        device=args.device
    )
    print(f"Treino concluído; modelo candidato: {out / 'faltando_cnn_candidate.pt'}")
    print(f"Relatório: {out / 'training_report.json'}")
    print(f"Resumo: {out / 'training_summary.txt'}")
    print("O ODIN em produção não foi alterado.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
