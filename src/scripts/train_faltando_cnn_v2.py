"""Treinamento OFFLINE CNN FALTANDO v2: duas escalas e relatório por evento.

Uso:
    python -m src.scripts.train_faltando_cnn_v2 --epochs 25 --device cpu

Os pares são lidos do staging existente, sem exigir qualification.json.
A validação é de DESENVOLVIMENTO, não teste cego independente: o mesmo acervo
foi examinado na v1. Nenhum checkpoint é ativado no ODIN em produção.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, LIGHTS, MODEL_SCHEMA_V2
from src.scripts.train_faltando_cnn import load_events, split_events
from src.services.faltando_neural_qualification import latest_manifest

RUN_SCHEMA_V2 = "visionx.faltando_training_experiment.v2"


def _read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 12:
        raise ValueError(f"Recorte AOI indisponível ou inválido: {path}")
    return image


def _letterbox_rgb(image: np.ndarray, size: int) -> torch.Tensor:
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    target = (max(1, round(w * ratio)), max(1, round(h * ratio)))
    interpolation = cv2.INTER_AREA if ratio <= 1 else cv2.INTER_LINEAR
    frame = cv2.resize(image, target, interpolation=interpolation)
    canvas = np.full((size, size, 3), 127, dtype=np.uint8)
    y = (size - target[1]) // 2
    x = (size - target[0]) // 2
    canvas[y:y+target[1], x:x+target[0]] = frame
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.


def _focus_crop(image: np.ndarray, fraction: float) -> np.ndarray:
    """Zoom central: hipótese geométrica explícita, sem usar rótulo/ROI manual."""
    height, width = image.shape[:2]
    crop_h, crop_w = max(12, round(height * fraction)), max(12, round(width * fraction))
    cy, cx = height // 2, width // 2
    top = max(0, min(height - crop_h, cy - crop_h // 2))
    left = max(0, min(width - crop_w, cx - crop_w // 2))
    return image[top:top+crop_h, left:left+crop_w]


class DualScaleEventDataset(Dataset):
    def __init__(self, events: list[dict], data: dict, *, size: int = 160,
                 focus_fraction: float = .70, augment: bool = False):
        self.events, self.data, self.size = events, data, size
        self.focus_fraction, self.augment = focus_fraction, augment

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int):
        event = self.events[index]
        tensors = [torch.zeros((3, 3, self.size, self.size)) for _ in range(4)]
        lights = torch.zeros(3, dtype=torch.float32)
        for pos, light in enumerate(LIGHTS):
            if light not in event["observations"]:
                continue
            source = event["observations"][light]
            item = self.data["sample_map"][source]
            root = self.data["run_dir"]
            reference = _read_bgr(root / item["reference_path"])
            test = _read_bgr(root / item["test_path"])
            tensors[0][pos] = _letterbox_rgb(reference, self.size)
            tensors[1][pos] = _letterbox_rgb(test, self.size)
            tensors[2][pos] = _letterbox_rgb(
                _focus_crop(reference, self.focus_fraction), self.size
            )
            tensors[3][pos] = _letterbox_rgb(
                _focus_crop(test, self.focus_fraction), self.size
            )
            lights[pos] = 1.0

        if self.augment:
            # A MESMA transformação para gabarito/teste e para as duas escalas.
            if torch.rand(()) < .5:
                tensors = [t.flip(-1) for t in tensors]
            if torch.rand(()) < .2:
                tensors = [t.flip(-2) for t in tensors]
            gain = float(torch.empty(1).uniform_(.90, 1.10))
            bias = float(torch.empty(1).uniform_(-.03, .03))
            tensors = [(t * gain + bias).clamp(0., 1.) for t in tensors]

        return (
            *tensors, lights,
            torch.tensor(float(event["label"] == "NG")),
            event["id"],
        )


def _metrics(records: list[dict]) -> dict:
    tp = sum(x["label"] == "NG" and x["predicted"] == "NG" for x in records)
    tn = sum(x["label"] == "OK" and x["predicted"] == "OK" for x in records)
    fp = sum(x["label"] == "OK" and x["predicted"] == "NG" for x in records)
    fn = sum(x["label"] == "NG" and x["predicted"] == "OK" for x in records)
    count = len(records)
    return {
        "cases": count, "TP_NG": tp, "TN_OK": tn,
        "FP_OK_as_NG": fp, "FN_NG_as_OK": fn,
        "ng_recall": round(tp/(tp+fn), 4) if tp+fn else None,
        "ok_specificity": round(tn/(tn+fp), 4) if tn+fp else None,
        "balanced_accuracy": (
            round((tp/(tp+fn) + tn/(tn+fp))/2, 4)
            if (tp+fn) and (tn+fp) else None
        ),
        "accuracy": round((tp+tn)/count, 4) if count else None,
    }


def _evaluate(model: FaltandoCNNV2, loader: DataLoader,
              device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    rows: list[dict] = []
    total_loss = 0.
    with torch.no_grad():
        for full_ref, full_test, focus_ref, focus_test, mask, truth, event_ids in loader:
            tensors = [t.to(device) for t in
                       (full_ref, full_test, focus_ref, focus_test, mask)]
            logits, per_light = model(*tensors)
            batch_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, truth.to(device), reduction="sum"
            )
            total_loss += float(batch_loss.item())
            scores = torch.sigmoid(logits).cpu().tolist()
            each = torch.sigmoid(per_light).cpu().tolist()
            for j, eid in enumerate(event_ids):
                observations = {
                    light: round(each[j][k], 6)
                    for k, light in enumerate(LIGHTS) if mask[j, k] > 0
                }
                actual = "NG" if float(truth[j]) >= .5 else "OK"
                predicted = "NG" if scores[j] >= .5 else "OK"
                rows.append({
                    "event_id_or_group_hint": eid,
                    "label": actual,
                    "predicted": predicted,
                    "ng_score": round(scores[j], 6),
                    "per_light_ng_scores": observations,
                    "correct": predicted == actual,
                    "error": (
                        "FN_NG_AS_OK" if actual == "NG" and predicted == "OK"
                        else "FP_OK_AS_NG" if actual == "OK" and predicted == "NG"
                        else None
                    ),
                })
    metrics = _metrics(rows)
    metrics["loss"] = round(total_loss/max(len(rows), 1), 6)
    return metrics, rows


def train_v2(
    manifest: Path, *, epochs: int = 25, batch_size: int = 4,
    size: int = 160, focus_fraction: float = .70,
    seed: int = 42, holdout: float = .20,
    patience: int = 7, device: str = "cpu",
) -> tuple[dict, Path]:
    if not (1 <= epochs <= 500 and 1 <= batch_size <= 128 and 2 <= patience <= 100):
        raise ValueError("epochs, batch_size ou patience inválido")
    if size % 32 or not 64 <= size <= 512:
        raise ValueError("size deve ser múltiplo de 32 entre 64 e 512")
    if not .5 <= focus_fraction <= .90:
        raise ValueError("focus_fraction deve estar entre 0.5 e 0.9")
    if not .1 <= holdout <= .4:
        raise ValueError("holdout deve estar entre 0.1 e 0.4")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA solicitado, mas indisponível")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    events, data = load_events(manifest)
    train_indices, validation_indices, split = split_events(
        events, data, seed=seed, holdout=holdout
    )
    train_events = [events[i] for i in train_indices]
    validation_events = [events[i] for i in validation_indices]
    train_data = DualScaleEventDataset(
        train_events, data, size=size, focus_fraction=focus_fraction, augment=True
    )
    validation_data = DualScaleEventDataset(
        validation_events, data, size=size, focus_fraction=focus_fraction
    )
    positives = sum(x["label"] == "NG" for x in train_events)
    negatives = len(train_events) - positives
    if positives < 2 or negatives < 2:
        raise ValueError("Treino sem NG/OK suficientes")
    # Balancear amostragem em vez de duplicar também pos_weight:
    # evita penalizar NG duplamente e ignorar todos os OK.
    sample_weights = [
        1. / (positives if x["label"] == "NG" else negatives)
        for x in train_events
    ]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_events), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              sampler=sampler, num_workers=0)
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, num_workers=0
    )

    torch_device = torch.device(device)
    model = FaltandoCNNV2().to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.05)
    # Mesmo acervo analisado na v1: dev validation, não teste cego.
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    best_rows = []
    best_metrics = {}
    wait = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.
        for full_ref, full_test, focus_ref, focus_test, mask, truth, _ in train_loader:
            inputs = [
                t.to(torch_device)
                for t in (full_ref, full_test, focus_ref, focus_test, mask)
            ]
            truth = truth.to(torch_device)
            logits, per_light = model(*inputs)
            primary = nn.functional.binary_cross_entropy_with_logits(logits, truth)
            auxiliary = nn.functional.binary_cross_entropy_with_logits(
                per_light, truth[:, None].expand_as(per_light),
                reduction="none"
            )
            auxiliary = (auxiliary * inputs[-1]).sum() / inputs[-1].sum().clamp_min(1)
            loss = primary + .15 * auxiliary
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.)
            optimizer.step()
            loss_sum += float(loss.item()) * len(truth)

        metrics, rows = _evaluate(model, validation_loader, torch_device)
        entry = {
            "epoch": epoch, "training_loss": round(loss_sum/len(train_events), 6),
            "dev_validation": metrics,
        }
        history.append(entry)
        improved = metrics["loss"] < best_loss - .0001
        if improved:
            best_loss = metrics["loss"]
            best_epoch = epoch
            best_state = {
                name: weight.detach().cpu().clone()
                for name, weight in model.state_dict().items()
            }
            best_metrics = metrics
            best_rows = rows
            wait = 0
        else:
            wait += 1
        print(
            f"[v2 {epoch:03d}/{epochs}] train_loss={entry['training_loss']:.4f} "
            f"dev_loss={metrics['loss']:.4f} "
            f"NG_detectados={metrics['TP_NG']} "
            f"NG_liberados={metrics['FN_NG_as_OK']} "
            f"OK_falsos_NG={metrics['FP_OK_as_NG']}",
            flush=True,
        )
        if wait >= patience:
            print(f"Early stopping: {patience} épocas sem melhoria na perda de desenvolvimento.")
            break

    if best_state is None or best_epoch is None:
        raise RuntimeError("Treinamento não produziu pesos válidos")
    when = datetime.now(timezone.utc)
    output = data["root"] / "reports" / "faltando_neural" / "models" / (
        "experiment_v2_" + when.strftime("%Y%m%dT%H%M%S_%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    source_hash = sha256(Path(manifest).read_bytes()).hexdigest()
    checkpoint = {
        "schema": MODEL_SCHEMA_V2, "state_dict": best_state,
        "image_size": size, "focus_fraction": focus_fraction,
        "lights": LIGHTS, "best_epoch": best_epoch,
        "source_manifest_sha256": source_hash,
        "experimental": True, "production_approved": False,
    }
    torch.save(checkpoint, output / "faltando_cnn_v2_candidate.pt")

    report = {
        "schema": RUN_SCHEMA_V2,
        "model_schema": MODEL_SCHEMA_V2,
        "trained_at_utc": when.isoformat(),
        "source_manifest": str(Path(manifest).resolve()),
        "source_manifest_sha256": source_hash,
        "experimental": True, "production_approved": False,
        "knn_used": False,
        "training_only": True,
        "model_version": 2,
        "images": len(data["sample_map"]),
        "events": len(events),
        "class_counts_events": dict(Counter(x["label"] for x in events)),
        "split": split,
        "training_event_ids": [x["id"] for x in train_events],
        "development_event_ids": [x["id"] for x in validation_events],
        "evaluation_limitations": [
            "DESENVOLVIMENTO: holdout reutiliza o acervo já observado na v1; não é teste cego independente.",
            "Apenas 10 NG SIDE históricos; zero NG TOP/MID reais.",
            "Agrupamento multilight por nome+OCR não comprova event_id.",
            "Crop de foco central pressupõe componente aproximadamente centralizado.",
            "Precisão/recall no conjunto de 13 eventos não certifica Produção.",
        ],
        "training_parameters": {
            "max_epochs": epochs, "completed_epochs": len(history),
            "best_epoch": best_epoch, "batch_size": batch_size,
            "image_size": size, "focus_fraction": focus_fraction,
            "seed": seed, "holdout": holdout, "patience": patience,
            "device": device,
            "balanced_sampler": True,
            "loss": "BCE_event+0.15_BCE_masked_per_light",
        },
        "selection_policy": (
            "melhor época por BCE em validação de desenvolvimento; "
            "não representa teste cego nem calibração"
        ),
        "best_dev_validation": best_metrics,
        "final_epoch_dev_validation": history[-1]["dev_validation"],
        "per_case_dev_predictions": best_rows,
        "errors": [x for x in best_rows if not x["correct"]],
        "history": history,
    }
    (output / "training_report_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "holdout_predictions_v2.json").write_text(
        json.dumps(best_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "training_summary_v2.txt").write_text(
        "ODIN CNN FALTANDO v2 — TREINO EXPERIMENTAL OFFLINE\n"
        f"Imagens: {report['images']} | eventos: {report['events']}\n"
        f"Split (treino): {split['train']} | desenvolvimento: {split['validation']}\n"
        f"Melhor época: {best_epoch} / {len(history)}\n"
        f"Métricas em DESENVOLVIMENTO: {best_metrics}\n"
        "IMPORTANTE: conjunto não é teste cego; a v1 já o examinou.\n"
        "NG TOP/MID reais: zero.\n"
        "A CNN NÃO ESTÁ CONECTADA À PRODUÇÃO.\n",
        encoding="utf-8"
    )
    return report, output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Treinar CNN FALTANDO v2 offline.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--focus-fraction", type=float, default=.70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout", type=float, default=.20)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args(argv)
    manifest = arguments.manifest or latest_manifest(arguments.root)
    print("Manifesto:", manifest, flush=True)
    report, path = train_v2(
        manifest, epochs=arguments.epochs, batch_size=arguments.batch_size,
        size=arguments.size, focus_fraction=arguments.focus_fraction,
        seed=arguments.seed, holdout=arguments.holdout,
        patience=arguments.patience, device=arguments.device,
    )
    print(f"Pesos candidatos: {path / 'faltando_cnn_v2_candidate.pt'}")
    print(f"Relatório: {path / 'training_report_v2.json'}")
    print(f"Previsões por evento: {path / 'holdout_predictions_v2.json'}")
    print(f"Resumo: {path / 'training_summary_v2.txt'}")
    print("O ODIN em produção não foi alterado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
