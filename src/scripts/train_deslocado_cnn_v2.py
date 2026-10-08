"""DESLOCADO CNN v2 — treino offline sobre OK e movimento de máscara de componente.

Correção da v1: não desloca um patch retangular, e usa reconstrução do
fundo tanto em OK sintético quanto em pseudo-NG. Mantém REAL_OK intacto
na validação. NÃO existe NG real no arquivo: não autoriza Produção.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.core.neural.deslocado_cnn import DeslocadoCNN
from src.core.neural.faltando_cnn_v2 import LIGHTS
from src.scripts.train_deslocado_cnn import load_ok_events, _safe_png, _read
from src.scripts.train_faltando_cnn_v2 import _focus_crop, _letterbox_rgb
from src.services.deslocado_neural_dataset import latest_deslocado_manifest
from src.services.deslocado_proxy_v2 import (
    PROXY_VERSION, paired_photometric, simulate_component_shift,
)

SCHEMA = "visionx.deslocado_training_experiment.v2"
MODEL_SCHEMA = "visionx.deslocado_comparative_cnn.v2"
KINDS = ("REAL_OK", "RECOMPOSED_OK", "SYNTHETIC_SHIFT_PROXY")


class DeslocadoV2Dataset(Dataset):
    """Eventos separados; cada ROI multilight permanece indivisível.

    Só gera proxy se o componente for localizável em TODAS as luzes do
    evento. Se a máscara for ambígua, conserva apenas o REAL_OK.
    """

    def __init__(self, events: list[dict], data: dict, *,
                 size: int = 160, training: bool = False,
                 proxy_variants: int = 3, focus_fraction: float = .70):
        self.size, self.training = size, training
        self.focus_fraction = focus_fraction
        self.items: list[dict] = []
        self.instances: list[tuple[int,str,int]] = []
        self.unresolved: list[dict] = []
        self.per_light_unresolved = Counter()
        for event in events:
            if event.get("label", "OK") != "OK":
                # v2 ainda é bootstrap OK-only; não reclassificar NG humano.
                raise ValueError("DESLOCADO v2 bootstrap espera apenas OK reais")
            views = {}
            for light, source in event["observations"].items():
                item = data["samples"][source]
                ref = _read(_safe_png(data["run"], item["reference_path"]))
                tst = _read(_safe_png(data["run"], item["test_path"]))
                views[light] = (ref, tst)
            obj = {"id": event["id"], "views": views}
            entry = len(self.items)
            self.items.append(obj)
            self.instances.append((entry, "REAL_OK", 0))
            # Mesma direção de deslocamento em todas as luzes do evento.
            available = True
            for variant in range(proxy_variants):
                seed = int(sha256(
                    (event["id"] + "|proxy_v2|" + str(variant)).encode()
                ).hexdigest()[:8], 16)
                proxies = {
                    light: simulate_component_shift(tst, seed)
                    for light, (_, tst) in views.items()
                }
                missing = [light for light, proxy in proxies.items()
                           if proxy is None]
                if missing:
                    for light in missing:
                        self.per_light_unresolved[light] += 1
                    available = False
                    continue
                obj.setdefault("proxies", {})[variant] = proxies
                self.instances.append((entry, "RECOMPOSED_OK", variant))
                self.instances.append((entry, "SYNTHETIC_SHIFT_PROXY", variant))
            if not available:
                self.unresolved.append({
                    "event_id": event["id"],
                    "reason": "AMBIGUOUS_OR_MISSING_COMPONENT_MASK",
                    "lights": list(views),
                })

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index: int):
        entry, kind, variant = self.instances[index]
        event = self.items[entry]
        pictures = [torch.zeros((3, 3, self.size, self.size))
                    for _ in range(4)]
        mask = torch.zeros(3, dtype=torch.float32)
        for pos, light in enumerate(LIGHTS):
            if light not in event["views"]:
                continue
            ref, real = event["views"][light]
            if kind == "REAL_OK":
                test = real
            else:
                proxy = event["proxies"][variant][light]
                test = (proxy.normal if kind == "RECOMPOSED_OK"
                        else proxy.shifted)
            if self.training:
                # A mesma distribuição de iluminação em OK e proxy:
                # não introduz uma pista do rótulo.
                seed = int(np.random.randint(0, 2**31-1))
                ref, test = paired_photometric(
                    ref, test, seed, independent=True
                )
            frames = (
                ref, test,
                _focus_crop(ref, self.focus_fraction),
                _focus_crop(test, self.focus_fraction),
            )
            for j, frame in enumerate(frames):
                pictures[j][pos] = _letterbox_rgb(frame, self.size)
            mask[pos] = 1.
        if self.training and bool(torch.rand(()) < .5):
            pictures = [t.flip(-1) for t in pictures]
        return (
            *pictures, mask,
            torch.tensor(float(kind == "SYNTHETIC_SHIFT_PROXY")),
            event["id"], kind,
        )

    def counts(self) -> dict:
        return dict(Counter(kind for _,kind,_ in self.instances))


def _measure(model: DeslocadoCNN, dataset: DeslocadoV2Dataset) -> dict:
    model.eval()
    rows = []
    total_loss = 0.
    loader = DataLoader(dataset, batch_size=4, num_workers=0)
    with torch.inference_mode():
        for ref, test, focus_ref, focus_test, mask, truth, ids, kinds in loader:
            logits, per_light = model(ref, test, focus_ref, focus_test, mask)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, truth, reduction="sum"
            )
            total_loss += float(loss)
            scores = torch.sigmoid(logits).tolist()
            light_scores = torch.sigmoid(per_light).tolist()
            for row_idx, (key, kind) in enumerate(zip(ids, kinds)):
                positive = kind == "SYNTHETIC_SHIFT_PROXY"
                predicted = scores[row_idx] >= .5
                by_light = {
                    light: {
                        "score": round(light_scores[row_idx][i], 6),
                        "predicted_proxy": light_scores[row_idx][i] >= .5,
                        "correct": (light_scores[row_idx][i] >= .5) == positive,
                    }
                    for i, light in enumerate(LIGHTS) if mask[row_idx, i] > 0
                }
                rows.append({
                    "event_id": key, "truth_type": kind,
                    "score_proxy_ng_uncalibrated": round(scores[row_idx], 6),
                    "predicted_proxy_ng": predicted,
                    "correct": predicted == positive,
                    "by_light": by_light,
                })
    totals = Counter(x["truth_type"] for x in rows)
    correct = Counter(
        x["truth_type"] for x in rows if x["correct"]
    )
    ok_rows = [x for x in rows if x["truth_type"] == "REAL_OK"]
    proxy_rows = [x for x in rows if x["truth_type"] == "SYNTHETIC_SHIFT_PROXY"]
    return {
        "total": len(rows),
        "loss": round(total_loss/max(len(rows), 1), 6),
        "real_ok": len(ok_rows),
        "real_ok_correct": correct["REAL_OK"],
        "real_ok_false_ng": len(ok_rows)-correct["REAL_OK"],
        "recomposed_ok": totals["RECOMPOSED_OK"],
        "recomposed_ok_correct": correct["RECOMPOSED_OK"],
        "synthetic_proxy": len(proxy_rows),
        "synthetic_proxy_correct": correct["SYNTHETIC_SHIFT_PROXY"],
        "real_ng": 0,
        "real_ng_recall": None,
        "zero_false_ng_on_real_ok": (
            bool(ok_rows) and len(ok_rows) == correct["REAL_OK"]
        ),
        "real_ok_per_light": {
            light: {
                "tested": sum(light in x["by_light"] for x in ok_rows),
                "correct": sum(
                    x["by_light"][light]["correct"]
                    for x in ok_rows if light in x["by_light"]
                ),
            }
            for light in LIGHTS
        },
        "predictions": rows,
    }


def train_deslocado_v2(
    manifest_path: Path, *, epochs: int = 25, size: int = 160,
    batch_size: int = 4, seed: int = 42, patience: int = 8,
    proxy_variants: int = 3,
) -> tuple[dict, Path]:
    if not 1 <= epochs <= 200 or not 1 <= batch_size <= 64:
        raise ValueError("epochs ou batch inválidos")
    if size < 64 or size > 512 or size % 32:
        raise ValueError("Tamanho deve ser múltiplo de 32 entre 64 e 512")
    if not 2 <= patience <= 100 or not 1 <= proxy_variants <= 8:
        raise ValueError("Paciência ou variantes inválidas")
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    events, data = load_ok_events(manifest_path)
    groups = defaultdict(list)
    for e in events:
        groups[e["split_key"]].append(e)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 3:
        raise ValueError(
            "São necessários 3 ou mais grupos de placa/componente "
            "para uma validação de desenvolvimento isolada."
        )
    n_holdout = min(len(keys)-1, max(1, round(.20*len(keys))))
    val_keys = set(keys[:n_holdout])
    training = [e for e in events if e["split_key"] not in val_keys]
    validation = [e for e in events if e["split_key"] in val_keys]
    assert not (
        {e["split_key"] for e in training}
        & {e["split_key"] for e in validation}
    )
    train_data = DeslocadoV2Dataset(
        training, data, size=size, training=True,
        proxy_variants=proxy_variants
    )
    dev_data = DeslocadoV2Dataset(
        validation, data, size=size, training=False,
        proxy_variants=1,
    )
    if train_data.counts().get("SYNTHETIC_SHIFT_PROXY", 0) == 0:
        raise ValueError(
            "Nenhum componente segmentável no treino: não criar NG sintéticos "
            "de regiões incertas. Revise o ROI ou forneça máscaras manuais."
        )
    if dev_data.counts().get("SYNTHETIC_SHIFT_PROXY", 0) == 0:
        raise ValueError(
            "Validação sem proxy de componente confiável. "
            "Não é possível medir discriminação sintética."
        )

    # Balancear ligeiramente para OK. Os OK reconstruídos contêm os
    # mesmos artefatos de inpaint que as versões deslocadas.
    class_freq = Counter(
        kind == "SYNTHETIC_SHIFT_PROXY" for _, kind, _ in train_data.instances
    )
    weights = [
        (1.25 if kind != "SYNTHETIC_SHIFT_PROXY" else 1.0)
        / class_freq[kind == "SYNTHETIC_SHIFT_PROXY"]
        for _,kind,_ in train_data.instances
    ]
    sampler = WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(train_data), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        train_data, batch_size=batch_size, sampler=sampler, num_workers=0
    )
    model = DeslocadoCNN()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=.00025, weight_decay=.06
    )
    best = None
    history = []
    wait = 0
    for epoch in range(1, epochs+1):
        model.train()
        loss_sum = 0.
        for ref, test, fr, ft, mask, truth, _, _ in loader:
            logits, lights = model(ref, test, fr, ft, mask)
            primary = nn.functional.binary_cross_entropy_with_logits(
                logits, truth
            )
            auxiliary = nn.functional.binary_cross_entropy_with_logits(
                lights, truth[:, None].expand_as(lights), reduction="none"
            )
            loss = primary + .12*(
                auxiliary*mask
            ).sum()/mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optimizer.step()
            loss_sum += float(loss)*len(truth)
        metrics = _measure(model, dev_data)
        # Prioridade industrial: zero FP nos OK reais do holdout.
        # Depois acerto nos deslocamentos PROXY; só então loss.
        priority = (
            metrics["real_ok_false_ng"],
            metrics["recomposed_ok"]-metrics["recomposed_ok_correct"],
            metrics["synthetic_proxy"]-metrics["synthetic_proxy_correct"],
            metrics["loss"],
        )
        previous_loss = best[0][3] if best else float("inf")
        if best is None or priority < best[0]:
            best = (
                priority, epoch,
                {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                metrics,
            )
            wait = 0
        else:
            wait += 1
        history.append({
            "epoch": epoch,
            "train_loss": round(loss_sum/len(train_data), 6),
            "dev_real_ok_false_ng": metrics["real_ok_false_ng"],
            "dev_synthetic_proxy_correct": metrics["synthetic_proxy_correct"],
            "dev_synthetic_proxy_total": metrics["synthetic_proxy"],
            "dev_loss": metrics["loss"],
        })
        print(
            f"[DESLOCADO v2 {epoch:03d}/{epochs}] "
            f"loss={history[-1]['train_loss']:.5f} | "
            f"dev OK falsos NG={metrics['real_ok_false_ng']}/"
            f"{metrics['real_ok']} | proxy="
            f"{metrics['synthetic_proxy_correct']}/{metrics['synthetic_proxy']}",
            flush=True
        )
        if wait >= patience:
            print("Parada antecipada do desenvolvimento.", flush=True)
            break

    assert best is not None
    priority, best_epoch, best_state, dev = best
    model.load_state_dict(best_state, strict=True)
    training_metrics = _measure(model, DeslocadoV2Dataset(
        training, data, size=size, proxy_variants=1
    ))
    gate = bool(dev["zero_false_ng_on_real_ok"] and
                dev["recomposed_ok_correct"] == dev["recomposed_ok"] and
                dev["synthetic_proxy_correct"] == dev["synthetic_proxy"])
    now = datetime.now(timezone.utc)
    output = data["root"]/"reports"/"deslocado_neural"/"models"/(
        "experiment_v2_"+now.strftime("%Y%m%dT%H%M%S_%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest_hash = sha256(Path(manifest_path).read_bytes()).hexdigest()
    ckpt = {
        "schema": MODEL_SCHEMA,
        "state_dict": best_state,
        "lights": LIGHTS,
        "image_size": size,
        "focus_fraction": .70,
        "source_manifest_sha256": manifest_hash,
        "best_epoch": best_epoch,
        "proxy_generator": PROXY_VERSION,
        "trained_only_with_real_ok_and_synthetic_proxy": True,
        "real_ng_used": 0,
        "experimental": True,
        "production_approved": False,
        "allow_automatic_classification": False,
        "development_ok_gate_passed": gate,
    }
    torch.save(ckpt, output/"deslocado_cnn_v2_candidate.pt")
    report = {
        "schema": SCHEMA,
        "category": "DESLOCADO",
        "source_manifest": str(Path(manifest_path).resolve()),
        "source_manifest_sha256": manifest_hash,
        "model_schema": MODEL_SCHEMA,
        "real_ok_images": len(data["samples"]),
        "real_ok_events": len(events),
        "real_ng_count": 0,
        "real_ng_recall": None,
        "train_events": len(training),
        "development_events": len(validation),
        "split_strategy": "same_board_and_component_group",
        "train_event_ids": [x["id"] for x in training],
        "development_event_ids": [x["id"] for x in validation],
        "proxy_generator": PROXY_VERSION,
        "proxy_segmentation_unverified": True,
        "training_proxy_counts": train_data.counts(),
        "development_proxy_counts": dev_data.counts(),
        "train_proxy_unresolved": train_data.unresolved,
        "development_proxy_unresolved": dev_data.unresolved,
        "dev_real_ok_zero_false_ng_gate_passed": dev["zero_false_ng_on_real_ok"],
        "development_combined_gate_passed": gate,
        "production_approved": False,
        "activation_disabled": True,
        "best_epoch": best_epoch,
        "configuration": {
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "batch_size": batch_size, "image_size": size,
            "seed": seed, "patience": patience,
            "proxy_variants_train": proxy_variants,
        },
        "development": dev,
        "training": training_metrics,
        "history": history,
        "limitations": [
            "Nenhum defeito DESLOCADO real foi usado ou avaliado.",
            "Máscara do componente é hipótese visual não validada.",
            "Reconstrução e deslocamento sintéticos não provam defeito industrial real.",
            "O conjunto de desenvolvimento já foi usado para escolher época/modelo.",
            "Acerto 100% em OK reais não demonstra recall de NG real.",
            "Não integrar CNN DESLOCADO à produção; preservar especialistas físicos.",
        ],
    }
    (output/"training_report_deslocado_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output/"holdout_predictions_deslocado_v2.json").write_text(
        json.dumps(dev["predictions"], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (output/"training_summary_deslocado_v2.txt").write_text(
        "ODIN CNN DESLOCADO v2 — apenas candidato OFFLINE\n"
        f"OK reais: {len(events)} eventos, NG reais: ZERO.\n"
        f"Treino: {len(training)} eventos; dev: {len(validation)} eventos.\n"
        f"Melhor época: {best_epoch}; executadas: {len(history)}.\n"
        f"OK verdadeiros no holdout: {dev['real_ok_correct']}/{dev['real_ok']}.\n"
        f"OK reconstruídos no holdout: "
        f"{dev['recomposed_ok_correct']}/{dev['recomposed_ok']}.\n"
        f"Proxies sintéticos: {dev['synthetic_proxy_correct']}/{dev['synthetic_proxy']}.\n"
        f"Gate desenvolvimento OK: {'PASSOU' if dev['zero_false_ng_on_real_ok'] else 'REPROVOU'}.\n"
        f"Gate combinado: {'PASSOU' if gate else 'REPROVOU'}.\n"
        "RECALL DESLOCADO NG REAL: NÃO MENSURÁVEL.\n"
        "CHECKPOINT NÃO ATIVADO; MOTORES FÍSICOS MANTIDOS.\n",
        encoding="utf-8"
    )
    return report, output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Treinar CNN DESLOCADO v2 em OK e proxy de componente"
    )
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--proxy-variants", type=int, default=3)
    args = parser.parse_args(argv)
    manifest = args.manifest or latest_deslocado_manifest(args.root)
    report, output = train_deslocado_v2(
        manifest, epochs=args.epochs, size=args.size,
        batch_size=args.batch_size, patience=args.patience,
        proxy_variants=args.proxy_variants,
    )
    print("Relatório:", output/"training_report_deslocado_v2.json")
    print("Resumo:", output/"training_summary_deslocado_v2.txt")
    print("Previsões:", output/"holdout_predictions_deslocado_v2.json")
    print("CNN DESLOCADO permanece INATIVA em produção.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
