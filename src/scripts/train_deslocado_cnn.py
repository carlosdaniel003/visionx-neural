"""Treino inicial especializado DESLOCADO a partir de OK + proxies SINTÉTICOS.

NÃO há exemplos reais NG nesta fase, portanto nenhum resultado, score
ou checkpoint deste protótipo é aprovado para julgar Produção.

Usar no Win10:
 python -m src.services.deslocado_neural_dataset
 python -m src.scripts.train_deslocado_cnn --epochs 15
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.core.neural.deslocado_cnn import DeslocadoCNN, MODEL_SCHEMA_DESLOCADO
from src.core.neural.faltando_cnn_v2 import LIGHTS
from src.scripts.train_faltando_cnn_v2 import _focus_crop, _letterbox_rgb
from src.services.deslocado_neural_dataset import (
    SCHEMA as DATASET_SCHEMA, latest_deslocado_manifest,
)

TRAIN_SCHEMA = "visionx.deslocado_training_experiment.v1"


def _safe_png(parent: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Caminho inválido")
    file = (parent/relative).resolve()
    if file.suffix.lower() != ".png" or not file.is_file() or (
        parent.resolve()/"pairs"
    ) not in file.parents or file.is_symlink():
        raise ValueError("Par extraído ausente/fora de pairs")
    return file


def _read(path: Path) -> np.ndarray:
    frame = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None or frame.ndim != 3 or min(frame.shape[:2]) < 12:
        raise ValueError("Imagem gabarito/teste inválida")
    return frame


def load_ok_events(manifest_file: Path) -> tuple[list[dict], dict]:
    """Confere todas as fontes originais e NÃO confunde trincas com 3 eventos."""
    manifest_file = Path(manifest_file).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("Manifesto de DESLOCADO incompatível")
    run = manifest_file.parent
    if run.parent.name != "deslocado_neural" or run.parent.parent.name != "reports":
        raise ValueError("Manifests DESLOCADO apenas em reports/deslocado_neural/run_*")
    root = Path(manifest["root"]).resolve()
    if run.parent.parent.resolve() != (root/"reports").resolve():
        raise ValueError("Raiz do relatório difere do repositório")
    samples = {}
    for sample in manifest.get("samples", []):
        if sample.get("status") != "EXTRACTED_PENDING_REVIEW":
            raise ValueError("Recorte DESLOCADO com falha: treino interrompido")
        if sample.get("category_hint") != "DESLOCADO":
            raise ValueError("Mistura de categorias no manifesto")
        label = sample.get("expected_label_from_archive")
        if label == "NG":
            raise ValueError(
                "Já existem NG DESLOCADO reais: use protocolo supervisionado "
                "e holdout de NG reais; não executar bootstrap OK-only."
            )
        if label != "OK" or sample.get("lighting_mode") not in LIGHTS:
            raise ValueError("Classe/iluminação inválida")
        rel = sample["source_path"]
        if rel in samples:
            raise ValueError("Fonte duplicada")
        source = (root/rel).resolve()
        if (root/"public"/"ok_archive").resolve() not in source.parents:
            raise ValueError("Origem fora de public/ok_archive")
        if sha256(source.read_bytes()).hexdigest() != sample["source_sha256"]:
            raise ValueError("PNG original alterado desde a preparação")
        for key in ("reference_path", "test_path"):
            _safe_png(run, sample[key])
        samples[rel] = sample
    if not samples:
        raise ValueError("Acervo DESLOCADO não possui OK preparado")

    events = []
    grouped = set()
    warnings = []
    for triplet in manifest.get("name_only_triplet_candidates", []):
        paths = triplet.get("paths", {})
        if (set(paths) != set(LIGHTS)
                or len(set(paths.values())) != 3
                or any(p not in samples or p in grouped for p in paths.values())):
            warnings.append("Trinca não agrupada: "+str(triplet.get("id")))
            continue
        rows = [samples[paths[mode]] for mode in LIGHTS]
        signatures = {
            tuple("".join(ch for ch in str(s.get("ocr_observed", {}).get(k, "")).upper()
                          if ch.isalnum()) for k in ("board", "parts", "value"))
            for s in rows
        }
        if len(signatures) != 1 or not all(next(iter(signatures))[:2]):
            warnings.append("Trinca OCR inconsistente: "+str(triplet.get("id")))
            continue
        grouped.update(paths.values())
        events.append({
            "id": "name_and_ocr:"+str(triplet["id"]),
            "observations": dict(paths),
            "association": "UNVERIFIED_NAME_OCR_CANDIDATE",
        })
    for rel, sample in sorted(samples.items()):
        if rel not in grouped:
            events.append({
                "id": "single:"+rel,
                "observations": {sample["lighting_mode"]: rel},
                "association": "SINGLE_FRAME",
            })
    for event in events:
        data = samples[next(iter(event["observations"].values()))]
        ocr = data.get("ocr_observed", {})
        board = "".join(x for x in str(ocr.get("board", "")).upper() if x.isalnum())
        part = "".join(x for x in str(ocr.get("parts", "")).upper() if x.isalnum())
        event["split_key"] = (
            "component:"+board+"/"+part if board and part else event["id"]
        )
    if sum(len(e["observations"]) for e in events) != len(samples):
        raise ValueError("Evento mal agrupado / amostra omitida")
    return events, {"root": root, "run": run, "samples": samples,
                    "warnings": warnings, "manifest": manifest}


def _proxy_displace(test: np.ndarray, seed: int) -> np.ndarray:
    """Move suavemente PATCH central. NÃO representa defeito NG real da AOI.

    É apenas tarefa proxy para inicializar uma CNN comparativa:
    nunca interpretar suas métricas como recall de DESLOCADO verdadeiro.
    """
    rng = np.random.default_rng(seed)
    h, w = test.shape[:2]
    y0, y1 = int(h*.18), int(h*.82)
    x0, x1 = int(w*.18), int(w*.82)
    roi = test[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    dx = max(2, int(w*.06)) * (1 if rng.integers(0, 2) else -1)
    dy = max(2, int(h*.05)) * (1 if rng.integers(0, 2) else -1)
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        roi, matrix, (rw, rh), borderMode=cv2.BORDER_REFLECT_101
    )
    # Feathered mask: reduz a borda artificial do mosaico; ainda é proxy.
    distance_y = np.minimum(np.arange(rh), np.arange(rh)[::-1])[:, None]
    distance_x = np.minimum(np.arange(rw), np.arange(rw)[::-1])[None, :]
    border = max(2, min(rh, rw)//10)
    alpha = np.minimum(1, np.minimum(distance_y, distance_x)/border)
    alpha = (alpha*.94).astype(np.float32)[..., None]
    result = test.copy()
    result[y0:y1, x0:x1] = np.clip(
        roi.astype(np.float32)*(1-alpha) +
        shifted.astype(np.float32)*alpha, 0, 255
    ).astype(np.uint8)
    return result


class DeslocadoProxyDataset(Dataset):
    def __init__(self, events: list[dict], data: dict, size=160, focus=.70):
        self.events, self.data, self.size, self.focus = events, data, size, focus

    def __len__(self):
        return len(self.events)*2

    def __getitem__(self, index: int):
        event = self.events[index//2]
        proxy = bool(index % 2)
        images = [torch.zeros((3, 3, self.size, self.size)) for _ in range(4)]
        mask = torch.zeros(3, dtype=torch.float32)
        for position, light in enumerate(LIGHTS):
            if light not in event["observations"]:
                continue
            item = self.data["samples"][event["observations"][light]]
            reference = _read(_safe_png(self.data["run"], item["reference_path"]))
            test = _read(_safe_png(self.data["run"], item["test_path"]))
            if proxy:
                key = int(sha256(
                    (event["id"]+":"+light).encode()
                ).hexdigest()[:8], 16)
                test = _proxy_displace(test, key)
            frames = (
                reference, test,
                _focus_crop(reference, self.focus),
                _focus_crop(test, self.focus),
            )
            for j, frame in enumerate(frames):
                images[j][position] = _letterbox_rgb(frame, self.size)
            mask[position] = 1.
        return *images, mask, torch.tensor(float(proxy)), event["id"]


def _evaluate(net: DeslocadoCNN, events: list[dict],
              data: dict, size: int) -> dict:
    net.eval()
    records = []
    loader = DataLoader(
        DeslocadoProxyDataset(events, data, size),
        batch_size=4, num_workers=0
    )
    with torch.inference_mode():
        for a, b, c, d, mask, actual, names in loader:
            logits, lights = net(a,b,c,d,mask)
            for j, name in enumerate(names):
                score = float(torch.sigmoid(logits[j]))
                records.append({
                    "event": name,
                    "truth_type": "SYNTHETIC_PROXY_SHIFT" if actual[j] else "REAL_OK",
                    "ng_proxy_score": round(score, 6),
                    "predicted_proxy": score >= .5,
                    "correct_proxy": (score >= .5) == bool(actual[j]),
                })
    ok = [r for r in records if r["truth_type"] == "REAL_OK"]
    proxy = [r for r in records if r["truth_type"] != "REAL_OK"]
    return {
        "real_ok": len(ok),
        "synthetic_proxy": len(proxy),
        "ok_correct": sum(not r["predicted_proxy"] for r in ok),
        "synthetic_correct": sum(r["predicted_proxy"] for r in proxy),
        "real_ng_count": 0,
        "real_ng_recall": None,
        "real_ng_validation_available": False,
        "predictions": records,
    }


def train_deslocado(
    manifest_path: Path, *, epochs: int = 15,
    size: int = 160, batch_size: int = 4, seed: int = 42,
) -> tuple[dict, Path]:
    if not 1 <= epochs <= 100 or not 1 <= batch_size <= 64:
        raise ValueError("Parâmetros de treino inválidos")
    if not 64 <= size <= 512 or size % 32:
        raise ValueError("Tamanho de entrada inválido")
    events, data = load_ok_events(manifest_path)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = DeslocadoCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.04)
    # As trincas já foram convertidas em um evento. Não misturar uma
    # mesma placa/componente em treino e desenvolvimento.
    groups = defaultdict(list)
    for event in events:
        groups[event["split_key"]].append(event)
    unique = sorted(groups)
    random.Random(seed).shuffle(unique)
    holdout_keys = set(unique[:max(1, round(len(unique)*.2))]) if len(unique) >= 3 else set()
    training = [e for e in events if e["split_key"] not in holdout_keys]
    development = [e for e in events if e["split_key"] in holdout_keys]
    if not training:
        raise ValueError("Sem OK suficientes para bootstrap")
    dl = DataLoader(
        DeslocadoProxyDataset(training, data, size),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        total = 0.
        for a,b,c,d,mask,truth,_ in dl:
            out, each = model(a,b,c,d,mask)
            loss = nn.functional.binary_cross_entropy_with_logits(out, truth)
            aux = nn.functional.binary_cross_entropy_with_logits(
                each, truth[:,None].expand_as(each), reduction="none"
            )
            loss = loss + .1*(aux*mask).sum()/mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optimizer.step()
            total += float(loss.item())*len(truth)
        history.append({"epoch": epoch, "training_proxy_loss": round(total/len(dl.dataset), 6)})
        print(f"[DESLOCADO {epoch}/{epochs}] proxy_loss={history[-1]['training_proxy_loss']:.5f}",
              flush=True)
    train_metrics = _evaluate(model, training, data, size)
    dev_metrics = (_evaluate(model, development, data, size)
                   if development else None)
    now = datetime.now(timezone.utc)
    folder = data["root"]/"reports"/"deslocado_neural"/"models"/(
        "experiment_"+now.strftime("%Y%m%dT%H%M%S_%fZ")
    )
    folder.mkdir(parents=True, exist_ok=False)
    source_sha = sha256(Path(manifest_path).read_bytes()).hexdigest()
    torch.save({
        "schema": MODEL_SCHEMA_DESLOCADO,
        "state_dict": model.cpu().state_dict(),
        "image_size": size, "focus_fraction": .70, "lights": LIGHTS,
        "training_type": "OK_AND_SYNTHETIC_SHIFT_PROXY",
        "source_manifest_sha256": source_sha,
        "experimental": True, "production_approved": False,
        "allow_automatic_classification": False,
        "real_ng_used": 0,
    }, folder/"deslocado_cnn_candidate.pt")
    report = {
        "schema": TRAIN_SCHEMA,
        "source_manifest": str(Path(manifest_path).resolve()),
        "source_manifest_sha256": source_sha,
        "category": "DESLOCADO",
        "model_schema": MODEL_SCHEMA_DESLOCADO,
        "experiments": "REAL_OK_VS_SYNTHETIC_LOCAL_SHIFT",
        "training_events": len(training),
        "development_events": len(development),
        "total_real_ok_events": len(events),
        "ng_real_events": 0,
        "can_measure_real_ng_recall": False,
        "production_approved": False,
        "activation_disabled": True,
        "split_by_board_part": True,
        "warnings": data["warnings"],
        "train_results": train_metrics,
        "development_results": dev_metrics,
        "training_history": history,
        "limitations": [
            "Sem nenhum NG DESLOCADO real: modelo apenas distingue exemplos OK de transformações artificiais.",
            "Um deslocamento sintético de patch central não equivale ao movimento real de um componente.",
            "Possíveis artefatos nas bordas do patch podem ser aprendidos indevidamente.",
            "Não usar sigmoid nem acurácia proxy como garantia para NG reais.",
            "CNN não integrada na decisão de produção; motores físicos existentes permanecem.",
        ],
    }
    (folder/"training_report_deslocado.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder/"training_summary_deslocado.txt").write_text(
        "ODIN CNN DESLOCADO — CANDIDATO EXPERIMENTAL\n"
        f"OK reais: {len(events)} eventos; NG reais: ZERO.\n"
        f"Treino {len(training)}, desenvolvimento {len(development)}.\n"
        f"Experimento: OK vs deslocamento LOCAL SINTÉTICO.\n"
        f"Avaliação de desenvolvimento: {dev_metrics}\n"
        "CNN SEM AUTORIZAÇÃO DE PRODUÇÃO. Motores físicos DESLOCADO mantidos.\n",
        encoding="utf-8"
    )
    return report, folder


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap CNN DESLOCADO sem NG real")
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    manifest = args.manifest or latest_deslocado_manifest(args.root)
    report, folder = train_deslocado(
        manifest, epochs=args.epochs, size=args.size, batch_size=args.batch_size
    )
    print("Relatório:", folder/"training_report_deslocado.json")
    print("Resumo:", folder/"training_summary_deslocado.txt")
    print("A CNN DESLOCADO não foi ativada em produção.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
