"""Treino incremental FALTANDO v2 em processo CPU independente da AOI.

Treina imediatamente após rótulo humano persistido, com replay do acervo
histórico + novos eventos. O peso em produção só muda se passar regressão
por CADA imagem conhecida e CADA novo exemplo. Falha => mantém champion.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from uuid import uuid4

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, LIGHTS, MODEL_SCHEMA_V2
from src.core.neural.faltando_live import (
    PINNED_CHECKPOINT_SHA256, PINNED_RELATIVE_CHECKPOINT,
)
from src.scripts.train_faltando_cnn import load_events
from src.scripts.train_faltando_cnn_v2 import (
    _focus_crop, _letterbox_rgb,
)
from src.services.neural_online_learning import ONLINE_SCHEMA, _known_human_source

ONLINE_MODEL_SCHEMA = "visionx.specialist_online_model.v1"
POINTER_SCHEMA = "visionx.specialist_active_pointer.v1"


def _safe_path(parent: Path, filename: str) -> Path:
    if (not isinstance(filename, str) or Path(filename).name != filename
            or not filename.endswith(".png")):
        raise ValueError("Nome de recorte não permitido")
    candidate = parent / filename
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("Recorte online ausente ou link simbólico")
    return candidate


def _read(path: Path) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if frame is None or frame.ndim != 3 or min(frame.shape[:2]) < 12:
        raise ValueError("Imagem de treino ilegível")
    return frame


def _on_disk_event(json_path: Path, root: Path) -> dict:
    folder = root / "reports" / "neural_online" / "events"
    if json_path.parent.resolve() != folder.resolve() or json_path.is_symlink():
        raise ValueError("Evento online fora do journal")
    row = json.loads(json_path.read_text(encoding="utf-8"))
    if (row.get("schema") != ONLINE_SCHEMA
            or row.get("category") != "FALTANDO"
            or row.get("label") not in ("OK", "NG")
            or not row.get("training_requested")
            or not _known_human_source(row.get("human_source", ""))):
        raise ValueError("Evento não possui rótulo humano treinável")
    if row.get("event_id") != json_path.stem:
        raise ValueError("Identidade do evento online diferente do arquivo")
    views = {}
    for item in row.get("images", []):
        mode = item.get("lighting_mode", "")
        if mode not in LIGHTS or mode in views:
            raise ValueError("Iluminação online inválida/repetida")
        paths = {}
        for part in ("reference", "test"):
            path = _safe_path(folder, item.get(part, ""))
            if sha256(path.read_bytes()).hexdigest() != item.get(part + "_sha256"):
                raise ValueError("Fingerprint online divergente")
            paths[part] = path
        views[mode] = paths
    if not views:
        raise ValueError("Evento sem pares válidos")
    return {
        "id": row["event_id"],
        "label": row["label"],
        "views": views,
        "aoi_info": row.get("aoi_info", {}),
        "source": json_path,
    }


def _base_model(root: Path) -> tuple[Path, dict]:
    """Checkpoint original ou último promovido (nunca candidato rejeitado)."""
    initial = (root / PINNED_RELATIVE_CHECKPOINT).resolve()
    base = initial
    expected = PINNED_CHECKPOINT_SHA256
    pointer = root / "reports" / "neural_online" / "live_active.json"
    if pointer.exists():
        row = json.loads(pointer.read_text(encoding="utf-8"))
        if row.get("schema") != POINTER_SCHEMA:
            raise ValueError("Ponteiro online incompatível")
        base = (root / row["checkpoint_relative_path"]).resolve()
        allowed = (root / "reports" / "neural_online" / "checkpoints").resolve()
        if allowed not in base.parents:
            raise ValueError("Checkpoint fora da pasta online permitida")
        expected = row["checkpoint_sha256"]
    if not base.is_file() or base.is_symlink():
        raise ValueError("Checkpoint ativo ausente")
    if sha256(base.read_bytes()).hexdigest() != expected:
        raise ValueError("Checkpoint ativo alterado/corrompido")
    obj = torch.load(base, map_location="cpu", weights_only=True)
    if (not isinstance(obj, dict)
            or obj.get("schema") != MODEL_SCHEMA_V2
            or obj.get("experimental") is not True
            or obj.get("production_approved") is not False
            or tuple(obj.get("lights", ())) != LIGHTS):
        raise ValueError("Modelo ativo não é CNN FALTANDO v2 experimental")
    return base, obj


def _historical(root: Path, checkpoint: dict) -> list[dict]:
    """Reusa o conjunto conhecido integral, sem omitir casos."""
    directory = root / "reports" / "faltando_neural"
    fingerprint = str(checkpoint.get("source_manifest_sha256", ""))
    matches = [
        path for path in directory.glob("run_*/manifest.json")
        if sha256(path.read_bytes()).hexdigest() == fingerprint
    ]
    if len(matches) != 1:
        raise ValueError(
            "Manifesto original não localizado (ou duplicado). "
            "Treino incremental não pode validar a regressão histórica."
        )
    events, data = load_events(matches[0])
    observed = len(data["sample_map"])
    actual = sum(len(e["observations"]) for e in events)
    if observed < 1 or observed != actual:
        raise ValueError("Histórico de imagens incompleto")
    output = []
    for event in events:
        views = {}
        for mode, source in event["observations"].items():
            item = data["sample_map"][source]
            views[mode] = {
                "reference": data["run_dir"] / item["reference_path"],
                "test": data["run_dir"] / item["test_path"],
            }
        output.append({
            "id": event["id"], "label": event["label"],
            "views": views, "source": matches[0],
        })
    return output


def _identity(event: dict) -> tuple:
    """Detecta conflito entre exemplos novos da MESMA peça e contexto."""
    info = event.get("aoi_info", {})
    return (
        "".join(c for c in str(info.get("board", "")).upper() if c.isalnum()),
        "".join(c for c in str(info.get("parts", "")).upper() if c.isalnum()),
        str(info.get("value", "")).strip(),
        tuple(sorted(
            (mode,
             sha256(v["reference"].read_bytes()).hexdigest(),
             sha256(v["test"].read_bytes()).hexdigest())
            for mode, v in event["views"].items()
        )),
    )


def _all_online(root: Path) -> list[dict]:
    folder = root / "reports" / "neural_online" / "events"
    rows = []
    # Uma amostra validada por operador equivale a um evento (3 luzes = 1).
    for path in sorted(folder.glob("*.json")):
        rows.append(_on_disk_event(path, root))
    by_identity = {}
    for row in rows:
        key = _identity(row)
        if key in by_identity and by_identity[key] != row["label"]:
            raise ValueError(
                "Mesmo par online tem rótulos humanos conflitantes: "
                "não promover o novo modelo."
            )
        by_identity[key] = row["label"]
    return rows


class PairDataset(Dataset):
    def __init__(self, events: list[dict], size: int, fraction: float):
        self.events, self.size, self.fraction = events, size, fraction

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index: int):
        event = self.events[index]
        pictures = [
            torch.zeros((3, 3, self.size, self.size))
            for _ in range(4)
        ]
        light_mask = torch.zeros(3)
        for ix, light in enumerate(LIGHTS):
            if light not in event["views"]:
                continue
            pair = event["views"][light]
            ref, tst = _read(pair["reference"]), _read(pair["test"])
            pictures[0][ix] = _letterbox_rgb(ref, self.size)
            pictures[1][ix] = _letterbox_rgb(tst, self.size)
            pictures[2][ix] = _letterbox_rgb(
                _focus_crop(ref, self.fraction), self.size
            )
            pictures[3][ix] = _letterbox_rgb(
                _focus_crop(tst, self.fraction), self.size
            )
            light_mask[ix] = 1.
        return (
            *pictures, light_mask,
            torch.tensor(float(event["label"] == "NG"), dtype=torch.float32),
        )


def _assess(model, records: list[dict], size: int, focus: float) -> dict:
    model.eval()
    outcomes = []
    loader = DataLoader(
        PairDataset(records, size, focus),
        batch_size=4, shuffle=False, num_workers=0,
    )
    cursor = 0
    with torch.inference_mode():
        for r, t, rf, tf, mask, target in loader:
            logits, by_light = model(r, t, rf, tf, mask)
            prob = torch.sigmoid(logits).tolist()
            mode_scores = torch.sigmoid(by_light).tolist()
            for idx, score in enumerate(prob):
                event = records[cursor]
                cursor += 1
                label = event["label"]
                verdict = "NG" if score >= .5 else "OK"
                individual = {}
                for j, light in enumerate(LIGHTS):
                    if light not in event["views"]:
                        continue
                    by_mode_pred = "NG" if mode_scores[idx][j] >= .5 else "OK"
                    individual[light] = {
                        "score": round(mode_scores[idx][j], 6),
                        "correct": by_mode_pred == label,
                    }
                outcomes.append({
                    "event_id_or_group_hint": event["id"],
                    "label": label, "predicted": verdict,
                    "ng_score": round(score, 6),
                    "correct": verdict == label,
                    "per_light": individual,
                })
    all_modes_ok = all(
        item["correct"] and all(x["correct"] for x in item["per_light"].values())
        for item in outcomes
    )
    return {
        "events": len(records),
        "images": sum(len(r["views"]) for r in records),
        "correct_events": sum(r["correct"] for r in outcomes),
        "correct_images": sum(
            int(mode["correct"])
            for r in outcomes for mode in r["per_light"].values()
        ),
        "all_correct": bool(outcomes) and all_modes_ok,
        "ng_false_ok": sum(
            r["label"] == "NG" and r["predicted"] == "OK" for r in outcomes
        ),
        "errors": [
            r for r in outcomes
            if not r["correct"] or
            not all(x["correct"] for x in r["per_light"].values())
        ],
    }


def train_online(root: Path, event: Path, *, epochs: int = 3) -> dict:
    """Modifica só candidate/pointer; nunca mexe em checkpoint base."""
    root = Path(root).resolve()
    event = Path(event).resolve()
    if not 1 <= epochs <= 10:
        raise ValueError("epochs fora do intervalo")
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    newest = _on_disk_event(event, root)
    _, prior = _base_model(root)
    historical = _historical(root, prior)
    online = _all_online(root)
    size = int(prior["image_size"])
    focus = float(prior["focus_fraction"])
    if not 64 <= size <= 512 or not .5 <= focus <= .90:
        raise ValueError("Checkpoint possui pré-processamento incorreto")
    net = FaltandoCNNV2()
    net.load_state_dict(prior["state_dict"], strict=True)

    # Retreino curto; a memória histórica participa de TODA atualização.
    examples = historical + online
    freq = Counter(r["label"] for r in examples)
    weights = [1. / freq[r["label"]] for r in examples]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(examples), replacement=True,
        generator=torch.Generator().manual_seed(42),
    )
    loader = DataLoader(
        PairDataset(examples, size, focus), batch_size=4,
        sampler=sampler, num_workers=0
    )
    opt = torch.optim.AdamW(net.parameters(), lr=.00002, weight_decay=.02)
    # Garantir gradiente da amostra nova em TODA época, em vez de
    # depender de sua seleção probabilística no replay balanceado.
    fresh_loader = DataLoader(
        PairDataset([newest], size, focus), batch_size=1,
        shuffle=False, num_workers=0,
    )
    net.train()
    for ix in range(1, epochs+1):
        average = 0.
        for r, t, rf, tf, mask, truth in loader:
            logits, by_light = net(r, t, rf, tf, mask)
            base = nn.functional.binary_cross_entropy_with_logits(logits, truth)
            lights = nn.functional.binary_cross_entropy_with_logits(
                by_light, truth[:, None].expand_as(by_light), reduction="none"
            )
            regularizer = (lights * mask).sum() / mask.sum().clamp_min(1)
            loss = base + .15 * regularizer
            if not bool(torch.isfinite(loss)):
                raise ValueError("Perda neural não finita")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            average += float(loss.item())
        # Atualização supervisionada obrigatória para o novo evento, com
        # SIDE/TOP/MID na mesma amostra (uma decisão humana por peça).
        for r, t, rf, tf, mask, truth in fresh_loader:
            logits, by_light = net(r, t, rf, tf, mask)
            primary = nn.functional.binary_cross_entropy_with_logits(logits, truth)
            aux = nn.functional.binary_cross_entropy_with_logits(
                by_light, truth[:, None].expand_as(by_light), reduction="none"
            )
            loss = primary + .15 * (aux * mask).sum()/mask.sum().clamp_min(1)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Perda neural não finita no evento novo")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
        print(f"CNN FALTANDO ONLINE época {ix}/{epochs} loss={average/len(loader):.6f}",
              flush=True)

    previous = FaltandoCNNV2()
    previous.load_state_dict(prior["state_dict"], strict=True)
    old_result = _assess(previous, historical, size, focus)
    old_online = _assess(previous, online, size, focus)
    new_result = _assess(net, historical, size, focus)
    new_online = _assess(net, online, size, focus)
    if not old_result["all_correct"]:
        raise ValueError("Modelo anterior não passou replay conhecido. Promoção bloqueada.")

    gates = {
        "historical_unchanged": new_result["all_correct"],
        "all_confirmed_online_correct": new_online["all_correct"],
        "no_new_historical_false_ok": new_result["ng_false_ok"] == 0,
        "source_historical_count": old_result["images"],
        "no_reduction_in_correct_historical_images": (
            new_result["correct_images"] >= old_result["correct_images"]
        ),
    }
    accepted = all(isinstance(v, bool) and v for k, v in gates.items()
                   if k not in ("source_historical_count",))
    when = datetime.now(timezone.utc)
    checkpoints = root / "reports" / "neural_online" / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    stem = "faltando_v2_" + when.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:6]
    target = checkpoints / (stem + ".pt")
    candidate = {
        **prior,
        "state_dict": {k: t.cpu().clone() for k, t in net.state_dict().items()},
        "online_training_schema": ONLINE_MODEL_SCHEMA,
        "online_training_last_event": event.stem,
        "online_training_at_utc": when.isoformat(),
        "experimental": True, "production_approved": False,
    }
    torch.save(candidate, target)
    digest = sha256(target.read_bytes()).hexdigest()
    report = {
        "schema": "visionx.specialist_online_training_report.v1",
        "event_id": event.stem,
        "created_at_utc": when.isoformat(),
        "category": "FALTANDO",
        "train_epochs": epochs,
        "new_event_forced_each_epoch": True,
        "training_event_count": len(examples),
        "online_events": len(online),
        "online_illuminations": dict(Counter(
            mode for item in online for mode in item["views"]
        )),
        "historical_baseline": old_result,
        "historical_candidate": new_result,
        "online_baseline": old_online,
        "online_candidate": new_online,
        "gates": gates,
        "promoted": accepted,
        "automatic_ok_authorized": False,
        "candidate": str(target.relative_to(root)),
        "candidate_sha256": digest,
        "note": (
            "Regressão de dados conhecidos é guarda contra esquecimento, "
            "não evidência independente de segurança para liberação automática."
        ),
    }
    reports = root / "reports" / "neural_online" / "training_reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / (stem + ".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    if accepted:
        pointer = root / "reports" / "neural_online" / "live_active.json"
        payload = {
            "schema": POINTER_SCHEMA,
            "category": "FALTANDO",
            "model_schema": MODEL_SCHEMA_V2,
            "checkpoint_relative_path": target.relative_to(root).as_posix(),
            "checkpoint_sha256": digest,
            "promoted_at_utc": when.isoformat(),
            "event_id": event.stem,
            "validation_report": report_path.relative_to(root).as_posix(),
            "experimental": True,
            "production_approved": False,
        }
        tmp = pointer.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        # Troca atômica: se cair energia aqui, champion anterior permanece.
        os.replace(tmp, pointer)
    # Resultado de máquina para o journal/status da UI. A promoção só
    # ocorre pelo ponteiro após a validação; um candidato rejeitado permanece
    # auditável no disco, sem tocar o modelo ativo.
    outcomes = root / "reports" / "neural_online" / "outcomes"
    outcomes.mkdir(parents=True, exist_ok=True)
    outcome = outcomes / (event.stem + ".json")
    temp = outcome.with_suffix(".json.tmp")
    temp.write_text(json.dumps({
        "event_id": event.stem,
        "promoted": accepted,
        "validation_report": report_path.relative_to(root).as_posix(),
        "candidate": target.relative_to(root).as_posix(),
        "historical_correct_images": new_result["correct_images"],
        "historical_images": new_result["images"],
        "online_correct_images": new_online["correct_images"],
        "online_images": new_online["images"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, outcome)
    print(
        f"CNN ONLINE: candidato {'PROMOVIDO' if accepted else 'REJEITADO'}; "
        f"histórico={new_result['correct_images']}/{new_result['images']}, "
        f"online={new_online['correct_images']}/{new_online['images']}",
        flush=True,
    )
    print("RELATÓRIO:", report_path, flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Treino online seguro CNN FALTANDO v2")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args(argv)
    result = train_online(args.root, args.event, epochs=args.epochs)
    print("Ativo atualizado:", result["promoted"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
