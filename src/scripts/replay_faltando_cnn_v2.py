"""Replay offline do acervo inteiro FALTANDO com CNN v2 (SEM KNN).

Executar no Win10:
    python -m src.scripts.replay_faltando_cnn_v2

Verifica todos os arquivos FALTANDO atuais em OK/NG archive, integridade do
manifesto e pesos treinados, computa julgamentos por PNG e por evento.
NUNCA treina, altera dataset, emite comando XP ou ativa modelo em Produção.

Acurácia no acervo usado no treinamento é regressão conhecida, NÃO avaliação
cega/generalização. Não promover checkpoint automaticamente em caso de 100%.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, LIGHTS, MODEL_SCHEMA_V2
from src.scripts.train_faltando_cnn import load_events
from src.scripts.train_faltando_cnn_v2 import DualScaleEventDataset
from src.services.faltando_neural_qualification import latest_manifest

REPLAY_SCHEMA = "visionx.faltando_archive_replay.v1"
THRESHOLD = .50


def _current_archive_paths(root: Path) -> set[str]:
    """Mesmo filtro FALTANDO da preparação; inclui _SIDE/_TOP/_MID_2."""
    public = root / "public"
    found = set()
    for folder in ("ng_archive", "ok_archive"):
        root_archive = public / folder
        if not root_archive.is_dir():
            raise FileNotFoundError(f"Arquivo de evidências indisponível: {root_archive}")
        for path in root_archive.rglob("*"):
            if (path.is_file() and path.suffix.lower() == ".png"
                    and "_FALTANDO" in path.stem.upper()):
                found.add(path.relative_to(root).as_posix())
    return found


def _select_checkpoint(root: Path, explicit: Path | None) -> Path:
    models_dir = root / "reports" / "faltando_neural" / "models"
    if explicit is not None:
        path = explicit.expanduser().resolve()
    else:
        candidates = sorted(
            models_dir.glob("experiment_v2_*/faltando_cnn_v2_candidate.pt"),
            key=lambda p: p.parent.name, reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                "Nenhum checkpoint CNN FALTANDO v2 encontrado. Informe --checkpoint."
            )
        path = candidates[0].resolve()
    if (
        models_dir.resolve() not in path.parents
        or not path.is_file()
        or path.name != "faltando_cnn_v2_candidate.pt"
        or path.is_symlink()
    ):
        raise ValueError("Checkpoint deve ser um candidato v2 dentro de reports/faltando_neural/models/")
    return path


def _counts(records: list[dict]) -> dict:
    true_ok = [r for r in records if r["label"] == "OK"]
    true_ng = [r for r in records if r["label"] == "NG"]
    tp = sum(r["predicted"] == "NG" for r in true_ng)
    tn = sum(r["predicted"] == "OK" for r in true_ok)
    fp = len(true_ok) - tn
    fn = len(true_ng) - tp
    return {
        "cases": len(records), "expected_OK": len(true_ok),
        "expected_NG": len(true_ng), "TP_NG": tp, "TN_OK": tn,
        "FP_OK_as_NG": fp, "FN_NG_as_OK": fn,
        "accuracy": round((tp+tn)/len(records), 6) if records else None,
        "passed_all": len(records) > 0 and fp == 0 and fn == 0,
    }


def replay_archive(
    root: Path, *, manifest: Path | None = None,
    checkpoint: Path | None = None, batch_size: int = 4,
    device: str = "cpu",
) -> tuple[dict, Path]:
    """Falha se qualquer arquivo atual não estiver presente no staging.

    Reusa exatamente DualScaleEventDataset e FaltandoCNNV2 do treinamento.
    """
    root = Path(root).expanduser().resolve()
    if batch_size < 1 or batch_size > 128:
        raise ValueError("batch_size inválido")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA não disponível")
    manifest = (
        Path(manifest).expanduser().resolve()
        if manifest is not None else latest_manifest(root)
    )
    selected_ckpt = _select_checkpoint(root, checkpoint)
    events, data = load_events(manifest)

    if data["root"].resolve() != root:
        raise ValueError("Raiz do staging diferente da raiz especificada")
    if data["warnings"]:
        raise ValueError(
            "Há trincas inconsistentes; replay incompleto: "
            + "; ".join(data["warnings"][:5])
        )

    checkpoint_hash = sha256(selected_ckpt.read_bytes()).hexdigest()
    contents = torch.load(selected_ckpt, map_location="cpu", weights_only=True)
    if (
        not isinstance(contents, dict)
        or contents.get("schema") != MODEL_SCHEMA_V2
        or tuple(contents.get("lights", ())) != LIGHTS
        or not contents.get("experimental", False)
        or contents.get("production_approved") is not False
    ):
        raise ValueError("Checkpoint diferente do modelo experimental v2 esperado")
    expected_manifest_hash = sha256(manifest.read_bytes()).hexdigest()
    if contents.get("source_manifest_sha256") != expected_manifest_hash:
        raise ValueError(
            "Checkpoint não corresponde ao manifesto de origem. "
            "Use o manifest.json exato do treinamento com --manifest."
        )
    size = contents["image_size"]
    focus = contents["focus_fraction"]
    if not isinstance(size, int) or not 64 <= size <= 512 or size % 32:
        raise ValueError("Tamanho inválido nos metadados do checkpoint")
    if not isinstance(focus, (int, float)) or not .5 <= focus <= .90:
        raise ValueError("Escala de foco inválida nos metadados do checkpoint")

    tracked_paths = set(data["sample_map"])
    actual_paths = _current_archive_paths(root)
    if actual_paths != tracked_paths:
        missing = sorted(tracked_paths - actual_paths)
        added = sorted(actual_paths - tracked_paths)
        raise ValueError(
            "O acervo atual difere do acervo preparado. "
            f"Adicionados={len(added)} {added[:4]}; "
            f"removidos={len(missing)} {missing[:4]}. "
            "Não declarar teste de todos os arquivos sem atualizar preparação."
        )
    if len(tracked_paths) != sum(len(e["observations"]) for e in events):
        raise ValueError("Algum PNG ficou fora dos eventos; replay abortado")

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    net = FaltandoCNNV2().to(device)
    net.load_state_dict(contents["state_dict"], strict=True)
    net.eval()
    dataset = DualScaleEventDataset(
        events, data, size=size, focus_fraction=float(focus), augment=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    by_event = {e["id"]: e for e in events}
    event_records = []
    image_records = []
    idx_total = 0
    with torch.inference_mode():
        for full_ref, full_test, focus_ref, focus_test, mask, truth, ids in loader:
            input_tensors = [
                tensor.to(device)
                for tensor in (full_ref, full_test, focus_ref, focus_test, mask)
            ]
            logits, light_logits = net(*input_tensors)
            event_scores = torch.sigmoid(logits).cpu()
            light_scores = torch.sigmoid(light_logits).cpu()
            for j, key in enumerate(ids):
                event = by_event[key]
                label = event["label"]
                score = float(event_scores[j].item())
                predicted = "NG" if score >= THRESHOLD else "OK"
                per_light = {}
                for k, light in enumerate(LIGHTS):
                    if light not in event["observations"]:
                        continue
                    source = event["observations"][light]
                    probability = float(light_scores[j, k].item())
                    local_predicted = "NG" if probability >= THRESHOLD else "OK"
                    frame = {
                        "source_path": source,
                        "event_id_or_group_hint": key,
                        "lighting_mode": light,
                        "lighting_source": data["sample_map"][source].get("lighting_source"),
                        "label": label,
                        "ng_score": round(probability, 8),
                        "predicted": local_predicted,
                        "correct": local_predicted == label,
                    }
                    per_light[light] = round(probability, 8)
                    image_records.append(frame)
                record = {
                    "event_id_or_group_hint": key,
                    "association": event["association"],
                    "label": label,
                    "predicted": predicted,
                    "ng_score": round(score, 8),
                    "per_light_ng_scores": per_light,
                    "sources": dict(event["observations"]),
                    "correct": predicted == label,
                    "error": (
                        "FN_NG_AS_OK" if label == "NG" and predicted == "OK"
                        else "FP_OK_AS_NG" if label == "OK" and predicted == "NG"
                        else None
                    ),
                }
                event_records.append(record)
                idx_total += 1
    if idx_total != len(events) or len(image_records) != len(tracked_paths):
        raise AssertionError("Inferência não abrangeu todos os arquivos/eventos")

    image_by_light = {
        light: _counts([
            r for r in image_records if r["lighting_mode"] == light
        ]) for light in LIGHTS
    }
    legacy = _counts([
        r for r in image_records if r["lighting_source"] == "LEGACY_DEFAULT"
    ])
    event_counts = _counts(event_records)
    frame_counts = _counts(image_records)
    # Para rótulos OK a aprovação por luz também importa; eventos NG
    # multilight futuros só exigem NG na decisão final por evento.
    passed = (
        event_counts["passed_all"]
        and frame_counts["passed_all"]
        and not data["warnings"]
        and len(image_records) == len(actual_paths)
    )
    now = datetime.now(timezone.utc)
    out = (
        root / "reports" / "faltando_neural" / "replays"
        / ("archive_v2_" + now.strftime("%Y%m%dT%H%M%S_%fZ"))
    )
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": REPLAY_SCHEMA,
        "created_at_utc": now.isoformat(),
        "model_schema": MODEL_SCHEMA_V2,
        "checkpoint": str(selected_ckpt),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_best_epoch": contents.get("best_epoch"),
        "source_manifest": str(manifest),
        "source_manifest_sha256": expected_manifest_hash,
        "threshold": THRESHOLD,
        "knn_used": False,
        "training_performed": False,
        "production_modified": False,
        "kind": "KNOWN_ARCHIVE_REGRESSION_NOT_INDEPENDENT_TEST",
        "can_prove_generalization": False,
        "ready_for_automatic_production": False,
        "archive_unchanged": True,
        "images_scanned": len(actual_paths),
        "events_scanned": len(events),
        "counts_per_event": event_counts,
        "counts_per_image": frame_counts,
        "counts_legacy_side": legacy,
        "counts_per_light": image_by_light,
        "passed_known_archive_regression": passed,
        "event_records": event_records,
        "image_records": image_records,
        "errors": {
            "events": [r for r in event_records if not r["correct"]],
            "images": [r for r in image_records if not r["correct"]],
        },
        "limitations": [
            "Grande parte deste acervo participou do treinamento da CNN v2.",
            "Isto é um teste de regressão de exemplos conhecidos, não teste cego.",
            "Não existem NG reais TOP ou MID no acervo atual.",
            "Trincas por nome+OCR não possuem event_id verificado.",
            "Scores sigmoid não são probabilidades calibradas.",
        ],
    }
    (out / "archive_replay_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = [
        "ODIN CNN FALTANDO v2 — REPLAY COMPLETO DE ARQUIVO",
        "SEM KNN, SEM RETREINO, SEM COMANDOS XP, SEM MUDAR PRODUÇÃO",
        f"Modelo: {selected_ckpt.name} | época {contents.get('best_epoch')}",
        f"Imagens avaliadas: {len(image_records)} | eventos: {len(event_records)}",
        f"Por evento: {event_counts}",
        f"Por imagem: {frame_counts}",
        f"SIDE legado: {legacy}",
    ]
    for light in LIGHTS:
        text.append(f"Iluminação {light}: {image_by_light[light]}")
    text.extend([
        f"REPLAY DOS CASOS CONHECIDOS: {'PASSOU' if passed else 'FALHOU'}",
        "PRONTO PARA PRODUÇÃO: NÃO DETERMINADO POR ESTE REPLAY",
        "Não é teste cego: várias imagens foram usadas no treinamento.",
        "Sem NG reais TOP/MID. Não habilitar automaticamente o modelo.",
        "DIVERGÊNCIAS:",
    ])
    for row in report["errors"]["events"]:
        text.append(
            f"{row['error']} | {row['event_id_or_group_hint']} "
            f"| real={row['label']} pred={row['predicted']} "
            f"score_ng={row['ng_score']}"
        )
    for row in report["errors"]["images"]:
        text.append(
            f"IMAGEM | {row['source_path']} [{row['lighting_mode']}] "
            f"| real={row['label']} pred={row['predicted']} "
            f"score_ng={row['ng_score']}"
        )
    (out / "archive_replay_v2.txt").write_text("\n".join(text)+"\n", encoding="utf-8")
    return report, out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Reprocessar todos PNG FALTANDO com CNN v2, sem KNN."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    report, output = replay_archive(
        args.root, manifest=args.manifest,
        checkpoint=args.checkpoint, batch_size=args.batch_size, device=args.device
    )
    print("Imagens:", report["images_scanned"],
          "| eventos:", report["events_scanned"])
    print("Matriz por evento:", report["counts_per_event"])
    print("Matriz por imagem:", report["counts_per_image"])
    print("SIDE legado:", report["counts_legacy_side"])
    for light, counts in report["counts_per_light"].items():
        print(light + ":", counts)
    print(
        "REGRESSÃO CONHECIDA:",
        "PASSOU" if report["passed_known_archive_regression"] else "FALHOU",
    )
    print("JSON:", output / "archive_replay_v2.json")
    print("TXT:", output / "archive_replay_v2.txt")
    print("Nenhum código de Produção foi ativado.")
    return 0 if report["passed_known_archive_regression"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
