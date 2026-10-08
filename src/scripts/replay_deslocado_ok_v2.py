"""Replay estrito de TODOS os OK DESLOCADO do arquivo local, com CNN v2.

Não altera motor, não re-treina, não aprova produção. PASSOU significa
somente ausência de falsos NG nos OK já conhecidos. Com zero NG reais
não é possível provar que a rede detecta DESLOCADO real; jamais
promover apenas por acertos em OK.
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

from src.core.neural.deslocado_cnn import DeslocadoCNN
from src.core.neural.faltando_cnn_v2 import LIGHTS
from src.scripts.train_deslocado_cnn import load_ok_events
from src.scripts.train_deslocado_cnn_v2 import (
    DeslocadoV2Dataset, MODEL_SCHEMA,
)
from src.services.deslocado_neural_dataset import latest_deslocado_manifest

SCHEMA = "visionx.deslocado_ok_archive_replay.v1"
THRESHOLD = 0.5


def all_archived_deslocado(root: Path) -> tuple[set[str], set[str]]:
    """Conferir acervo por caminho, inclusive NG novo não preparado."""
    public = root / "public"
    paths = []
    for label, folder in (("OK", "ok_archive"), ("NG", "ng_archive")):
        directory = public / folder
        if not directory.is_dir():
            raise FileNotFoundError("Pasta não encontrada: " + str(directory))
        found = set()
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("Arquivo simbólico no acervo: " + str(path))
            if (path.is_file() and path.suffix.lower() == ".png"
                    and "_DESLOCADO" in path.stem.upper()):
                found.add(path.relative_to(root).as_posix())
        paths.append(found)
    return paths[0], paths[1]


def checkpoint_path(root: Path, chosen: Path | None) -> Path:
    base = (root / "reports" / "deslocado_neural" / "models").resolve()
    if chosen is None:
        choices = sorted(
            base.glob("experiment_v2_*/deslocado_cnn_v2_candidate.pt"),
            key=lambda p: p.parent.name, reverse=True
        )
        if not choices:
            raise FileNotFoundError("Treine a v2 ou informe --checkpoint")
        selected = choices[0].resolve()
    else:
        selected = chosen.expanduser().resolve()
    if (base not in selected.parents or not selected.is_file()
            or selected.is_symlink()
            or selected.name != "deslocado_cnn_v2_candidate.pt"):
        raise ValueError("Checkpoint v2 deve estar em reports/deslocado_neural/models/")
    return selected


def summarize(rows: list[dict]) -> dict:
    passed = sum(r["correct"] for r in rows)
    return {
        "OK_total": len(rows), "OK_correct": passed,
        "false_NG_on_OK": len(rows) - passed,
        "all_OK_correct": len(rows) > 0 and passed == len(rows),
    }


def replay_deslocado(
    root: Path, *, manifest: Path | None = None,
    checkpoint: Path | None = None, batch_size: int = 4,
) -> tuple[dict, Path]:
    root = Path(root).expanduser().resolve()
    if not 1 <= batch_size <= 64:
        raise ValueError("batch_size inválido")
    manifest = (
        Path(manifest).expanduser().resolve() if manifest is not None
        else latest_deslocado_manifest(root)
    )
    ckpt = checkpoint_path(root, checkpoint)
    state_hash = sha256(ckpt.read_bytes()).hexdigest()
    content = torch.load(ckpt, weights_only=True, map_location="cpu")
    if (not isinstance(content, dict)
            or content.get("schema") != MODEL_SCHEMA
            or content.get("experimental") is not True
            or content.get("production_approved") is not False
            or content.get("allow_automatic_classification") is not False
            or content.get("real_ng_used") != 0
            or tuple(content.get("lights", ())) != LIGHTS):
        raise ValueError("Metadados CNN DESLOCADO v2 diferentes do esperado")
    size = content.get("image_size")
    fraction = content.get("focus_fraction")
    if (type(size) is not int or not 64 <= size <= 512 or size % 32 != 0
            or not isinstance(fraction, (int, float))
            or not .5 <= fraction <= .90):
        raise ValueError("Pré-processamento v2 incompatível")
    events, data = load_ok_events(manifest)
    if data["root"].resolve() != root:
        raise ValueError("Manifesto pertence a outra estação/repositório")
    if data["warnings"]:
        raise ValueError(
            "Grupos SIDE/TOP/MID ambíguos, rever manifesto: "
            + str(data["warnings"][:4])
        )
    current_ok, current_ng = all_archived_deslocado(root)
    prepared = set(data["samples"])
    if current_ng:
        raise ValueError(
            f"{len(current_ng)} NG DESLOCADO real(is) encontrado(s). "
            "Não usar protocolo OK-only; preparar validação NG supervisionada."
        )
    if not current_ok:
        raise ValueError("Nenhum PNG DESLOCADO OK disponível")
    if current_ok != prepared:
        missing = sorted(prepared - current_ok)
        added = sorted(current_ok - prepared)
        raise ValueError(
            "O manifesto não cobre TODO o arquivo OK atual. "
            f"Novos={len(added)} {added[:3]}, "
            f"removidos={len(missing)} {missing[:3]}. "
            "Reexecute python -m src.services.deslocado_neural_dataset "
            "e repita o replay com o manifesto atualizado."
        )
    # A função load_ok_events valida SHA-256 dos PNGs originais.
    if sum(len(event["observations"]) for event in events) != len(current_ok):
        raise ValueError("Cobertura por evento incompleta")

    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    model = DeslocadoCNN()
    model.load_state_dict(content["state_dict"], strict=True)
    model.eval()
    dataset = DeslocadoV2Dataset(
        events, data, size=size, training=False,
        proxy_variants=0, focus_fraction=float(fraction),
    )
    if dataset.counts() != {"REAL_OK": len(events)}:
        raise AssertionError("Replay não deve gerar defeitos sintéticos")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    events_by_id = {event["id"]: event for event in events}
    event_rows = []
    image_rows = []
    with torch.inference_mode():
        for full_ref, full_test, focus_ref, focus_test, mask, truth, ids, kinds in loader:
            event_logits, light_logits = model(
                full_ref, full_test, focus_ref, focus_test, mask
            )
            scores = torch.sigmoid(event_logits).tolist()
            light_scores = torch.sigmoid(light_logits).tolist()
            for j, key in enumerate(ids):
                if kinds[j] != "REAL_OK" or truth[j].item() != 0:
                    raise ValueError("Tentativa de avaliar proxy como OK original")
                info = events_by_id[key]
                score = float(scores[j])
                correct = score < THRESHOLD
                lights = {}
                for offset, mode in enumerate(LIGHTS):
                    if mode not in info["observations"]:
                        continue
                    source = info["observations"][mode]
                    value = float(light_scores[j][offset])
                    sample = data["samples"][source]
                    row = {
                        "source_path": source,
                        "event_id": key,
                        "lighting_mode": mode,
                        "lighting_source": sample.get("lighting_source"),
                        "expected": "OK", "predicted": (
                            "OK" if value < THRESHOLD else "NG"
                        ),
                        "ng_proxy_score_uncalibrated": round(value, 8),
                        "correct": value < THRESHOLD,
                    }
                    image_rows.append(row)
                    lights[mode] = row["ng_proxy_score_uncalibrated"]
                event_rows.append({
                    "event_id": key, "expected": "OK",
                    "predicted": "OK" if correct else "NG",
                    "ng_proxy_score_uncalibrated": round(score, 8),
                    "per_light_ng_scores": lights,
                    "paths": dict(info["observations"]),
                    "correct": correct,
                })
    if (len(image_rows) != len(current_ok)
            or len(event_rows) != len(events)
            or {row["source_path"] for row in image_rows} != current_ok):
        raise AssertionError("Nem todas as imagens OK foram classificadas")
    lights_counts = {
        mode: summarize([
            row for row in image_rows if row["lighting_mode"] == mode
        ]) for mode in LIGHTS
    }
    legacy_counts = summarize([
        row for row in image_rows if row["lighting_source"] == "LEGACY_DEFAULT"
    ])
    event_counts = summarize(event_rows)
    image_counts = summarize(image_rows)
    passed = event_counts["all_OK_correct"] and image_counts["all_OK_correct"]
    when = datetime.now(timezone.utc)
    output = (root / "reports" / "deslocado_neural" / "replays"
              / ("all_ok_v2_" + when.strftime("%Y%m%dT%H%M%S_%fZ")))
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": SCHEMA,
        "date_utc": when.isoformat(),
        "model": MODEL_SCHEMA,
        "model_best_epoch": content.get("best_epoch"),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": state_hash,
        "checkpoint_training_manifest_sha256": content.get("source_manifest_sha256"),
        "manifest": str(manifest),
        "evaluation_manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "evaluating_original_checkpoint_manifest": (
            content.get("source_manifest_sha256")
            == sha256(manifest.read_bytes()).hexdigest()
        ),
        "knn_used": False, "training_performed": False,
        "production_modified": False, "model_promoted": False,
        "historical_ok_archive_regression_passed": passed,
        "total_archive_pngs_evaluated": len(image_rows),
        "total_events_evaluated": len(event_rows),
        "real_NG_images_evaluated": 0,
        "real_NG_recall": None,
        "can_validate_NG_detection": False,
        "safe_to_replace_operational_engine": False,
        "safe_for_auto_OK": False,
        "failure_mode_if_all_OK_predicted": (
            "Always-OK classifier can score 100% on an OK-only archive."
        ),
        "per_image": image_counts,
        "per_event": event_counts,
        "legacy_SIDE": legacy_counts,
        "per_lighting": lights_counts,
        "failed_images": [r for r in image_rows if not r["correct"]],
        "failed_events": [r for r in event_rows if not r["correct"]],
        "all_image_predictions": image_rows,
        "all_event_predictions": event_rows,
        "limitations": [
            "O arquivo usado é integralmente OK, sem NG DESLOCADO real.",
            "O próprio modelo v2 foi treinado com parte deste acervo.",
            "Acertar todos os OK não prova que o detector reconhece um NG novo.",
            "O gerador de proxy v2 confundiu o caractere 104 com o corpo do resistor.",
            "Mantenha o motor físico e só habilite CNN em modo sombra até qualificar NG.",
        ],
    }
    (output/"deslocado_ok_replay_v2.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    text = [
        "ODIN DESLOCADO CNN v2 — REPLAY COMPLETO DOS OK",
        f"PNG OK: {len(image_rows)} | eventos: {len(event_rows)}",
        f"Por PNG: {image_counts}",
        f"Por evento: {event_counts}",
        f"LEGADO SIDE: {legacy_counts}",
        *[f"{mode}: {lights_counts[mode]}" for mode in LIGHTS],
        f"REGRESSÃO DE OK: {'PASSOU' if passed else 'REPROVOU'}",
        "NG DESLOCADO REAL: 0. RECALL NG NÃO MENSURÁVEL.",
        "SUBSTITUIÇÃO AUTOMÁTICA DO MOTOR: NÃO AUTORIZADA.",
        "Uma CNN que classifica tudo como OK também acertaria todo este arquivo.",
        "ERROS:",
        *[
            f"{row['source_path']} => NG, score={row['ng_proxy_score_uncalibrated']}"
            for row in report["failed_images"]
        ],
    ]
    (output/"deslocado_ok_replay_v2.txt").write_text(
        "\n".join(text)+"\n", encoding="utf-8"
    )
    return report, output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Avaliar todos os DESLOCADO OK, sem promover CNN à produção"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    report, folder = replay_deslocado(
        args.root, manifest=args.manifest,
        checkpoint=args.checkpoint, batch_size=args.batch_size,
    )
    print("Casos por PNG:", report["per_image"])
    print("Casos por evento:", report["per_event"])
    for mode, counts in report["per_lighting"].items():
        print(mode, counts)
    print("OK conhecidos:", "PASSOU" if report[
        "historical_ok_archive_regression_passed"
    ] else "REPROVOU")
    print("JSON:", folder/"deslocado_ok_replay_v2.json")
    print("TXT:", folder/"deslocado_ok_replay_v2.txt")
    print("CNN DESLOCADO não foi ativada em Produção.")
    return 0 if report["historical_ok_archive_regression_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
