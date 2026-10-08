"""Re-treino incremental DESLOCADO após rótulo humano NOVO nos três modos.

Registra decisões OK/NG do journal durável e gera apenas checkpoints
CANDIDATOS em reports/deslocado_neural. Com NG reais insuficientes,
NUNCA ativa pesos no ODIN; especialista físico continua em Produção.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from src.services.neural_online_learning import ONLINE_SCHEMA, _known_human_source
from src.services.deslocado_neural_dataset import latest_deslocado_manifest
from src.scripts.train_deslocado_cnn import train_deslocado


def load_online_deslocado(root: Path) -> tuple[list[dict], dict]:
    root = Path(root).resolve()
    folder = root/"reports"/"neural_online"/"events"
    events, views = [], {}
    for metadata in sorted(folder.glob("*.json")):
        if metadata.is_symlink():
            raise ValueError("Evento com link simbólico")
        data = json.loads(metadata.read_text(encoding="utf-8"))
        if data.get("schema") != ONLINE_SCHEMA:
            raise ValueError("Schema de journal inválido")
        if data.get("category") != "DESLOCADO":
            continue
        if (
            data.get("event_id") != metadata.stem
            or data.get("label") not in ("OK", "NG")
            or not _known_human_source(data.get("human_source", ""))
            or data.get("training_requested") is not True
        ):
            raise ValueError("Rótulo DESLOCADO online não verificável")
        observed = {}
        for item in data.get("images", []):
            mode = item.get("lighting_mode")
            if mode not in ("SIDE", "TOP", "MID") or mode in observed:
                raise ValueError("Iluminação duplicada ou inválida")
            verified = {}
            for part in ("reference", "test"):
                value = item.get(part)
                if not isinstance(value, str) or Path(value).name != value or not value.endswith(".png"):
                    raise ValueError("Nome inválido de PNG no journal")
                candidate = folder / value
                if not candidate.is_file() or candidate.is_symlink():
                    raise ValueError("Recorte online ausente")
                if sha256(candidate.read_bytes()).hexdigest() != item.get(part+"_sha256"):
                    raise ValueError("Hash online divergente")
                verified[part] = candidate
            identifier = "online:"+metadata.stem+":"+mode
            observed[mode] = identifier
            views[identifier] = verified
        if not observed:
            raise ValueError("Evento sem captura")
        ocr = data.get("aoi_info", {}) or {}
        clean = lambda text: "".join(
            c for c in str(text or "").upper() if c.isalnum()
        )
        board, part = clean(ocr.get("board")), clean(ocr.get("parts"))
        events.append({
            "id": "online:"+metadata.stem,
            "label": data["label"],
            "observations": observed,
            "split_key": (
                "component:"+board+"/"+part if board and part
                else "online:"+metadata.stem
            ),
        })
    return events, views


def run_online(root: Path, event_file: Path) -> dict:
    root = Path(root).resolve()
    journal = (root/"reports"/"neural_online"/"events").resolve()
    event_file = Path(event_file).resolve()
    if event_file.parent != journal or event_file.suffix != ".json":
        raise ValueError("Evento não pertence ao journal online")
    events, images = load_online_deslocado(root)
    if "online:"+event_file.stem not in {e["id"] for e in events}:
        raise ValueError("Evento DESLOCADO solicitado não está no journal")
    source = latest_deslocado_manifest(root)
    report, out = train_deslocado(
        source, epochs=3, size=160, batch_size=4,
        online_events=events, online_views=images
    )
    # Cada atualização gera um candidato. A ativação automática fica
    # interditada até NG REAL independente e qualificação da categoria.
    if report.get("production_approved") or not report.get("activation_disabled"):
        raise AssertionError("Falha no gate DESLOCADO de produção")
    print(
        "CNN DESLOCADO ONLINE: treinamento candidato concluído, "
        "sem promover checkpoint operacional."
    )
    print("Candidato:", out/"deslocado_cnn_candidate.pt")
    print("Relatório:", out/"training_report_deslocado.json")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Aprender DESLOCADO com novos rótulos humanos; sem ativar CNN."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--event", required=True, type=Path)
    args = parser.parse_args(argv)
    run_online(args.root, args.event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
