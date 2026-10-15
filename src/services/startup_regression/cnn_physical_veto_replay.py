"""Replay OFFLINE da contraprova física com pares supervisionados, sem CNN/XP.

Lê os PNGs preservados no journal online, confere seus SHA-256, aplica a
mesma sonda utilizada no consenso final e grava relatório de diagnóstico.
Não altera label, dataset, KNN, checkpoint, thresholds ou resultado de linha.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from src.core.physical_missing_veto import (
    LIGHTS, POLICY, physical_absence_probe,
)
from src.services.startup_regression.cnn_confirmed_event_audit import (
    audit_confirmed_event,
)


def _load_verified_image(folder: Path, name: str, digest: str):
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.lower().endswith(".png")
    ):
        raise ValueError("Caminho de PNG inválido")
    path = folder / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("PNG do evento indisponível")
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError("Integridade PNG divergente")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("PNG ilegível")
    return frame


def replay_event(root: Path, aoi_event_id: str) -> dict:
    """Vincula o julgamento humano ao evento AOI; nunca usa nome/data aproximados."""
    root = Path(root).resolve()
    audit = audit_confirmed_event(root, aoi_event_id)
    result = {
        "schema": "visionx.cnn_physical_veto_replay.v1",
        "policy": POLICY,
        "source_aoi_event_id": aoi_event_id,
        "state": audit["state"],
        "read_only": True,
        "events": [],
    }
    for row in audit["events"]:
        event_path = root / "reports" / "neural_online" / "events" / (
            row["journal_event_id"] + ".json"
        )
        event = json.loads(event_path.read_text(encoding="utf-8"))
        lights = {}
        for item in event.get("images", []):
            light = str(item.get("lighting_mode", ""))
            if light not in LIGHTS or light in lights:
                raise ValueError("Iluminação inválida ou duplicada")
            try:
                reference = _load_verified_image(
                    event_path.parent, item.get("reference", ""),
                    item.get("reference_sha256", ""),
                )
                test = _load_verified_image(
                    event_path.parent, item.get("test", ""),
                    item.get("test_sha256", ""),
                )
                lights[light] = physical_absence_probe(reference, test)
            except (OSError, ValueError, cv2.error, TypeError) as exc:
                lights[light] = {
                    "status": "UNAVAILABLE", "reason": type(exc).__name__, "scales": [],
                }
        for light in LIGHTS:
            lights.setdefault(light, {
                "status": "UNAVAILABLE", "reason": "MISSING_LIGHT_PAIR", "scales": [],
            })
        suspected = [
            light for light in LIGHTS
            if lights[light]["status"] == "SUSPECT_ABSENCE"
        ]
        unavailable = [
            light for light in LIGHTS
            if lights[light]["status"] == "UNAVAILABLE"
        ]
        result["events"].append({
            "journal_event_id": row["journal_event_id"],
            "human_label": row["label"],
            "category": row["category"],
            "lights": lights,
            "suspect_lights": suspected,
            "unavailable_lights": unavailable,
            "would_veto_cnn_all_ok": bool(suspected or unavailable),
            "note": (
                "Hipótese contrafactual: somente se a CNN tivesse votado OK "
                "em todas as luzes, NÃO é uma nova inferência CNN."
            ),
        })
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audita o veto de ausência FALTANDO em um evento confirmado"
    )
    parser.add_argument("--aoi-event-id", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = replay_event(args.root, args.aoi_event_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = args.output or (
        args.root / "reports" / "startup_regression" /
        ("cnn_physical_veto_" + stamp + ".json")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("Auditoria física CNN:", str(target))
    print("Estado:", result["state"], "eventos:", len(result["events"]))
    for event in result["events"]:
        print(event["human_label"], "suspeitas:", event["suspect_lights"],
              "indisponíveis:", event["unavailable_lights"],
              "veto:", event["would_veto_cnn_all_ok"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
