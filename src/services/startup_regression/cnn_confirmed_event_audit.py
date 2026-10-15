"""Auditoria SOMENTE LEITURA da aprendizagem CNN de um julgamento AOI humano.

Não treina, não altera pesos, não reclassifica rótulos e não envia 0/1 ao XP.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path


SCHEMA = "visionx.cnn_confirmed_event_audit.v1"
LIGHTS = {"SIDE", "TOP", "MID"}


def _json_if_present(path: Path) -> dict | None:
    if not path.is_file() or path.is_symlink():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Relatório JSON não é objeto: " + str(path))
    return data


def _asset_check(folder: Path, filename: str, expected: str) -> dict:
    if (not isinstance(filename, str) or not filename or
            Path(filename).name != filename or
            not filename.endswith(".png")):
        return {"valid": False, "reason": "INVALID_PATH"}
    target = folder / filename
    if not target.is_file() or target.is_symlink():
        return {"valid": False, "reason": "MISSING_FILE"}
    current = sha256(target.read_bytes()).hexdigest()
    return {
        "valid": current == expected,
        "reason": "SHA256_OK" if current == expected else "SHA256_MISMATCH",
    }


def audit_confirmed_event(root: Path, aoi_event_id: str) -> dict:
    """Verifica que uma confirmação chegou ao journal e seu treino terminou.

    Não declarar "aprendido" apenas porque um JSON de rótulo humano existe.
    O checkpoint pode seguir idêntico quando gates reprovaram o candidato.
    """
    if not isinstance(aoi_event_id, str) or not aoi_event_id.strip():
        raise ValueError("Informe o event_id real da inspeção AOI")
    journal = Path(root) / "reports" / "neural_online"
    folder = journal / "events"
    matches = []
    for path in sorted(folder.glob("*.json")):
        if path.is_symlink():
            raise ValueError("Journal online contém link simbólico")
        data = _json_if_present(path)
        if data is None or data.get("source_aoi_event_id") != aoi_event_id:
            continue
        if data.get("schema") != "visionx.specialist_online_case.v1":
            raise ValueError("Evento correspondente com schema incompatível")
        if path.stem != data.get("event_id"):
            raise ValueError("Evento correspondente sem identidade consistente")
        if data.get("label") not in {"OK", "NG"}:
            raise ValueError("Rótulo de evento online inválido")
        source = str(data.get("human_source", "")).lower()
        if source not in {
            "button", "xp_keyboard", "keyboard", "manual",
            "operator", "physical_keyboard", "human",
        } and not source.startswith("operator_"):
            raise ValueError("Evento correspondente sem fonte humana")
        views = {}
        for image in data.get("images", []):
            mode = image.get("lighting_mode")
            if mode not in LIGHTS or mode in views:
                raise ValueError("Iluminação duplicada ou inválida no journal")
            views[mode] = {
                kind: _asset_check(folder, image.get(kind),
                                   image.get(kind + "_sha256"))
                for kind in ("reference", "test")
            }
        status = _json_if_present(journal / "statuses" / path.name)
        outcome = _json_if_present(journal / "outcomes" / path.name)
        checkpoint = _json_if_present(journal / "live_active.json")
        matches.append({
            "journal_event_id": path.stem,
            "category": data.get("category"),
            "label": data["label"],
            "human_source": source,
            "recognition_route": data.get("recognition_route"),
            "views": views,
            "assets_verified": bool(views) and all(
                check["valid"] for assets in views.values()
                for check in assets.values()
            ),
            "job_status": (status or {}).get("status", "PENDING"),
            "exit_code": (status or {}).get("exit_code"),
            "promoted": (outcome or {}).get("promoted"),
            "validation_report": (outcome or {}).get("validation_report"),
            "checkpoint_active_for_event": bool(
                checkpoint and checkpoint.get("event_id") == path.stem
            ),
        })
    return {
        "schema": SCHEMA,
        "source_aoi_event_id": aoi_event_id,
        "matching_journals": len(matches),
        "state": (
            "NO_JOURNAL" if not matches
            else "MULTIPLE_JOURNALS" if len(matches) > 1
            else "JOURNAL_FOUND"
        ),
        "events": matches,
        "read_only": True,
        "note": (
            "JOURNAL_FOUND significa capturas supervisionadas preservadas, "
            "não que os pesos foram atualizados nem que a CNN reconheça o NG."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Auditoria local do NG confirmado CNN")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--aoi-event-id", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = audit_confirmed_event(args.root, args.aoi_event_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = args.output or (
        args.root / "reports" / "startup_regression" /
        ("cnn_confirmed_audit_" + stamp + ".json")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("Auditoria CNN:", target)
    print("Estado:", result["state"])
    for event in result["events"]:
        print("Rótulo:", event["label"], "Luzes:", sorted(event["views"]),
              "Integridade:", event["assets_verified"],
              "Treino:", event["job_status"], "Promovido:", event["promoted"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
