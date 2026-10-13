"""Pré-validação offline de recuperação CNN após retificação humana OK -> NG.

Somente leitura. Requer 3 eventos NG de 3 iluminações em journals com
correção COMPLETED, pares íntegros, 9 JSONs do dataset com mesmo event_id
e pixels de gabarito/teste conferidos. Nenhum vínculo por horário/nome.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

import cv2

from src.services.image_archive_dedup import image_fingerprint

LIGHTS = frozenset(("SIDE", "TOP", "MID"))
CATEGORIES = frozenset(("FALTANDO", "INVERTIDO"))
CORRECTION_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Arquivo de auditoria ausente/inseguro: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Auditoria sem objeto JSON: {path}")
    return data


def _file(parent: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name or "\\" in name or "/" in name:
        raise ValueError(f"Nome de imagem inválido: {name!r}")
    path = parent / name
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".png":
        raise ValueError(f"Imagem ausente/insegura: {path}")
    return path


def _decoded_hash(path: Path) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or min(img.shape[:2]) < 12:
        raise ValueError(f"Imagem de treino ilegível: {path}")
    return image_fingerprint(img)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def latest_completed_correction(root: Path | str) -> str:
    base = Path(root).resolve() / "reports" / "label_corrections"
    if not base.is_dir():
        raise ValueError("Não existe uma correção humana registrada")
    completed = []
    for folder in base.iterdir():
        if not folder.is_dir() or not CORRECTION_ID.fullmatch(folder.name):
            continue
        try:
            data = _read_json(folder / "status.json")
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        if (data.get("schema") == "visionx.label_correction.v1"
                and data.get("state") == "COMPLETED"
                and data.get("images_corrected") == 9
                and data.get("events_corrected") == 3
                and data.get("cnn_online_journal_corrected") == 3):
            completed.append(folder.name)
    if not completed:
        raise ValueError("Nenhuma correção completa de 9 imagens/3 eventos foi encontrada")
    return max(completed)


def validate_recovery_inputs(root: Path | str, correction_id: str) -> dict:
    root = Path(root).resolve()
    if not CORRECTION_ID.fullmatch(correction_id):
        raise ValueError("Identificador de correção inválido")
    from src.services.archive_label_correction import unfinished_corrections
    pending = unfinished_corrections(root)
    if pending:
        raise ValueError(f"Transação incompleta. Restaurar antes do treino: {pending[0]}")
    status = _read_json(
        root / "reports" / "label_corrections" / correction_id / "status.json"
    )
    if (status.get("schema") != "visionx.label_correction.v1"
            or status.get("state") != "COMPLETED"
            or status.get("correction_id") != correction_id
            or status.get("images_corrected") != 9
            or status.get("events_corrected") != 3
            or status.get("dataset_records_corrected") != 9
            or status.get("cnn_online_journal_corrected") != 3):
        raise ValueError("Retificação não atende ao contrato de 9 imagens/3 eventos")

    journal_dir = root / "reports" / "neural_online" / "events"
    if not journal_dir.is_dir():
        raise ValueError("Journal online ausente")
    matches = []
    for path in sorted(journal_dir.glob("*.json")):
        row = _read_json(path)
        if (row.get("label_correction") or {}).get("correction_id") != correction_id:
            continue
        if (row.get("label_correction") or {}).get("from") != "OK":
            raise ValueError("Histórico online sem proveniência OK original")
        if (row.get("label_correction") or {}).get("to") != "NG":
            raise ValueError("Histórico online não confirma retificação para NG")
        if (row.get("schema") != "visionx.specialist_online_case.v1"
                or row.get("event_id") != path.stem
                or row.get("label") != "NG"
                or row.get("category") not in CATEGORIES
                or row.get("training_requested") is not True):
            raise ValueError(f"Evento online inconsistente: {path.name}")
        source_event = str(row.get("source_aoi_event_id") or "").strip()
        if not source_event:
            raise ValueError("Evento de origem XP indisponível")
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 3:
            raise ValueError("Evento online não possui 3 iluminações")
        modes = [i.get("lighting_mode") for i in images if isinstance(i, dict)]
        if set(modes) != LIGHTS or len(modes) != 3:
            raise ValueError("Iluminações online não correspondem a SIDE/TOP/MID")
        pairs = {}
        for light in images:
            mode = light["lighting_mode"]
            pairs[mode] = {}
            for role in ("reference", "test"):
                asset = _file(journal_dir, light.get(role))
                if _file_hash(asset) != light.get(role + "_sha256"):
                    raise ValueError(f"SHA-256 do journal alterado: {asset}")
                pairs[mode][role] = _decoded_hash(asset)
        matches.append({
            "path": path, "event_id": source_event,
            "category": row["category"], "pairs": pairs,
        })
    if len(matches) != 3 or len({x["event_id"] for x in matches}) != 3:
        raise ValueError(
            "Esperados exatamente 3 journals distintos vinculados ao julgamento corrigido"
        )

    # Exige 9 registros efetivamente transferidos para anomalia; rótulo
    # coerente, mesmo event_id, mesma luz e pixels iguais aos do journal CNN.
    dataset = root / "public" / "dataset" / "anomalia"
    found = defaultdict(list)
    if not dataset.is_dir():
        raise ValueError("Dataset de anomalias ausente")
    target_events = {x["event_id"] for x in matches}
    for path in dataset.rglob("*.json"):
        data = _read_json(path)
        if (data.get("label_correction") or {}).get("correction_id") != correction_id:
            continue
        info = data.get("aoi_info") or {}
        light = str(info.get("lighting_mode") or "")
        event = str((data.get("multilight") or {}).get("event_id") or "")
        if (data.get("label") != "NG"
                or (data.get("decision") or {}).get("operator_label") != "NG"
                or event not in target_events or light not in LIGHTS):
            raise ValueError(f"Dataset corrigido sem rótulo/evento/luz válida: {path}")
        storage = data.get("storage") or {}
        assets = {
            role: _file(path.parent, storage.get(role + "_image_file"))
            for role in ("reference", "test", "source")
        }
        found[(event, light)].append({
            "category": info.get("category"),
            "reference": _decoded_hash(assets["reference"]),
            "test": _decoded_hash(assets["test"]),
        })
    if len(found) != 9 or any(len(v) != 1 for v in found.values()):
        raise ValueError("Dataset corrigido não contém exatamente 9 pares únicos")
    for match in matches:
        for mode in LIGHTS:
            sample = found.get((match["event_id"], mode))
            if not sample:
                raise ValueError(f"Sem dataset para {match['event_id']}/{mode}")
            row = sample[0]
            if (row["category"] != match["category"]
                    or row["reference"] != match["pairs"][mode]["reference"]
                    or row["test"] != match["pairs"][mode]["test"]):
                raise ValueError(
                    f"Pixels gabarito/teste diferem entre journal e dataset: "
                    f"{match['event_id']}/{mode}"
                )
    return {
        "correction_id": correction_id,
        "events": [match["path"] for match in matches],
        "event_ids": [match["event_id"] for match in matches],
        "categories": [match["category"] for match in matches],
        "dataset_pairs": len(found),
        "lighting": ["SIDE", "TOP", "MID"],
        "mode": "CANDIDATE_ONLY_OFFLINE",
        "training_authorized_for_production": False,
    }


__all__ = ["latest_completed_correction", "validate_recovery_inputs"]
