"""Retificação offline e auditável de julgamentos humanos incorretos.

A manutenção só opera com o ODIN produtivo encerrado. Nenhuma associação é
inferida por horário: exige captura XP pixel-identical, JSON e event_id reais.
Não retreina CNN, não envia comandos XP e não modifica um checkpoint.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4

import cv2

from src.config.settings import BASE_DIR
from src.core.anomaly_signature import build_anomaly_signature, valid_anomaly_signature
from src.core.archive_persistent_memory import (
    PersistentArchiveKNN, identity, make_operator_record, make_record,
)
from src.services.image_archive_dedup import image_fingerprint
from src.services.startup_regression.archive_inventory import filename_hints

LIGHTS = frozenset(("SIDE", "TOP", "MID"))
CATEGORIES = frozenset(("FALTANDO", "INVERTIDO"))
EXPECTED_IMAGES = 9


def _json(path: Path) -> dict:
    row = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise ValueError(f"JSON não é objeto: {path}")
    return row


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        tmp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _child(directory: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError(f"Nome de arquivo inseguro: {filename!r}")
    result = directory / filename
    if result.is_symlink():
        raise ValueError(f"Link simbólico não permitido: {result}")
    return result


def _image(path: Path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Imagem ausente/insegura: {path}")
    result = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if result is None or result.size == 0:
        raise ValueError(f"PNG não pode ser lido: {path}")
    return result


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _in_root(root: Path, file: Path) -> str:
    return file.relative_to(root).as_posix()


def _pairs_from_dataset(root: Path) -> list[dict]:
    result = []
    base = root / "public" / "dataset" / "nao_anomalia"
    if not base.is_dir():
        raise ValueError("Dataset nao_anomalia não encontrado")
    for json_path in sorted(base.rglob("*.json")):
        if json_path.is_symlink():
            raise ValueError(f"JSON com link simbólico: {json_path}")
        row = _json(json_path)
        storage = row.get("storage") or {}
        if not isinstance(storage, dict):
            continue
        source_name = str(storage.get("source_image_file") or "")
        if not source_name:
            continue
        source_path = _child(json_path.parent, source_name)
        if not source_path.is_file():
            raise ValueError(f"Imagem fonte referenciada ausente: {source_path}")
        result.append({
            "json_path": json_path, "data": row,
            "source_path": source_path,
            "source_fingerprint": image_fingerprint(_image(source_path)),
        })
    return result


def _online_events(root: Path, event_ids: set[str], by_event_mode: dict) -> list[dict]:
    events_path = root / "reports" / "neural_online" / "events"
    found = []
    if not events_path.is_dir():
        return found
    for file in sorted(events_path.glob("*.json")):
        if file.is_symlink():
            raise ValueError(f"Evento CNN é link simbólico: {file}")
        row = _json(file)
        event = str(row.get("source_aoi_event_id") or "")
        if event not in event_ids:
            continue
        if row.get("label") != "OK" or row.get("category") not in CATEGORIES:
            raise ValueError(f"Evento CNN possui rótulo/categoria inesperado: {file.name}")
        images = row.get("images") or []
        if not images or not isinstance(images, list):
            raise ValueError(f"Evento CNN sem imagens: {file.name}")
        for view in images:
            mode = str(view.get("lighting_mode") or "")
            target = by_event_mode.get((event, mode))
            if target is None:
                raise ValueError("Evento CNN contém iluminação sem vínculo confiável")
            for role, expected in (("reference", target["ref_hash"]), ("test", target["test_hash"])):
                img = _child(events_path, str(view.get(role) or ""))
                if not img.is_file() or _sha(img) != view.get(role + "_sha256"):
                    raise ValueError(f"Arquivo do journal CNN adulterado/ausente: {img}")
                if image_fingerprint(_image(img)) != expected:
                    raise ValueError("Frame CNN não corresponde ao dataset do mesmo evento")
        found.append({"path": file, "data": row})
    return found


def prepare_correction(root: Path | str, selected: list[str | Path]) -> dict:
    """Preflight rigoroso. Não altera NENHUM dado original."""
    root = Path(root).resolve()
    ok = root / "public" / "ok_archive"
    ng = root / "public" / "ng_archive"
    files = [Path(p).resolve() for p in selected]
    if len(files) != EXPECTED_IMAGES or len(set(files)) != EXPECTED_IMAGES:
        raise ValueError("Selecione exatamente 9 PNGs diferentes")
    if any(p.parent != ok.resolve() or p.suffix.lower() != ".png" for p in files):
        raise ValueError("Seleção deve conter somente PNGs de public/ok_archive")
    if not ng.is_dir():
        raise ValueError("Arquivo NG não encontrado")
    if any(not p.is_file() or p.is_symlink() or (ng / p.name).exists() for p in files):
        raise ValueError("Imagem ausente, link simbólico ou nome já existente em ng_archive")

    status_root = root / "reports" / "label_corrections"
    if status_root.is_dir():
        for record in status_root.glob("*/status.json"):
            state = _json(record).get("state")
            if state not in ("COMPLETED", "ROLLED_BACK"):
                raise ValueError(
                    "Correção anterior interrompida: faça auditoria/restauração do backup "
                    f"antes de continuar ({record.parent})"
                )

    latest = root / "reports" / "neural_online" / "latest_event.json"
    if latest.is_file() and (
        _json(latest).get("state") in ("TRAINING", "QUEUED")
        or _json(latest).get("status") in ("TRAINING", "QUEUED")
    ):
        raise ValueError("Treinamento CNN ainda está na fila/em execução. Aguarde seu término")

    selected_data = {}
    for p in files:
        hint = filename_hints(p)
        if (hint["lighting_source"] != "EXPLICIT_SUFFIX"
                or hint["lighting_mode"] not in LIGHTS
                or hint["category_hint"] not in CATEGORIES):
            raise ValueError(f"Imagem sem iluminação/categoria esperada: {p.name}")
        selected_data[image_fingerprint(_image(p))] = p
    if len(selected_data) != EXPECTED_IMAGES:
        raise ValueError("Duas capturas selecionadas possuem pixels idênticos: revisão manual necessária")

    # Não modificar arquivos participantes de manifestos históricos explícitos.
    for manifest in ok.glob("*.json"):
        data = _json(manifest)
        frames = data.get("frames") or {}
        if not isinstance(frames, dict):
            continue
        if any(Path(str(v.get("path") or "")).name in {p.name for p in files}
               for v in frames.values() if isinstance(v, dict)):
            raise ValueError(f"Manifesto multilight afetado: {manifest.name}; revisão específica necessária")

    available = _pairs_from_dataset(root)
    matches = defaultdict(list)
    for record in available:
        if record["source_fingerprint"] in selected_data:
            matches[record["source_fingerprint"]].append(record)
    items = []
    by_event_mode = {}
    for image_hash, archive_png in selected_data.items():
        associated = matches.get(image_hash, [])
        if len(associated) != 1:
            raise ValueError(
                f"{archive_png.name}: esperava 1 JSON com captura XP idêntica; "
                f"encontrados {len(associated)}. Não é seguro corrigir automaticamente."
            )
        row = associated[0]
        data = row["data"]
        info = data.get("aoi_info") or {}
        multi = data.get("multilight") or {}
        decision = data.get("decision") or {}
        storage = data.get("storage") or {}
        hint = filename_hints(archive_png)
        mode = hint["lighting_mode"]
        category = hint["category_hint"]
        event_id = str(multi.get("event_id") or "").strip()
        if (data.get("label") != "OK" or decision.get("operator_label") != "OK"
                or (data.get("analysis") or {}).get("operator_label") != "OK"
                or info.get("category") != category
                or info.get("lighting_mode") != mode
                or str(multi.get("lighting_mode") or "") != mode
                or not event_id):
            raise ValueError(f"Metadados inconsistentes no dataset: {row['json_path']}")
        if storage.get("visual_deduplicated") or storage.get("duplicate_visual_of_json"):
            raise ValueError("Dataset possui referência deduplicada; requer auditoria individual")
        reference = _child(row["json_path"].parent, str(storage.get("reference_image_file") or ""))
        test = _child(row["json_path"].parent, str(storage.get("test_image_file") or ""))
        for item in (reference, test):
            if not item.is_file():
                raise ValueError(f"Par de treino ausente: {item}")
        ref_image, test_image = _image(reference), _image(test)
        ref_hash, test_hash = image_fingerprint(ref_image), image_fingerprint(test_image)
        if test_hash != storage.get("test_image_fingerprint"):
            raise ValueError("Fingerprint do teste diferente do metadata")
        key = identity(category, mode, ref_hash, test_hash)
        slot = (event_id, mode)
        if slot in by_event_mode:
            raise ValueError("Duas imagens associadas ao mesmo event_id e iluminação")
        metadata_files = [row["json_path"], row["source_path"], reference, test]
        base_name = row["json_path"].stem
        if not base_name.startswith("memory_OK_"):
            raise ValueError("Dataset antigo não segue o formato memory_OK_")
        destinations = []
        target_folder = root / "public" / "dataset" / "anomalia" / row["json_path"].parent.name
        for old in metadata_files:
            name = old.name.replace(base_name, base_name.replace("memory_OK_", "memory_NG_", 1), 1)
            destination = _child(target_folder, name)
            if destination.exists():
                raise ValueError(f"Destino dataset já existe: {destination}")
            destinations.append(destination)
        item = {
            "archive": archive_png, "archive_dest": ng / archive_png.name,
            "archive_sha": _sha(archive_png), "record": row,
            "metadata_files": metadata_files, "dataset_destinations": destinations,
            "event_id": event_id, "category": category, "mode": mode, "key": key,
            "ref_image": ref_image, "test_image": test_image,
            "ref_hash": ref_hash, "test_hash": test_hash,
        }
        items.append(item)
        by_event_mode[slot] = item

    event_ids = {item["event_id"] for item in items}
    if len(event_ids) != 3 or any(
        {mode for event, mode in by_event_mode if event == event_id} != LIGHTS
        for event_id in event_ids
    ):
        raise ValueError("As 9 imagens não formam 3 eventos completos e distintos por event_id")
    if any(
        len({item["category"] for item in items if item["event_id"] == event}) != 1
        for event in event_ids
    ):
        raise ValueError("Iluminações do mesmo evento apresentam categorias diferentes")

    # Se outras memórias apontarem para uma das fontes ou para os mesmos
    # event_ids com identidade diferente, não editar parcialmente o registro.
    memory = PersistentArchiveKNN(root)
    rows = memory.records()
    keys = {tuple(item["key"]) for item in items}
    archived_paths = {"public/ok_archive/" + item["archive"].name for item in items}
    for rec in rows:
        for source in rec["sources"]:
            if (source.get("source_path") in archived_paths
                    or (source.get("source_kind") == "OPERATOR_CONFIRMED"
                        and source.get("event_id") in event_ids)):
                if tuple(rec["identity"]) not in keys:
                    raise ValueError("Memória humana associada à peça usa outra identidade visual")

    # Não criar links inválidos nas memórias deduplicadas de outros eventos.
    moved_json_paths = {_in_root(root, item["record"]["json_path"]) for item in items}
    for previous in available:
        if previous["json_path"] in {x["record"]["json_path"] for x in items}:
            continue
        linked = str((previous["data"].get("storage") or {}).get("duplicate_visual_of_json") or "")
        if linked and any(linked.endswith(Path(p).name) or linked == p for p in moved_json_paths):
            raise ValueError("Outro exemplo referencia uma memória OK que seria movida")

    journal = _online_events(root, event_ids, by_event_mode)
    return {
        "root": root, "items": sorted(items, key=lambda x: x["archive"].name),
        "events": event_ids, "online": journal,
        "memory": memory,
        "memories_ok": sum(rec["label"] == "OK" and tuple(rec["identity"]) in keys
                           for rec in rows),
        "memories_ng": sum(rec["label"] == "NG" and tuple(rec["identity"]) in keys
                           for rec in rows),
    }


def _repair_metadata(data: dict, original_stem: str, correction_id: str) -> dict:
    row = deepcopy(data)
    row["label"] = "NG"
    row["decision"]["operator_label"] = "NG"
    row["decision"]["disagreement"] = (
        str(row["decision"].get("ai_label") or "").upper() != "NG"
    )
    row["analysis"]["operator_label"] = "NG"
    for key in ("test_image_file", "reference_image_file", "source_image_file"):
        value = str(row["storage"].get(key) or "")
        if value:
            row["storage"][key] = value.replace(
                original_stem, original_stem.replace("memory_OK_", "memory_NG_", 1), 1
            )
    row["image_file"] = str(row.get("image_file") or "").replace(
        original_stem, original_stem.replace("memory_OK_", "memory_NG_", 1), 1
    )
    row["label_correction"] = {
        "from": "OK", "to": "NG", "reason": "Julgamento humano retificado",
        "correction_id": correction_id,
    }
    return row


def apply_correction(plan: dict) -> dict:
    """Transação offline. Backups por arquivo e rollback em falha tratável."""
    # Reexecutar todo o preflight imediatamente antes de qualquer gravação.
    current = prepare_correction(
        plan["root"], [item["archive"] for item in plan["items"]]
    )
    root = current["root"]
    transaction = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    folder = root / "reports" / "label_corrections" / transaction
    folder.mkdir(parents=True, exist_ok=False)
    status = folder / "status.json"

    pairs = []
    for item in current["items"]:
        pairs.append((item["archive"], item["archive_dest"], None))
        for src, dest in zip(item["metadata_files"], item["dataset_destinations"]):
            pairs.append((src, dest, item if src == item["record"]["json_path"] else None))
    for event in current["online"]:
        pairs.append((event["path"], event["path"], event))
    memory_path = current["memory"].path
    if not memory_path.is_file():
        raise ValueError("Snapshot de memória KNN não encontrado; retificação interrompida")

    originals = sorted({source for source, _, _ in pairs} | {memory_path})
    backups = {}
    changes = []
    try:
        for source in originals:
            relative = source.relative_to(root)
            snapshot = folder / "before" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, snapshot)
            if _sha(source) != _sha(snapshot):
                raise ValueError(f"Falha na verificação do backup: {source}")
            backups[source] = snapshot
        _write_json(status, {
            "schema": "visionx.label_correction.v1", "state": "PREPARED",
            "correction_id": transaction,
            "original_files": [_in_root(root, p) for p in originals],
            "targets": [str(item["archive"].name) for item in current["items"]],
        })
        # Antes do move, garantir que o destino de cada arquivo continua vazio.
        if any(src != dst and dst.exists() for src, dst, _ in pairs):
            raise ValueError("Outro processo criou arquivos de destino; interrompido")
        _write_json(status, {
            "schema": "visionx.label_correction.v1", "state": "APPLYING",
            "correction_id": transaction,
            "original_files": [_in_root(root, p) for p in originals],
        })
        for src, dest, metadata in pairs:
            if src != dest:
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(src, dest)
                changes.append((src, dest))
            if metadata is None:
                continue
            if "record" in metadata and src == metadata["record"]["json_path"]:
                replacement = _repair_metadata(metadata["record"]["data"], src.stem, transaction)
                _write_json(dest, replacement)
            elif "data" in metadata and "path" in metadata and src == metadata["path"]:
                corrected = deepcopy(metadata["data"])
                corrected["label"] = "NG"
                corrected["label_correction"] = {
                    "from": "OK", "to": "NG", "correction_id": transaction,
                }
                _write_json(dest, corrected)

        existing = current["memory"].records()
        records_by_key = defaultdict(list)
        for record in existing:
            records_by_key[tuple(record["identity"])].append(record)
        proposed = []
        for item in current["items"]:
            context = item["record"]["data"]["aoi_info"]
            previous = records_by_key.get(tuple(item["key"]), [])
            signature = next(
                (record["anomaly_signature"] for record in previous
                 if valid_anomaly_signature(record.get("anomaly_signature"))),
                None,
            )
            if signature is None:
                signature = build_anomaly_signature(
                    item["ref_image"], item["test_image"], {},
                    {**context, "lighting_mode": item["mode"]},
                )
            if not valid_anomaly_signature(signature):
                raise ValueError("Assinatura de memória não é válida")
            proposal = make_operator_record(
                label="NG", source="human", event_id=item["event_id"],
                category=item["category"], lighting=item["mode"],
                reference=item["ref_image"], test=item["test_image"],
                anomaly_signature=signature, aoi_info=context,
            )
            proposal["sources"][0]["correction_id"] = transaction
            proposed.append(proposal)
            proposed.append(make_record(
                label="NG", source_path="public/ng_archive/" + item["archive"].name,
                source_sha256=item["archive_sha"], category=item["category"],
                lighting=item["mode"], reference=item["ref_image"],
                test=item["test_image"], anomaly_signature=signature,
                aoi_info=context,
            ))
        memory_result = current["memory"].replace_verified_labels(proposed)
        for item in current["items"]:
            if current["memory"].lookup_identity(item["key"]).get("label") != "NG":
                raise ValueError("Memória corrigida não confirma NG")
            if not item["archive_dest"].is_file() or _sha(item["archive_dest"]) != item["archive_sha"]:
                raise ValueError("PNG NG final ausente ou diferente")
        final = {
            "schema": "visionx.label_correction.v1", "state": "COMPLETED",
            "correction_id": transaction,
            "images_corrected": len(current["items"]),
            "events_corrected": len(current["events"]),
            "dataset_records_corrected": len(current["items"]),
            "cnn_online_journal_corrected": len(current["online"]),
            "knn": memory_result,
            "backup_folder": str(folder / "before"),
            "cnn_weights_modified": False,
            "cnn_checkpoint_needs_audit": True,
        }
        _write_json(status, final)
        return final
    except Exception as exc:
        # Restaurar origens (inclusive JSON editados in-place) a partir do
        # backup em vez de depender do estado de arquivos movidos parcialmente.
        rollback_ok = True
        try:
            for src, dest in reversed(changes):
                if dest.is_file():
                    dest.unlink()
            for src, snapshot in backups.items():
                src.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot, src)
                if _sha(src) != _sha(snapshot):
                    raise ValueError(f"Rollback não confirmou SHA de {src}")
        except Exception:
            rollback_ok = False
        _write_json(status, {
            "schema": "visionx.label_correction.v1",
            "state": "ROLLED_BACK" if rollback_ok else "NEEDS_RECOVERY",
            "error": str(exc), "correction_id": transaction,
            "backup_folder": str(folder / "before"),
        })
        raise RuntimeError(
            f"Correção não concluída: {exc}. "
            + ("Backup restaurado." if rollback_ok else
               f"RESTABELEÇA O BACKUP MANUALMENTE em {folder / 'before'} antes de iniciar o ODIN.")
        ) from exc


__all__ = ["prepare_correction", "apply_correction"]
