"""Inventário somente leitura das evidências históricas OK/NG.

Etapa 1 do ODIN Startup Regression Gate. Não executa OCR, especialistas,
treinamento ou comandos à AOI. Não cria diretórios dentro dos arquivos
visuais e nunca deduz event_id por proximidade de nome ou data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

import cv2
import numpy as np

from src.services.image_archive_dedup import image_fingerprint
from src.utils.text_normalizer import ALIASES, CATEGORIES

SCHEMA = "visionx.archive_inventory.v1"
MANIFEST_SCHEMA = "visionx.archive_regression.v1"
LIGHTING_ORDER = ("SIDE", "TOP", "MID")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PNG_BYTES = 128 * 1024 * 1024
MAX_PIXELS = 80_000_000
SYSTEM_FILE_NAMES = {"desktop.ini", "thumbs.db", ".ds_store"}
LIGHTING_SUFFIX = re.compile(r"_(SIDE|TOP|MID)$", re.IGNORECASE)
COPY_SUFFIX = re.compile(r"_(?:[2-9]|[1-9][0-9]+)$")
CATEGORY_TERMS = sorted(
    {
        (re.sub(r"[^A-Z0-9]+", "_", key.upper()).strip("_"), val)
        for key, val in ALIASES.items()
    }
    | {
        (re.sub(r"[^A-Z0-9]+", "_", key.upper()).strip("_"), key)
        for key in CATEGORIES
    },
    key=lambda pair: len(pair[0]),
    reverse=True,
)


def filename_hints(path: Path) -> dict[str, str]:
    """Retorna apenas *pistas* de nome, nunca OCR nem identidade de evento."""
    stem = path.stem.upper()
    without_copy = COPY_SUFFIX.sub("", stem)
    lighting_match = LIGHTING_SUFFIX.search(without_copy)
    lighting = lighting_match.group(1) if lighting_match else "SIDE"
    lighting_source = "EXPLICIT_SUFFIX" if lighting_match else "LEGACY_DEFAULT"
    without_light = (
        without_copy[: lighting_match.start()]
        if lighting_match
        else without_copy
    )

    category = "UNKNOWN"
    for token, canonical in CATEGORY_TERMS:
        if without_light == token or without_light.endswith("_" + token):
            category = canonical
            break

    return {
        "lighting_mode": lighting,
        "lighting_source": lighting_source,
        "category_hint": category,
        "category_source": "FILENAME_HINT" if category != "UNKNOWN" else "UNRESOLVED",
    }


def _issue(kind: str, path: str, detail: str) -> dict[str, str]:
    return {"code": kind, "path": path, "detail": detail}


def _hash_file(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inspect_png(path: Path, root: Path, label: str) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    result: dict[str, Any] = {
        "path": relative,
        "expected_label": label,
        **filename_hints(path),
        "format": "PNG",
        "status": "INVALID_PNG",
        "size_bytes": None,
        "file_sha256": "",
        "pixel_sha256": "",
        "width": None,
        "height": None,
        "channels": None,
        "event_id": None,
        "manifest_path": None,
        "ocr_status": "NOT_EVALUATED_STAGE_1",
        "decision_status": "NOT_EVALUATED_STAGE_1",
        "issues": [],
    }
    try:
        stat = path.stat()
        result["size_bytes"] = stat.st_size
        if stat.st_size > MAX_PNG_BYTES:
            raise ValueError(f"PNG maior que limite de {MAX_PNG_BYTES} bytes")
        contents = path.read_bytes()
        result["file_sha256"] = _hash_file(contents)
        if not contents.startswith(PNG_SIGNATURE):
            raise ValueError("Assinatura de PNG ausente ou incorreta")
        if len(contents) < 24 or contents[12:16] != b"IHDR":
            raise ValueError("Cabeçalho IHDR ausente")
        width = int.from_bytes(contents[16:20], "big")
        height = int.from_bytes(contents[20:24], "big")
        if not width or not height or width * height > MAX_PIXELS:
            raise ValueError("Dimensão PNG inválida ou acima do limite")
        # imdecode recebe bytes e funciona com caminhos Unicode no Windows.
        image = cv2.imdecode(
            np.frombuffer(contents, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if image is None or image.size == 0:
            raise ValueError("OpenCV não conseguiu decodificar a imagem")
        if (int(image.shape[1]), int(image.shape[0])) != (width, height):
            raise ValueError("Dimensões declaradas diferem da decodificação")
        result.update({
            "status": "VALID_PNG",
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "channels": int(image.shape[2]) if image.ndim == 3 else 1,
            "pixel_sha256": image_fingerprint(image),
        })
    except (OSError, ValueError, cv2.error) as exc:
        result["issues"].append(
            _issue("INVALID_PNG", relative, f"{type(exc).__name__}: {exc}")
        )
    return result


def _safe_manifest_target(archive: Path, value: str) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute() or "\\" in value:
        return None
    try:
        resolved = (archive / candidate).resolve()
        resolved.relative_to(archive.resolve())
        return resolved
    except (ValueError, OSError):
        return None


def _read_manifest(
    path: Path,
    archive: Path,
    root: Path,
    label: str,
    records: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    relative = path.relative_to(root).as_posix()
    manifest: dict[str, Any] = {
        "path": relative,
        "status": "INVALID_MANIFEST",
        "event_id": None,
        "expected_label": None,
        "frame_paths": {},
        "issues": [],
    }
    issues = manifest["issues"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Manifesto JSON não é objeto")
        if payload.get("schema") != MANIFEST_SCHEMA:
            raise ValueError("Schema desconhecido; não inferir vínculo")
        event_id = str(payload.get("event_id", "") or "").strip()
        expected_label = str(payload.get("expected_label", "") or "").upper()
        frames = payload.get("frames")
        if not event_id or expected_label != label or not isinstance(frames, dict):
            raise ValueError("event_id/rótulo/frames incompletos ou conflitantes")
        if set(frames) != set(LIGHTING_ORDER):
            raise ValueError("Evento multilight requer SIDE/TOP/MID completos")
        manifest["event_id"] = event_id
        manifest["expected_label"] = expected_label
        aoi_info = payload.get("aoi_info")
        if not isinstance(aoi_info, dict):
            issues.append(_issue("MANIFEST_MISSING_OCR", relative, "Sem aoi_info confiável"))
        for mode in LIGHTING_ORDER:
            frame = frames.get(mode)
            if not isinstance(frame, dict):
                issues.append(_issue("MANIFEST_FRAME_INVALID", relative, mode))
                continue
            target = _safe_manifest_target(archive, frame.get("path"))
            if target is None:
                issues.append(_issue("MANIFEST_UNSAFE_PATH", relative, mode))
                continue
            try:
                frame_relative = target.relative_to(root.resolve()).as_posix()
            except ValueError:
                issues.append(_issue("MANIFEST_UNSAFE_PATH", relative, mode))
                continue
            manifest["frame_paths"][mode] = frame_relative
            current = records.get(frame_relative)
            if current is None:
                issues.append(
                    _issue("MANIFEST_MISSING_FRAME", relative, f"{mode}: {frame_relative}")
                )
                continue
            if current["status"] != "VALID_PNG":
                issues.append(_issue("MANIFEST_INVALID_FRAME", relative, frame_relative))
                continue
            if current["expected_label"] != label:
                issues.append(_issue("MANIFEST_LABEL_CONFLICT", relative, frame_relative))
            if (
                current["lighting_source"] == "EXPLICIT_SUFFIX"
                and current["lighting_mode"] != mode
            ):
                issues.append(_issue("MANIFEST_LIGHT_CONFLICT", relative, frame_relative))
            provided_hash = str(frame.get("sha256", "") or "").lower()
            if not re.fullmatch(r"[0-9a-f]{64}", provided_hash):
                issues.append(_issue("MANIFEST_MISSING_HASH", relative, mode))
            elif provided_hash != current["file_sha256"]:
                issues.append(_issue("MANIFEST_HASH_MISMATCH", relative, frame_relative))
            if current.get("manifest_path") is not None:
                issues.append(
                    _issue("FRAME_CLAIMED_TWICE", relative, frame_relative)
                )
            else:
                current["manifest_path"] = relative
                current["event_id"] = event_id
                # O manifesto é a origem explícita da iluminação.
                current["lighting_mode"] = mode
                current["lighting_source"] = "MANIFEST"

        if not issues:
            manifest["status"] = "LINKED"
        else:
            manifest["status"] = "NEEDS_QUALIFICATION"
    except (ValueError, OSError, UnicodeError) as exc:
        issues.append(
            _issue("INVALID_MANIFEST", relative, f"{type(exc).__name__}: {exc}")
        )
    return manifest, issues


def inventory_archives(
    root: Path,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Inspeciona arquivo por arquivo, sem modificar nenhum arquivo original.

    Não cria diretórios, não usa OCR, não simula decisão da IA e não
    estabelece vínculos multilight sem manifesto real.
    """
    root = Path(root).expanduser().resolve()
    roots = {"OK": root / "public" / "ok_archive",
             "NG": root / "public" / "ng_archive"}
    issues: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    manifest_paths: list[tuple[Path, Path, str]] = []
    non_png_files: list[dict[str, str]] = []
    png_paths: list[tuple[Path, str]] = []

    for label, archive in roots.items():
        if not archive.is_dir():
            issues.append(
                _issue("MISSING_ARCHIVE_DIR", str(archive), "Pasta não encontrada")
            )
            continue
        try:
            entries = sorted(
                (p for p in archive.rglob("*") if p.is_file()),
                key=lambda p: str(p).casefold(),
            )
        except OSError as exc:
            issues.append(_issue("DIRECTORY_READ_ERROR", str(archive), str(exc)))
            continue
        for path in entries:
            suffix = path.suffix.lower()
            if suffix == ".png":
                png_paths.append((path, label))
            elif suffix == ".json":
                manifest_paths.append((path, archive, label))
            elif path.name.lower() in SYSTEM_FILE_NAMES:
                non_png_files.append({
                    "path": path.relative_to(root).as_posix(),
                    "classification": "KNOWN_SYSTEM_FILE",
                })
            else:
                relative = path.relative_to(root).as_posix()
                non_png_files.append({
                    "path": relative,
                    "classification": "UNSUPPORTED_FILE",
                })
                issues.append(
                    _issue("UNSUPPORTED_FILE", relative, "Arquivo não PNG/manifesto")
                )

    total = len(png_paths)
    for number, (path, label) in enumerate(png_paths, start=1):
        record = _inspect_png(path, root, label)
        records.append(record)
        issues.extend(record["issues"])
        if record["category_hint"] == "UNKNOWN":
            issues.append(
                _issue("UNKNOWN_CATEGORY_HINT", record["path"],
                       "Categoria não identificável pelo nome; OCR será Etapa 2")
            )
        if progress:
            progress(number, total, record["path"])

    by_path = {r["path"]: r for r in records}
    manifests = []
    event_ids: dict[str, str] = {}
    for path, archive, label in manifest_paths:
        manifest, manifest_issues = _read_manifest(
            path, archive, root, label, by_path
        )
        manifests.append(manifest)
        issues.extend(manifest_issues)
        event_id = manifest.get("event_id")
        if event_id:
            if event_id in event_ids:
                issues.append(
                    _issue(
                        "DUPLICATE_EVENT_ID", manifest["path"],
                        f"Já aparece em {event_ids[event_id]}",
                    )
                )
            else:
                event_ids[event_id] = manifest["path"]

    cross_label: dict[str, set[str]] = defaultdict(set)
    visual_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        fingerprint = record.get("pixel_sha256")
        if fingerprint:
            cross_label[fingerprint].add(record["expected_label"])
            visual_groups[fingerprint].append(record["path"])
        if (
            record["status"] == "VALID_PNG"
            and record["lighting_source"] == "EXPLICIT_SUFFIX"
            and not record.get("manifest_path")
        ):
            issues.append(
                _issue(
                    "MULTILIGHT_WITHOUT_MANIFEST", record["path"],
                    "Iluminação explícita, mas sem associação confiável por event_id",
                )
            )

    conflicts = [
        {
            "pixel_sha256": fp,
            "paths": visual_groups[fp],
            "labels": ["NG", "OK"],
        }
        for fp, labels in sorted(cross_label.items())
        if len(labels) > 1
    ]
    for group in conflicts:
        issues.append(
            _issue(
                "CROSS_LABEL_PIXEL_CONFLICT",
                ", ".join(group["paths"]),
                "Conteúdo pixel a pixel idêntico nas pastas OK e NG",
            )
        )

    duplicates = [
        {"pixel_sha256": fp, "paths": paths}
        for fp, paths in sorted(visual_groups.items())
        if len(paths) > 1
    ]
    by_label = Counter(r["expected_label"] for r in records)
    by_light = Counter(r["lighting_mode"] for r in records)
    by_category = Counter(r["category_hint"] for r in records)
    valid = sum(r["status"] == "VALID_PNG" for r in records)
    if not records:
        issues.append(
            _issue("NO_COVERAGE", str(root / "public"), "Nenhuma imagem PNG encontrada")
        )

    linked = sum(m["status"] == "LINKED" for m in manifests)
    summary = {
        "png_count": len(records),
        "valid_png": valid,
        "invalid_png": len(records) - valid,
        "by_label": dict(sorted(by_label.items())),
        "by_lighting": dict(sorted(by_light.items())),
        "by_category_hint": dict(sorted(by_category.items())),
        "legacy_side_count": sum(
            r["lighting_source"] == "LEGACY_DEFAULT" for r in records
        ),
        "explicit_unlinked_count": sum(
            r["lighting_source"] == "EXPLICIT_SUFFIX"
            and not r["manifest_path"] for r in records
        ),
        "manifest_linked_png_count": sum(
            bool(r["manifest_path"]) for r in records
        ),
        "manifest_count": len(manifests),
        "linked_event_count": linked,
        "pixel_duplicate_groups": len(duplicates),
        "cross_label_conflict_groups": len(conflicts),
        "unsupported_file_count": sum(
            r["classification"] == "UNSUPPORTED_FILE" for r in non_png_files
        ),
        "issue_count": len(issues),
    }
    return {
        "schema": SCHEMA,
        "phase": "STAGE_1_INVENTORY_ONLY",
        "is_operational_gate": False,
        "analysis_performed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "archive_directories": {k: str(v) for k, v in roots.items()},
        "status": "NEEDS_QUALIFICATION" if issues else "INVENTORIED",
        "summary": summary,
        "images": records,
        "manifests": manifests,
        "duplicate_visual_groups": duplicates,
        "cross_label_conflicts": conflicts,
        "other_files": non_png_files,
        "issues": issues,
    }


__all__ = [
    "MANIFEST_SCHEMA",
    "SCHEMA",
    "filename_hints",
    "inventory_archives",
]
