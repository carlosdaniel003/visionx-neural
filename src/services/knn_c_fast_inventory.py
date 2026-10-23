"""Inventário rápido para KNN C: validar SHA antes de decodificar PNG conhecido.

A memória C usa APENAS arquivos PNG e luz; não lê manifestos nem rótulos
da AOI como autoridade. Para cada arquivo, verifica SHA-256 real dos bytes
ANTES de reutilizar assinatura validada de cache. Se o hash mudou, executa
a validação completa do PNG, inclusive CRC/decodificação/pixel hash.
Nunca reutiliza cache após mudança de rótulo, caminho ou luz.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Callable

from src.services.startup_regression.archive_inventory import (
    SCHEMA, MAX_PNG_BYTES, _inspect_png, filename_hints,
)

MAX_IMAGES = 2000


def inventory_for_knn_c(
    root: Path, cache: dict, *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    root = Path(root).resolve()
    files = []
    for label, folder in (("OK", root / "public/ok_archive"),
                          ("NG", root / "public/ng_archive")):
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.suffix.lower() == ".png" and (path.is_file() or path.is_symlink()):
                files.append((path, label))
    files.sort(key=lambda row: str(row[0]).casefold())
    if len(files) > MAX_IMAGES:
        raise ValueError("Mais de 2000 PNGs na KNN C")
    items = []
    reused = 0
    by_digest = defaultdict(lambda: {"labels": set(), "paths": []})
    for number, (path, label) in enumerate(files, 1):
        relative = path.relative_to(root).as_posix()
        hints = filename_hints(path)
        stored = cache.get(relative)
        valid_cached = (
            isinstance(stored, dict)
            and stored.get("source_path") == relative
            and stored.get("expected_label") == label
            and stored.get("lighting_mode") == hints["lighting_mode"]
        )
        if path.is_symlink():
            item = {"path": relative, "expected_label": label, **hints,
                    "status": "INVALID_PNG", "pixel_sha256": "",
                    "file_sha256": "", "issues": [
                        {"code": "SYMLINK", "detail": "Link simbólico ignorado"}]}
        else:
            try:
                before = path.stat()
                if before.st_size > MAX_PNG_BYTES:
                    raise ValueError("PNG excede 128 MiB")
                # SHA é conferido EM TODO INÍCIO de sessão, independentemente
                # da data ou mtime. Evita confiar em caches desatualizados.
                digest = sha256(path.read_bytes()).hexdigest()
                after = path.stat()
                if ((before.st_size, before.st_mtime_ns) !=
                        (after.st_size, after.st_mtime_ns)):
                    raise ValueError("Arquivo mudou durante leitura")
                if (valid_cached and digest == stored.get("source_file_sha256")
                        and len(stored.get("source_pixel_sha256") or "") == 64):
                    item = {
                        "path": relative, "expected_label": label, **hints,
                        "status": "VALID_PNG", "file_sha256": digest,
                        "pixel_sha256": stored["source_pixel_sha256"],
                    }
                    reused += 1
                else:
                    # Arquivo novo ou modificado: CRC, decode e pixel hash
                    # obrigatórios antes de ser admitido na memória.
                    item = _inspect_png(path, root, label)
            except (OSError, ValueError) as exc:
                item = {"path": relative, "expected_label": label, **hints,
                        "status": "INVALID_PNG", "pixel_sha256": "",
                        "file_sha256": "", "issues": [
                            {"code": "READ_ERROR", "detail": str(exc)}]}
        items.append(item)
        if item["status"] == "VALID_PNG" and item.get("pixel_sha256"):
            entry = by_digest[item["pixel_sha256"]]
            entry["labels"].add(label)
            entry["paths"].append(relative)
        if progress and (number % 16 == 0 or number == len(files)):
            progress(number, len(files), relative)
    conflicts = [
        {"pixel_sha256": digest, "paths": sorted(data["paths"]),
         "labels": ["NG", "OK"]}
        for digest, data in sorted(by_digest.items())
        if len(data["labels"]) > 1
    ]
    return {
        "schema": SCHEMA,
        "images": items,
        "summary": {
            "png_count": len(items),
            "valid_png": sum(x["status"] == "VALID_PNG" for x in items),
            "invalid_png": sum(x["status"] != "VALID_PNG" for x in items),
        },
        "cross_label_conflicts": conflicts,
        "reused_pixel_hashes_from_sha_verified_cache": reused,
        "source": "SHA_VERIFIED_FAST_KNN_C_INVENTORY",
    }


__all__ = ["inventory_for_knn_c"]
