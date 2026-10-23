"""Cache local verificável dos descritores KNN C, nunca memória decisória.

Armazena apenas assinaturas visuais/geométricas (sem imagens completas), com
identidade de origem SHA-256 verificada pelo inventário nesta execução.
JSON+gzip, sem pickle. Falha de cache é tratada como cache vazio.
"""
from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from src.services.startup_regression.knn_archive_focus_signature import (
    valid_focus_signature,
)
from src.services.startup_regression.knn_archive_geometry_experiment import (
    SCHEMA as GEO_SCHEMA,
)

SCHEMA = "visionx.knn_c_signature_cache.v1"
MAX_CACHE_UNCOMPRESSED = 95 * 1024 * 1024
MAX_ENTRIES = 2000
MAX_REASONABLE_MAP = 16


def _geometry_valid(obj):
    if not isinstance(obj, dict) or obj.get("schema") != GEO_SCHEMA:
        return False
    scales = obj.get("scales")
    if not isinstance(scales, dict) or set(scales) != {"global", "outer", "inner"}:
        return False
    for item in scales.values():
        if item is None:
            continue
        if not isinstance(item, dict):
            return False
        try:
            for key in ("missing_map", "added_map"):
                a = np.asarray(item.get(key), dtype=np.float32)
                if a.shape != (MAX_REASONABLE_MAP, MAX_REASONABLE_MAP) or not np.all(np.isfinite(a)):
                    return False
            for key in ("missing_stats", "added_stats"):
                a = np.asarray(item.get(key), dtype=np.float32)
                if a.shape != (3,) or not np.all(np.isfinite(a)):
                    return False
        except (TypeError, ValueError):
            return False
    return True


def _identifiable(row):
    if not isinstance(row, dict):
        return False
    if (row.get("expected_label") not in {"OK", "NG"}
            or row.get("lighting_mode") not in {"SIDE", "TOP", "MID"}
            or not isinstance(row.get("source_path"), str)
            or not isinstance(row.get("source_pixel_sha256"), str)
            or len(row["source_pixel_sha256"]) != 64):
        return False
    return (valid_focus_signature(row.get("focus_signature"))
            and _geometry_valid(row.get("geometry_signature")))


def read_cache(root: Path) -> dict:
    path = Path(root).resolve() / "reports/startup_regression/knn_c_signature_cache.json.gz"
    try:
        if path.stat().st_size > MAX_CACHE_UNCOMPRESSED:
            return {}
        with gzip.open(path, "rb") as inp:
            raw = inp.read(MAX_CACHE_UNCOMPRESSED + 1)
        if len(raw) > MAX_CACHE_UNCOMPRESSED:
            return {}
        obj = json.loads(raw.decode("utf-8"))
        if obj.get("schema") != SCHEMA:
            return {}
        rows = obj.get("entries")
        if not isinstance(rows, list) or len(rows) > MAX_ENTRIES:
            return {}
        result = {}
        for row in rows:
            if not _identifiable(row):
                return {}
            key = row["source_path"]
            if key in result:
                return {}
            result[key] = row
        return result
    except (OSError, EOFError, ValueError, TypeError, UnicodeError):
        return {}


def write_cache(root: Path, rows: list[dict]) -> bool:
    if len(rows) > MAX_ENTRIES or any(not _identifiable(x) for x in rows):
        return False
    target = Path(root).resolve() / "reports/startup_regression/knn_c_signature_cache.json.gz"
    temp = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp",
                                         delete=False) as f:
            temp = Path(f.name)
        with gzip.open(temp, "wt", encoding="utf-8", compresslevel=4) as stream:
            json.dump({"schema": SCHEMA, "entries": rows}, stream,
                      ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        os.replace(temp, target)
        return True
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if temp is not None:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = ["SCHEMA", "read_cache", "write_cache"]
