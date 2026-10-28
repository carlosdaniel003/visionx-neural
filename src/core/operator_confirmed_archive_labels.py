"""Verified canonical OK reference and retired erroneous AOI NG provenance.

The original NG judgment is retained in the human audit JSON, not in KNN
training. After the operator deletes the erroneous PNG, all live matching
depends exclusively on an unmodified, SHA-pinned OK archive reference.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from src.config.settings import BASE_DIR

EVIDENCE_PATH = "docs/odin_operator_confirmed_quality_20261026.json"
RETIRED_SOURCE = "public/ng_archive/2026-10-21_1414_FALTANDO.png"
RETIRED_SOURCE_FILE_SHA256 = "b33a8a9b9a9f7ef2684564f851ee989c5a76783ab158095cd5b910c52311dc65"
CANONICAL_OK_SOURCE = "public/ok_archive/2026-10-21_1416_FALTANDO.png"
CANONICAL_OK_FILE_SHA256 = "118f0396fdb3759d58063582dbfcd2a64f83d13389991c1d96050f0d58011043"
CANONICAL_OK_PIXEL_SHA256 = "0fddce88f7179bc816a094a1c2b7b9117cbef24a71d717cf29b3646ea619970f"


def verified_canonical_ok_source(root: Path | None = None) -> bool:
    """Only the verified live OK PNG is eligible; old NG must be gone."""
    root = Path(root or BASE_DIR).resolve()
    try:
        evidence = root / EVIDENCE_PATH
        canonical = root / CANONICAL_OK_SOURCE
        obsolete = root / RETIRED_SOURCE
        if (not evidence.is_file() or evidence.is_symlink()
                or not canonical.is_file() or canonical.is_symlink()
                or obsolete.exists() or obsolete.is_symlink()
                or not canonical.resolve().is_relative_to(root / "public/ok_archive")):
            return False
        document = json.loads(evidence.read_text(encoding="utf-8"))
        if (document.get("schema") != "visionx.operator_confirmed_quality.v1"
                or document.get("reviewed_on") != "2026-10-26"
                or document.get("approved_for_production") is not False
                or not isinstance(document.get("labels"), list)):
            return False
        rows = {source: [item for item in document["labels"]
                         if isinstance(item, dict)
                         and item.get("source_path") == source]
                for source in (RETIRED_SOURCE, CANONICAL_OK_SOURCE)}
        old, good = rows[RETIRED_SOURCE], rows[CANONICAL_OK_SOURCE]
        if not (len(old) == len(good) == 1
                and old[0].get("source_sha256") == RETIRED_SOURCE_FILE_SHA256
                and old[0].get("quality_label") == "OK"
                and good[0].get("source_sha256") == CANONICAL_OK_FILE_SHA256
                and good[0].get("quality_label") == "OK"
                and good[0].get("physical_type") == "COMPONENT_PRESENT"):
            return False
        return sha256(canonical.read_bytes()).hexdigest() == CANONICAL_OK_FILE_SHA256
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
