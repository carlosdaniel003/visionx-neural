"""Human-confirmed AOI archive quality labels for *live* KNN recall.

Archive locations encode the original AOI judgment, not necessarily physical
quality. Never edit historical images or treat a non-matching SHA as confirmed.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from src.config.settings import BASE_DIR

EVIDENCE_PATH = "docs/odin_operator_confirmed_quality_20261026.json"
CORRECTED_SOURCE = "public/ng_archive/2026-10-21_1414_FALTANDO.png"
CORRECTED_SOURCE_FILE_SHA256 = "b33a8a9b9a9f7ef2684564f851ee989c5a76783ab158095cd5b910c52311dc65"
CORRECTED_SOURCE_PIXEL_SHA256 = "4481318c87aa3568959554fa95b438ef1b1c74a9d2b4876c2bd14c7ea751f9a3"


def verified_operator_archive_override(
    root: Path | None = None, source: str = CORRECTED_SOURCE,
) -> str | None:
    """Return validated physical-quality override, or None (fail closed)."""
    if source != CORRECTED_SOURCE:
        return None
    root = Path(root or BASE_DIR).resolve()
    try:
        document = root / EVIDENCE_PATH
        image = root / CORRECTED_SOURCE
        if (document.is_symlink() or image.is_symlink()
                or not document.is_file() or not image.is_file()
                or not image.resolve().is_relative_to(root / "public")):
            return None
        payload = json.loads(document.read_text(encoding="utf-8"))
        if (payload.get("schema") != "visionx.operator_confirmed_quality.v1"
                or payload.get("reviewed_on") != "2026-10-26"
                or payload.get("approved_for_production") is not False
                or not isinstance(payload.get("labels"), list)):
            return None
        rows = [r for r in payload["labels"] if isinstance(r, dict)
                and r.get("source_path") == CORRECTED_SOURCE]
        if (len(rows) != 1 or rows[0].get("quality_label") != "OK"
                or rows[0].get("source_sha256") != CORRECTED_SOURCE_FILE_SHA256
                or rows[0].get("physical_type") != "COMPONENT_PRESENT"):
            return None
        if sha256(image.read_bytes()).hexdigest() != CORRECTED_SOURCE_FILE_SHA256:
            return None
        return "OK"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
