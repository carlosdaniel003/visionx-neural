"""Carregamento somente-leitura de um candidato CNN V3 para fine-tuning.

Um checkpoint anterior deve ser o mesmo que foi publicado no relatório
offline, com SHA-256 e schema válidos. Nunca modifica o candidato antigo.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re

import torch

from src.core.neural.aoi_quality_cnn_v3 import (
    AOIQualityCNNV3, MODEL_SCHEMA_V3,
)

REPORT_SCHEMA = "visionx.aoi_quality_cnn_v3_training_report.v1"
CANDIDATE_SCHEMA = "visionx.aoi_quality_cnn_v3_candidate_only.v1"
SHA = re.compile(r"^[0-9a-f]{64}$")


def load_verified_v3_candidate(
    root: Path, report_path: Path, manifest_sha256: str,
) -> tuple[AOIQualityCNNV3, dict]:
    root = root.resolve()
    path = Path(report_path)
    path = (path if path.is_absolute() else root/path).resolve()
    candidates = (root/"reports/aoi_quality_cnn_v3/candidates").resolve()
    if (
        path.name != "training_report.json"
        or path.is_symlink() or not path.is_file()
        or not path.is_relative_to(candidates)
    ):
        raise ValueError("Relatório V3 fora dos candidatos offline")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(doc, dict)
        or doc.get("schema") != REPORT_SCHEMA
        or doc.get("candidate_only") is not True
        or doc.get("production_approved") is not False
        or doc.get("automatic_xp_authorized") is not False
        or doc.get("provenance", {}).get("manifest_sha256") != manifest_sha256
        or not SHA.fullmatch(str(doc.get("candidate_sha256", "")))
    ):
        raise ValueError("Relatório candidato V3 incompatível/não auditado")
    candidate = (root / str(doc.get("candidate_path", ""))).resolve()
    if (
        candidate.parent != path.parent
        or candidate.name != "aoi_quality_cnn_v3_candidate.pt"
        or candidate.is_symlink() or not candidate.is_file()
    ):
        raise ValueError("Pesos V3 fora do relatório offline")
    digest = sha256(candidate.read_bytes()).hexdigest()
    if digest != doc["candidate_sha256"]:
        raise ValueError("SHA-256 do candidato V3 foi alterado")
    raw = torch.load(candidate, weights_only=True, map_location="cpu")
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != CANDIDATE_SCHEMA
        or raw.get("architecture_schema") != MODEL_SCHEMA_V3
        or raw.get("candidate_only") is not True
        or raw.get("production_approved") is not False
        or raw.get("automatic_xp_authorized") is not False
        or raw.get("source_manifest_sha256") != manifest_sha256
        or raw.get("image_size") not in (64, 96, 128, 160)
        or raw.get("focus_fraction") != .70
        or tuple(raw.get("lights", ())) != ("SIDE", "TOP", "MID")
    ):
        raise ValueError("Checkpoint V3 não é candidato offline íntegro")
    model = AOIQualityCNNV3()
    model.load_state_dict(raw["state_dict"], strict=True)
    return model, {
        "report_path": path.relative_to(root).as_posix(),
        "candidate_sha256": digest,
        "candidate_path": candidate.relative_to(root).as_posix(),
        "image_size": raw["image_size"],
        "focus_fraction": raw["focus_fraction"],
    }


__all__ = ["load_verified_v3_candidate"]
