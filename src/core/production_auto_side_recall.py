"""Memória operacional exata das decisões automáticas de Produção.

Uma ordem 0/1 confirmada pelo XP NÃO é rótulo humano nem dado de treino.
Preserva proveniência em SQLite separado e permite reanálise SIDE rápida
somente de pares AOI pixel-idênticos, com vetos físicos e humanos revalidados.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from src.config.settings import BASE_DIR
from src.core.archive_persistent_memory import PersistentArchiveKNN
from src.core.physical_missing_veto import physical_absence_probe
from src.core.strict_category_memory import canonical_memory_category
from src.services.image_archive_dedup import image_fingerprint

RULE = "production_auto_side_exact_recall_v1"
SCHEMA = "visionx.production_auto_side_exact.v1"
DEFAULT_DB = Path(BASE_DIR) / "data" / "production_auto_side" / "decisions.sqlite3"
LIGHTS = ("SIDE", "TOP", "MID")


def _valid(image) -> bool:
    return (
        isinstance(image, np.ndarray)
        and image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[2] == 3
        and min(image.shape[:2]) >= 12
    )


def _canonical_category(value) -> str:
    category = canonical_memory_category(value)
    return category or "AOI_UNCLASSIFIED"


def _identity(category, reference, test) -> tuple[str, str, str] | None:
    cat = _canonical_category(category)
    if not _valid(reference) or not _valid(test):
        return None
    return cat, image_fingerprint(reference), image_fingerprint(test)


def _side_sample(task: dict) -> dict:
    samples = task.get("multilight_samples")
    if isinstance(samples, list):
        for sample in samples:
            if isinstance(sample, dict) and sample.get("lighting_mode") == "SIDE":
                return sample
    return {
        "lighting_mode": "SIDE",
        "sample_image": task.get("sample_image"),
        "test_image": task.get("ng_image"),
        "source_frame": task.get("source_frame"),
        "analysis": task.get("analysis"),
    }


def _side_raw_ok(analysis: dict | None) -> bool:
    if not isinstance(analysis, dict):
        return False
    detail = analysis.get("detail")
    detail = detail if isinstance(detail, dict) else {}
    trace = detail.get("decision_trace")
    trace = trace if isinstance(trace, dict) else {}
    veto = detail.get("physical_missing_veto")
    veto = veto if isinstance(veto, dict) else {}
    return bool(
        analysis.get("is_defect") is False
        and analysis.get("verdict") == "FALHA FALSA"
        and analysis.get("production_review_required") is not True
        and detail.get("capture_error") is not True
        and detail.get("missing_hard_absence") is not True
        and trace.get("hard_missing_evidence") is not True
        and veto.get("status") not in {
            "SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW",
        }
    )


def _safety_evidence_ok(analysis: dict | None, label: str, reference, test) -> str:
    """NG somente rejeita; OK exige concordância SIDE e veto físico independente."""
    if not isinstance(analysis, dict):
        return ""
    detail = analysis.get("detail")
    detail = detail if isinstance(detail, dict) else {}
    trace = detail.get("decision_trace")
    trace = trace if isinstance(trace, dict) else {}
    veto = detail.get("physical_missing_veto")
    veto = veto if isinstance(veto, dict) else {}
    if (
        detail.get("capture_error") is True
        or detail.get("missing_hard_absence") is True
        or trace.get("hard_missing_evidence") is True
        or veto.get("status") in {
            "SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW",
        }
    ):
        return ""
    if label == "NG":
        return "REJECTION_ONLY"
    if label != "OK" or not _side_raw_ok(analysis):
        return ""
    probe = physical_absence_probe(reference, test)
    return (
        "NO_STRONG_EVIDENCE"
        if isinstance(probe, dict) and probe.get("status") == "NO_STRONG_EVIDENCE"
        else ""
    )


def _proof_digest(payload: dict) -> str:
    clean = {key: value for key, value in payload.items() if key != "integrity"}
    return sha256(json.dumps(
        clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


class ProductionAutoSideMemory:
    """Índice SQLite autocontido; não faz parte do treino ou memória KNN humana."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or DEFAULT_DB)

    @contextmanager
    def _connection(self, *, create: bool):
        if not create and not self.path.is_file():
            yield None
            return
        if self.path.is_symlink():
            raise ValueError("Memória automática não pode ser symlink")
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=3.0)
        try:
            connection.execute("PRAGMA busy_timeout=3000")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    reference_sha256 TEXT NOT NULL,
                    test_sha256 TEXT NOT NULL,
                    source_frame_sha256 TEXT NOT NULL,
                    label TEXT NOT NULL CHECK(label IN ('OK','NG')),
                    source_reason TEXT NOT NULL,
                    original_side_defect INTEGER,
                    analyzed_lights TEXT NOT NULL,
                    recorded_at_ns INTEGER NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_auto_side_pair "
                "ON decisions(category, reference_sha256, test_sha256)"
            )
            yield connection
            connection.commit()
        finally:
            connection.close()

    def record(self, task: dict) -> bool:
        if not isinstance(task, dict) or task.get("source") != "production_auto":
            return False
        label = str(task.get("label", "")).upper()
        event_id = str(task.get("event_id", "") or "").strip()
        analysis = task.get("analysis")
        detail = analysis.get("detail") if isinstance(analysis, dict) else {}
        policy = detail.get("production_decision_policy") if isinstance(detail, dict) else {}
        if not (
            label in {"OK", "NG"}
            and event_id
            and isinstance(policy, dict)
            and policy.get("resolution") == "automatic"
            and policy.get("auto_allowed") is True
            and policy.get("proposed_decision") == label
            and not str(policy.get("reason", "")).startswith("production_auto_side_recall_")
        ):
            return False
        sample = _side_sample(task)
        info = task.get("aoi_info") or {}
        if not isinstance(info, dict):
            return False
        identity = _identity(info.get("category"), sample.get("sample_image"),
                             sample.get("test_image"))
        if identity is None:
            return False
        original_side = sample.get("analysis")
        side_defect = (
            int(original_side["is_defect"])
            if isinstance(original_side, dict)
            and type(original_side.get("is_defect")) is bool else None
        )
        if label == "OK" and side_defect is not None and side_defect != 0:
            # Não memorizar um OK quando a própria SIDE detectou NG.
            return False
        frames = task.get("multilight_samples")
        lights = [
            row.get("lighting_mode") for row in frames
            if isinstance(row, dict) and row.get("lighting_mode") in LIGHTS
        ] if isinstance(frames, list) else ["SIDE"]
        frame = sample.get("source_frame")
        frame_hash = image_fingerprint(frame) if _valid(frame) else ""
        with self._connection(create=True) as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO decisions (
                    event_id, category, reference_sha256, test_sha256,
                    source_frame_sha256, label, source_reason,
                    original_side_defect, analyzed_lights, recorded_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, *identity, frame_hash, label,
                 str(policy.get("reason") or ""), side_defect,
                 ",".join(lights), time.time_ns()),
            )
            return cursor.rowcount == 1

    def find(self, identity: tuple[str, str, str]) -> dict | None:
        with self._connection(create=False) as conn:
            if conn is None:
                return None
            rows = conn.execute(
                """SELECT id, event_id, label, source_reason,
                          original_side_defect, source_frame_sha256,
                          analyzed_lights, recorded_at_ns
                   FROM decisions
                   WHERE reference_sha256=? AND test_sha256=?
                   ORDER BY id DESC""",
                identity[1:],
            ).fetchall()
        if not rows:
            return None
        if len({row[2] for row in rows}) != 1:
            return {"status": "CONFLICT"}
        latest = rows[0]
        return {
            "status": "KNOWN", "id": latest[0], "event_id": latest[1],
            "label": latest[2], "source_reason": latest[3],
            "original_side_defect": latest[4],
            "source_frame_sha256": latest[5],
            "analyzed_lights": latest[6],
            "recorded_at_ns": latest[7],
        }


def _human_or_archive_contradiction(
    identity: tuple[str, str, str], frame_hash: str, auto_label: str,
    *, human_store=None,
) -> bool:
    """Qualquer confirmação humana para este par/frame prevalece sobre IA."""
    records = (
        human_store if human_store is not None else PersistentArchiveKNN()
    ).records()
    _category, ref_hash, test_hash = identity
    for record in records:
        key = record.get("identity", [])
        same_pair = (
            len(key) == 4 and key[1:] == ["SIDE", ref_hash, test_hash]
        )
        for source in record.get("sources", []):
            if source.get("source_kind") == "OPERATOR_CONFIRMED" and (
                same_pair
                or (frame_hash and len(key) == 4 and key[1] == "SIDE"
                    and source.get("source_frame_sha256") == frame_hash)
            ):
                return True
        # NG físico conhecido é contraprova contra auto-OK, mesmo que a
        # fonte seja um PNG histórico e não uma correção humana recente.
        if same_pair and auto_label == "OK" and record.get("label") == "NG":
            return True
    return False


def build_auto_side_recall(
    side: dict | None, sample: dict | None, category: str, *,
    store: ProductionAutoSideMemory | None = None, human_store=None,
) -> dict | None:
    """Apenas na produção: retorno SIDE antes de coletar TOP/MID."""
    if not isinstance(side, dict) or not isinstance(sample, dict):
        return None
    ref, test = sample.get("sample_image"), sample.get("test_image")
    identity = _identity(category, ref, test)
    if identity is None:
        return None
    frame = sample.get("source_frame")
    frame_hash = image_fingerprint(frame) if _valid(frame) else ""
    index = store if store is not None else ProductionAutoSideMemory()
    try:
        saved = index.find(identity)
        if not saved or saved.get("status") != "KNOWN":
            return None
        if _human_or_archive_contradiction(
            identity, frame_hash, saved["label"], human_store=human_store,
        ):
            return None
    except (OSError, ValueError, sqlite3.DatabaseError, KeyError):
        return None

    label = saved["label"]
    # Não transformar o NG preventivo histórico em uma prova de OK.
    if label == "OK" and saved["original_side_defect"] != 0:
        return None
    physical = _safety_evidence_ok(side, label, ref, test)
    if not physical:
        return None

    proof = {
        "schema": SCHEMA, "record_id": saved["id"],
        "event_id": saved["event_id"], "recorded_at_ns": saved["recorded_at_ns"],
        "category": identity[0], "reference_sha256": identity[1],
        "test_sha256": identity[2], "source_frame_sha256": frame_hash,
        "label": label, "source_reason": saved["source_reason"],
        "analyzed_lights": saved["analyzed_lights"],
        "original_side_defect": saved["original_side_defect"],
        "physical_probe": physical,
    }
    proof["integrity"] = _proof_digest(proof)
    result = deepcopy(side)
    d = result.setdefault("detail", {})
    d["production_auto_side_recall"] = proof
    d["fusion_rule"] = RULE
    d["multilight_final"] = True
    d["eligible_for_final_decision"] = True
    d["multilight_category"] = identity[0]
    d["operator_review_required"] = False
    result.update({
        "lighting_mode": "SIDE", "multilight_final": True,
        "eligible_for_final_decision": True,
        "multilight_category": identity[0],
        "production_review_required": False,
        "is_defect": label == "NG",
        "verdict": "DEFEITO REAL" if label == "NG" else "FALHA FALSA",
        "reason": (
            f"Memória operacional AUTO {label}, SIDE AOI pixel-idêntica; "
            f"origem {saved['source_reason']}. Não é rótulo humano. "
            "TOP/MID dispensadas somente após gate XP."
        ),
    })
    trace = d.get("decision_trace")
    if isinstance(trace, dict):
        trace["fusion_rule"] = RULE
        trace["operator_review_required"] = False
        trace["production_auto_memory_provenance"] = "PRODUCTION_AUTO"
    return result


def auto_side_display_provenance(analysis: dict | None) -> dict | None:
    """Confere prova local da rota AUTO para a UI, sem I/O ou autorização XP.

    verify_auto_side_recall() permanece o único gate que relê SQLite,
    memória humana e vetos físicos imediatamente antes do comando.
    """
    if not isinstance(analysis, dict):
        return None
    detail = analysis.get("detail")
    if not isinstance(detail, dict) or detail.get("fusion_rule") != RULE:
        return None
    proof = detail.get("production_auto_side_recall")
    if not isinstance(proof, dict):
        return None
    try:
        label = proof.get("label")
        if (
            proof.get("schema") != SCHEMA
            or label not in {"OK", "NG"}
            or not isinstance(proof.get("integrity"), str)
            or proof["integrity"] != _proof_digest(proof)
            or type(proof.get("record_id")) is not int
            or proof["record_id"] <= 0
            or type(proof.get("recorded_at_ns")) is not int
            or proof["recorded_at_ns"] <= 0
            or not isinstance(proof.get("event_id"), str)
            or not proof["event_id"]
            or any(
                not isinstance(proof.get(field), str)
                or len(proof[field]) != 64
                or any(char not in "0123456789abcdef" for char in proof[field])
                for field in ("reference_sha256", "test_sha256")
            )
            or _canonical_category(proof.get("category")) != proof.get("category")
            or analysis.get("lighting_mode") != "SIDE"
            or analysis.get("multilight_final") is not True
            or detail.get("multilight_final") is not True
            or analysis.get("eligible_for_final_decision") is not True
            or detail.get("eligible_for_final_decision") is not True
            or analysis.get("is_defect") is not (label == "NG")
            or analysis.get("verdict") != (
                "DEFEITO REAL" if label == "NG" else "FALHA FALSA"
            )
            or analysis.get("production_review_required") is True
            or detail.get("capture_error") is True
                or detail.get("missing_hard_absence") is True
        ):
            return None
        if label == "OK" and (
            proof.get("physical_probe") != "NO_STRONG_EVIDENCE"
            or proof.get("original_side_defect") != 0
        ):
            return None
        trace = detail.get("decision_trace")
        trace = trace if isinstance(trace, dict) else {}
        veto = detail.get("physical_missing_veto")
        veto = veto if isinstance(veto, dict) else {}
        if (
            trace.get("hard_missing_evidence") is True
            or veto.get("status") in {
                "SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW",
            }
        ):
            return None
        return {
            "label": label, "lighting": "SIDE",
            "analyzed_lights": str(proof.get("analyzed_lights") or ""),
            "source": "PRODUCTION_AUTO",
        }
    except (TypeError, ValueError, KeyError, AttributeError):
        return None


def verify_auto_side_recall(
    analysis: dict | None, *,
    store: ProductionAutoSideMemory | None = None, human_store=None,
) -> str | None:
    """Revalidação do disco + da origem humana antes do PRESS_0/PRESS_1."""
    if not isinstance(analysis, dict):
        return None
    d = analysis.get("detail")
    if not isinstance(d, dict) or d.get("fusion_rule") != RULE:
        return None
    proof = d.get("production_auto_side_recall")
    if not isinstance(proof, dict):
        return None
    try:
        if (
            proof.get("schema") != SCHEMA
            or proof.get("label") not in {"OK", "NG"}
            or proof.get("integrity") != _proof_digest(proof)
            or analysis.get("lighting_mode") != "SIDE"
            or analysis.get("multilight_final") is not True
            or analysis.get("eligible_for_final_decision") is not True
            or d.get("multilight_final") is not True
            or d.get("eligible_for_final_decision") is not True
            or analysis.get("is_defect") is not (proof["label"] == "NG")
            or d.get("capture_error") is True
            or d.get("missing_hard_absence") is True
        ):
            return None
        if (
            proof["label"] == "OK"
            and (proof.get("physical_probe") != "NO_STRONG_EVIDENCE"
                 or proof.get("original_side_defect") != 0)
        ):
            return None
        trace = d.get("decision_trace")
        if isinstance(trace, dict) and trace.get("hard_missing_evidence") is True:
            return None
        veto = d.get("physical_missing_veto")
        if isinstance(veto, dict) and veto.get("status") in {
            "SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW",
        }:
            return None
        identity = (
            proof["category"], proof["reference_sha256"],
            proof["test_sha256"],
        )
        saved = (
            store if store is not None else ProductionAutoSideMemory()
        ).find(identity)
        if (not saved or saved.get("status") != "KNOWN"
                or saved["id"] != proof["record_id"]
                or saved["label"] != proof["label"]
                or saved["event_id"] != proof["event_id"]
                or saved["recorded_at_ns"] != proof["recorded_at_ns"]):
            return None
        if _human_or_archive_contradiction(
            identity, proof.get("source_frame_sha256", ""),
            proof["label"], human_store=human_store,
        ):
            return None
        return proof["label"]
    except (OSError, ValueError, KeyError, TypeError, sqlite3.DatabaseError):
        return None


__all__ = [
    "RULE", "SCHEMA", "ProductionAutoSideMemory",
    "build_auto_side_recall", "verify_auto_side_recall",
    "auto_side_display_provenance",
]
