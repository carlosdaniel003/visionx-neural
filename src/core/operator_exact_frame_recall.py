"""Reconhecimento do mesmo frame XP confirmado por humano em três luzes.

Complementa o par AOI exato: OCR/recorte pode variar, mas o frame inteiro
continua byte a byte idêntico em pixels. Nunca promove KNN 98% a exato.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import numpy as np

from src.core.archive_persistent_memory import PersistentArchiveKNN
from src.core.physical_missing_veto import physical_absence_probe
from src.core.strict_category_memory import canonical_memory_category
from src.services.image_archive_dedup import image_fingerprint

RULE = "operator_latest_exact_frame_triplet_v1"
SIDE_RULE = "operator_latest_exact_frame_side_v1"
SCHEMA = "visionx.operator_latest_exact_frame.v1"
LIGHTS = ("SIDE", "TOP", "MID")


def _proof_digest(proof: dict) -> str:
    parts = {
        key: proof[key] for key in (
            "schema", "category", "label", "source_event_id",
            "confirmed_at_ns", "lights", "matches", "physical_probes",
        )
    }
    return sha256(json.dumps(
        parts, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def build_latest_operator_exact_frame(
    fused: dict, samples: dict | None, category: str, *, store=None,
    lights: tuple[str, ...] = LIGHTS,
) -> dict:
    """Aplica o último 0/1 humano somente à mesma trinca RGB completa.

    Não modifica resultado se falta luz, se há erro físico, se a fonte
    humana difere entre luzes ou se qualquer frame mudou um pixel.
    """
    if lights not in (LIGHTS, ("SIDE",)):
        return fused
    if not isinstance(fused, dict) or not isinstance(samples, dict):
        return fused
    if not all(isinstance(samples.get(light), dict) for light in lights):
        return fused
    cat = canonical_memory_category(category)
    if cat in {"", "UNKNOWN", "UNKNOW", "DESCONHECIDO"}:
        return fused
    memory = store if store is not None else PersistentArchiveKNN()
    matches = {}
    for light in lights:
        sample = samples[light]
        frame = sample.get("source_frame")
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return fused
        found = memory.latest_confirmed_human_for_frame(frame, cat, light)
        if found.get("status") != "KNOWN":
            return fused
        matches[light] = {
            "frame_sha256": found["frame_sha256"],
            "record_id": found["record_id"],
            "confirmed_at_ns": found["confirmed_at_ns"],
            "source_event_id": found["source_event_id"],
            "label": found["label"],
        }
    labels = {row["label"] for row in matches.values()}
    moments = {row["confirmed_at_ns"] for row in matches.values()}
    events = {row["source_event_id"] for row in matches.values()}
    if len(labels) != 1 or len(moments) != 1 or len(events) != 1 or not next(iter(events)):
        return fused

    label = next(iter(labels))
    detail = fused.get("detail")
    if not isinstance(detail, dict):
        return fused
    trace = detail.get("decision_trace")
    trace = trace if isinstance(trace, dict) else {}
    veto = detail.get("physical_missing_veto")
    veto = veto if isinstance(veto, dict) else {}
    if (detail.get("capture_error") is True
            or detail.get("category_unknown") is True
            or detail.get("multilight_physical_disagreement") is True
            or detail.get("missing_hard_absence") is True
            or trace.get("hard_missing_evidence") is True
            or veto.get("status") in {
                "SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW",
            }):
        return fused

    probes = {}
    if label == "OK":
        for light in lights:
            sample = samples[light]
            ref, test = sample.get("sample_image"), sample.get("test_image")
            probe = physical_absence_probe(ref, test)
            status = probe.get("status") if isinstance(probe, dict) else ""
            if status != "NO_STRONG_EVIDENCE":
                return fused
            probes[light] = status

    proof = {
        "schema": SCHEMA,
        "category": cat,
        "label": label,
        "source_event_id": next(iter(events)),
        "confirmed_at_ns": next(iter(moments)),
        "lights": list(lights),
        "matches": matches,
        "physical_probes": probes,
    }
    proof["integrity"] = _proof_digest(proof)

    result = deepcopy(fused)
    d = result["detail"]
    d["operator_latest_exact_frame"] = proof
    d["verdict_before_latest_exact_frame"] = result.get("verdict", "")
    rule = SIDE_RULE if lights == ("SIDE",) else RULE
    d["fusion_rule"] = rule
    d["operator_review_required"] = False
    if lights == ("SIDE",):
        result["lighting_mode"] = "SIDE"
        result["multilight_final"] = True
        result["eligible_for_final_decision"] = True
        result["multilight_category"] = cat
        d["multilight_final"] = True
        d["eligible_for_final_decision"] = True
        d["multilight_category"] = cat
    result["is_defect"] = label == "NG"
    result["verdict"] = "FALHA FALSA" if label == "OK" else "DEFEITO REAL"
    result["production_review_required"] = False
    result["reason"] = (
        f"Última confirmação humana {label} prevaleceu: "
        f"{len(lights)} tela(s) XP pixel-idênticas, "
        "categoria/iluminação e prova física válidas. "
        "Scores antigos da CNN preservados no diagnóstico."
    )
    old_trace = d.get("decision_trace")
    if isinstance(old_trace, dict):
        old_trace["operator_review_required"] = False
        old_trace["fusion_rule"] = rule
        old_trace["operator_latest_exact_frame_integrity"] = proof["integrity"]
    return result


def build_latest_operator_exact_side(
    side: dict | None, sample: dict | None, category: str, *, store=None,
) -> dict | None:
    """Atalho SIDE com o mesmo frame e última confirmação humana real."""
    if not isinstance(side, dict) or not isinstance(sample, dict):
        return None
    result = build_latest_operator_exact_frame(
        side, {"SIDE": sample}, category, store=store, lights=("SIDE",),
    )
    return result if result is not side else None


def verified_latest_operator_exact_frame(
    analysis: dict | None, *, store=None,
) -> str | None:
    """Reconfere snapshot humano antes da autorização XP de 0/1."""
    if not isinstance(analysis, dict):
        return None
    d = analysis.get("detail")
    if not isinstance(d, dict) or d.get("fusion_rule") not in {RULE, SIDE_RULE}:
        return None
    lights = ("SIDE",) if d.get("fusion_rule") == SIDE_RULE else LIGHTS
    if lights == ("SIDE",) and analysis.get("lighting_mode") != "SIDE":
        return None
    proof = d.get("operator_latest_exact_frame")
    if not isinstance(proof, dict):
        return None
    try:
        if (proof.get("schema") != SCHEMA
                or proof.get("label") not in {"OK", "NG"}
                or not isinstance(proof.get("source_event_id"), str)
                or not proof["source_event_id"]
                or not isinstance(proof.get("confirmed_at_ns"), int)
                or proof["confirmed_at_ns"] <= 0
                or not isinstance(proof.get("matches"), dict)
                or set(proof["matches"]) != set(lights)
                or proof.get("lights") != list(lights)
                or proof.get("integrity") != _proof_digest(proof)
                or canonical_memory_category(d.get("multilight_category"))
                != proof.get("category")
                or d.get("multilight_final") is not True
                or analysis.get("multilight_final") is not True
                or d.get("capture_error") is True
                or d.get("category_unknown") is True
                or d.get("multilight_physical_disagreement") is True
                or d.get("missing_hard_absence") is True):
            return None
        trace = d.get("decision_trace")
        trace = trace if isinstance(trace, dict) else {}
        if trace.get("hard_missing_evidence") is True:
            return None
        veto = d.get("physical_missing_veto")
        veto = veto if isinstance(veto, dict) else {}
        if veto.get("status") in {
            "SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW",
        }:
            return None
        if proof["label"] == "OK" and proof.get("physical_probes") != {
            light: "NO_STRONG_EVIDENCE" for light in lights
        }:
            return None
        for light in lights:
            row = proof["matches"][light]
            if (row.get("label") != proof["label"]
                    or row.get("source_event_id") != proof["source_event_id"]
                    or row.get("confirmed_at_ns") != proof["confirmed_at_ns"]
                    or not isinstance(row.get("frame_sha256"), str)
                    or len(row["frame_sha256"]) != 64):
                return None
        memory = store if store is not None else PersistentArchiveKNN()
        # Um único snapshot válido, não confiar em flags da interface.
        records = memory.records()
        for light in lights:
            expected = proof["matches"][light]
            candidates = [
                (source["confirmed_at_ns"], item, source)
                for item in records
                if item["identity"][:2] == [proof["category"], light]
                for source in item["sources"]
                if source.get("source_kind") == "OPERATOR_CONFIRMED"
                and source.get("source_label") == item["label"]
                and source.get("source_frame_sha256")
                == expected["frame_sha256"]
                and type(source.get("confirmed_at_ns")) is int
            ]
            if not candidates:
                return None
            latest_ns = max(c[0] for c in candidates)
            newest = [c for c in candidates if c[0] == latest_ns]
            if (len({c[1]["label"] for c in newest}) != 1
                    or latest_ns != expected["confirmed_at_ns"]):
                return None
            selected = newest[-1]
            if (selected[1]["label"] != proof["label"]
                    or selected[1]["id"] != expected["record_id"]
                    or selected[2].get("event_id") != expected["source_event_id"]):
                return None
        return proof["label"]
    except (ValueError, TypeError, KeyError, OSError, AttributeError):
        return None


__all__ = [
    "build_latest_operator_exact_frame", "build_latest_operator_exact_side",
    "verified_latest_operator_exact_frame", "RULE", "SIDE_RULE", "SCHEMA",
]
