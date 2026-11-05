"""Proveniência consolidada dos votos V3 ao final das três luzes.

O voto V3 não aprova 0/OK. Um NG presente somente em SIDE, TOP
ou MID pode bloquear uma autorização OK que venha do motor legado.
"""
from __future__ import annotations

import math
from src.core.neural.aoi_v3_trial_contract import (
    LIGHTS, POINTER_SCHEMA, SNAPSHOT_SHA256,
)
from src.core.neural.aoi_v3_epoch5_contract import (
    EPOCH as EPOCH5, EPOCH5_VOTE_SCHEMA, SNAPSHOT_SHA256 as EPOCH5_SHA256,
)


def _valid_vote(vote: dict, light: str) -> bool:
    if not isinstance(vote, dict):
        return False
    score = vote.get("ng_score_uncalibrated")
    return (
        (vote.get("schema"), vote.get("epoch"), vote.get("checkpoint_sha256"))
        in {
            (POINTER_SCHEMA, 3, SNAPSHOT_SHA256),
            (EPOCH5_VOTE_SCHEMA, EPOCH5, EPOCH5_SHA256),
        }
        and vote.get("status") == "READY"
        and vote.get("light") == light
        and vote.get("automatic_ok_from_v3_authorized") is False
        and isinstance(score, (int, float)) and not isinstance(score, bool)
        and math.isfinite(score) and 0. <= score <= 1.
        and vote.get("predicted") == ("NG" if score >= .5 else "OK")
        and vote.get("ng_veto") is (score >= .5)
    )


def attach_v3_multilight_evidence(fused: dict, source: dict) -> dict:
    if not isinstance(fused, dict):
        return fused
    votes = {}
    for light in LIGHTS:
        item = source.get(light, {})
        detail = item.get("detail", {}) if isinstance(item, dict) else {}
        value = detail.get("aoi_v3_trial") if isinstance(detail, dict) else None
        if value is not None:
            votes[light] = value
    if not votes:
        return fused
    detail = fused.setdefault("detail", {})
    if not isinstance(detail, dict):
        detail = {}
        fused["detail"] = detail
    if set(votes) != set(LIGHTS) or not all(
        _valid_vote(votes.get(light), light) for light in LIGHTS
    ):
        detail["aoi_v3_multilight"] = {
            "status": "ERROR",
            "reason": "INCOMPLETE_OR_INVALID_V3_LIGHT_PROVENANCE",
            "automatic_ok_from_v3_authorized": False,
        }
        return fused
    identities = {
        (votes[light]["epoch"], votes[light]["checkpoint_sha256"])
        for light in LIGHTS
    }
    if len(identities) != 1:
        detail["aoi_v3_multilight"] = {
            "status": "ERROR",
            "reason": "MIXED_V3_EPOCHS",
            "automatic_ok_from_v3_authorized": False,
        }
        return fused
    epoch, digest = next(iter(identities))
    detail["aoi_v3_multilight"] = {
        "status": "READY",
        "checkpoint_sha256": digest,
        "epoch": epoch,
        "mode": (
            "v3_primary_epoch5_operator_override"
            if epoch == EPOCH5 else "ng_veto_only"
        ),
        "lights": {
            light: {
                "predicted": votes[light]["predicted"],
                "ng_score_uncalibrated": votes[light]["ng_score_uncalibrated"],
            } for light in LIGHTS
        },
        "ng_veto": any(votes[light]["ng_veto"] for light in LIGHTS),
        "automatic_ok_from_v3_authorized": False,
        "blind_test_of_selected_epoch": False,
    }
    return fused


def active_v3_veto(analysis: dict) -> tuple[str, bool]:
    """Retorna (ABSENT | READY | ERROR, ng_veto), sem confiar em texto livre."""
    if not isinstance(analysis, dict):
        return "ABSENT", False
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return "ABSENT", False
    overall = detail.get("aoi_v3_multilight")
    if overall is not None:
        if (
            not isinstance(overall, dict)
            or overall.get("status") != "READY"
            or (overall.get("epoch", 3), overall.get("checkpoint_sha256"), overall.get("mode"))
               not in {
                   (3, SNAPSHOT_SHA256, "ng_veto_only"),
                   (EPOCH5, EPOCH5_SHA256, "v3_primary_epoch5_operator_override"),
               }
            or overall.get("automatic_ok_from_v3_authorized") is not False
            or not isinstance(overall.get("lights"), dict)
            or set(overall["lights"]) != set(LIGHTS)
        ):
            return "ERROR", False
        for light in LIGHTS:
            item = overall["lights"][light]
            if not isinstance(item, dict):
                return "ERROR", False
            score = item.get("ng_score_uncalibrated")
            if (
                isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1
                or item.get("predicted") != ("NG" if score >= .5 else "OK")
            ):
                return "ERROR", False
        veto = any(
            overall["lights"][light]["predicted"] == "NG" for light in LIGHTS
        )
        return ("READY", veto) if overall.get("ng_veto") is veto else ("ERROR", False)
    single = detail.get("aoi_v3_trial")
    if single is None:
        return "ABSENT", False
    if not _valid_vote(single, str(analysis.get("lighting_mode") or "")):
        return "ERROR", False
    return "READY", single["ng_veto"]


__all__ = ["attach_v3_multilight_evidence", "active_v3_veto"]
