"""Métricas binárias OK/NG da CNN V3, sem taxonomia da AOI.

A categoria registrada no nome do screenshot não é fonte de verdade.
Apenas a qualidade confirmada/provisória do par e sua iluminação contam.
"""
from __future__ import annotations

from collections import Counter

LIGHTS = ("SIDE", "TOP", "MID")


def _confusion(labels: list[tuple[str, str]]) -> dict:
    count = Counter(labels)
    ng = count[("NG", "NG")] + count[("NG", "OK")]
    ok = count[("OK", "OK")] + count[("OK", "NG")]
    tp = count[("NG", "NG")]
    tn = count[("OK", "OK")]
    return {
        "total": ng + ok,
        "correct": tp + tn,
        "accuracy": (tp+tn)/(ng+ok) if ng+ok else 0.,
        "NG_total": ng, "OK_total": ok,
        "NG_correct": tp, "NG_false_OK": ng-tp,
        "OK_correct": tn, "OK_false_NG": ok-tn,
        "ng_recall": tp/ng if ng else None,
        "ok_specificity": tn/ok if ok else None,
        "balanced_accuracy": .5*(tp/ng+tn/ok) if ng and ok else None,
    }


def binary_quality_metrics(records: list[dict], prediction: dict) -> dict:
    """Garante 1:1 entre registros e previsões antes de medir o modelo."""
    cases = prediction.get("cases") if isinstance(prediction, dict) else None
    if not isinstance(cases, list) or len(cases) != len(records):
        raise ValueError("Avaliação CNN V3 sem correspondência 1:1 de imagens")
    ids = [row.get("id") for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("IDs duplicados para avaliação de qualidade V3")
    overall = []
    lights = {light: [] for light in LIGHTS}
    for row, pred in zip(records, cases):
        if (
            not isinstance(pred, dict)
            or row.get("label") not in ("OK", "NG")
            or pred.get("event_id_or_group_hint") != row["id"]
            or pred.get("label") != row["label"]
            or pred.get("predicted") not in ("OK", "NG")
            or set(pred.get("per_light", {})) != set(row.get("views", {}))
        ):
            raise ValueError("Previsão divergente do par ou qualidade V3")
        label = row["label"]
        overall.append((label, pred["predicted"]))
        for light, view in pred["per_light"].items():
            if light not in LIGHTS or not isinstance(view, dict):
                raise ValueError("Vista CNN V3 inválida")
            score = view.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not 0 <= score <= 1
                or view.get("correct") is not ((score >= .5) == (label == "NG"))
            ):
                raise ValueError("Score de luz CNN V3 não corresponde ao veredito")
            lights[light].append((label, "NG" if score >= .5 else "OK"))
    summary = _confusion(overall)
    summary["by_lighting"] = {
        light: _confusion(view) for light, view in lights.items()
    }
    summary["source_quality"] = "OPERATOR_AND_PROVISIONAL_ARCHIVE_LABELS"
    summary["aoi_category_used_for_metrics"] = False
    summary["threshold"] = .5
    return summary


def quality_coverage(rows: list[dict]) -> dict:
    provenance = Counter(r.get("label_provenance", "UNKNOWN") for r in rows)
    by_light = Counter(
        (light, r["label"]) for r in rows for light in r["views"]
    )
    return {
        "by_quality_provenance": dict(provenance),
        "by_lighting_and_quality": {
            light: {label: by_light[(light, label)] for label in ("OK", "NG")}
            for light in LIGHTS
        },
        "each_available_light_has_both_labels": all(
            by_light[(light, "OK")] > 0 and by_light[(light, "NG")] > 0
            for light in LIGHTS if any(by_light[(light, x)] for x in ("OK", "NG"))
        ),
        "fully_operator_verified": bool(rows) and all(
            r.get("label_provenance") in {
                "HUMAN_CONFIRMED_SHA256",
                "HUMAN_INDIVIDUAL_PAIR",
                "HUMAN_CONTACT_SHEET_NG",
                "HUMAN_V3_IMAGE_SHA256",
                "HUMAN_V3_NG_ESCAPE_SHA256",
            } for r in rows
        ),
        "human_verified_v3": provenance["HUMAN_V3_IMAGE_SHA256"],
        "human_verified_v3_ng_escape": provenance["HUMAN_V3_NG_ESCAPE_SHA256"],
        "aoi_category_used_for_coverage": False,
    }


__all__ = ["binary_quality_metrics", "quality_coverage"]
