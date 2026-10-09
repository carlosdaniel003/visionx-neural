"""Replay SOMENTE-CNN de todos os screenshots OK/NG históricos disponíveis.

Não consulta KNN, não usa memória v2/v3 para decidir e não treina.
Analisa 100% dos PNGs no inventário e NÃO ignora erros, revisão,
categorias sem CNN ou imagens ilegíveis ao calcular a meta 98%.

Escopo atual: a CNN FALTANDO V2 atende FALTANDO, EMBORCADO, INVERTIDO
e DESLOCADO. ADESIVO não possui CNN especializada, portanto fica
SEM_CNN_PARA_CATEGORIA (não é computado como aprovação).

As etiquetas nas pastas OK/NG são gabarito HISTÓRICO, não evidência de
generalização a imagens inéditas. JSONs v2 sem PNG não podem ser
reinferidos por CNN a partir de seus vetores de 224 valores.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Callable

from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.strict_category_memory import canonical_memory_category
from src.services.faltando_cross_category_audit import _archive_image
from src.services.startup_regression.archive_inventory import inventory_archives
from src.utils.text_normalizer import normalize_aoi_text

SCHEMA = "visionx.cnn_full_historical_archive_replay.v1"
MODEL_ABSENCE = "CNN_FALTANDO_V2"
UNSUPPORTED = "SEM_CNN_PARA_CATEGORIA"
VERDICT_FOR_LABEL = {"OK": "FALHA FALSA", "NG": "DEFEITO REAL"}
MIN_RETENTION = .98


def _classify_model(category: str):
    category = canonical_memory_category(category)
    return MODEL_ABSENCE if uses_faltando_v2(category) else None


def _row(item: dict):
    return {
        "source_path": item.get("path"),
        "source_file_sha256": item.get("file_sha256"),
        "expected_label": item.get("expected_label"),
        "category_hint": item.get("category_hint"),
        "lighting_mode": item.get("lighting_mode"),
        "lighting_source": item.get("lighting_source"),
        "event_id": item.get("event_id"),
        "manifest_links": item.get("manifest_links", []),
        "ocr_observed_category": None,
        "model": None,
        "checkpoint_sha256": None,
        "ng_score_uncalibrated": None,
        "raw_binary_prediction": None,
        "raw_binary_correct": False,
        "verdict": None,
        "status": "NAO_ANALISADO",
        "error": None,
    }


def _validate_cnn_output(output: dict):
    if not isinstance(output, dict):
        raise ValueError("CNN deve retornar dicionário")
    detail = output.get("detail")
    if not isinstance(detail, dict):
        raise ValueError("CNN sem auditoria do checkpoint")
    if (detail.get("cnn_v2_active") is not True
            or detail.get("cnn_v2_checkpoint_verified") is not True
            or detail.get("cnn_v2_status") != "INFERENCE_OK"):
        raise ValueError("CNN indisponível ou checkpoint não verificado")
    from hashlib import sha256
    checkpoint = detail.get("cnn_v2_checkpoint_sha256")
    if (not isinstance(checkpoint, str) or len(checkpoint) != 64
            or any(x not in "0123456789abcdefABCDEF" for x in checkpoint)):
        raise ValueError("Hash de checkpoint inválido")
    score = detail.get("cnn_v2_ng_score_uncalibrated")
    if isinstance(score, bool):
        raise ValueError("Score bool inválido")
    score = float(score)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Score NG inválido")
    verdict = str(output.get("verdict", "")).upper().strip()
    if verdict not in ("FALHA FALSA", "DEFEITO REAL", "REVISÃO OBRIGATÓRIA"):
        raise ValueError("Veredito CNN inválido")
    if detail.get("production_review_required") is True and verdict != "REVISÃO OBRIGATÓRIA":
        raise ValueError("Revisão obrigatória contraditória")
    if output.get("production_review_required") and verdict != "REVISÃO OBRIGATÓRIA":
        raise ValueError("Resposta CNN não respeitou revisão")
    if verdict != "REVISÃO OBRIGATÓRIA" and (
        (score >= .5) != (verdict == "DEFEITO REAL")
    ):
        raise ValueError("Veredito divergente do score CNN")
    return score, verdict, checkpoint.lower()


def _summarize(rows: list[dict]) -> dict:
    statuses = Counter(r["status"] for r in rows)
    labels = Counter(r["expected_label"] for r in rows)
    supported = [r for r in rows if r["model"] is not None]
    supported_ng = [r for r in supported if r["expected_label"] == "NG"]
    supported_ok = [r for r in supported if r["expected_label"] == "OK"]
    correct_ng = sum(r["status"] == "PASSOU" for r in supported_ng)
    correct_ok = sum(r["status"] == "PASSOU" for r in supported_ok)
    raw_correct = sum(r["raw_binary_correct"] for r in rows)
    available = len(supported)
    total = len(rows)
    passed = statuses["PASSOU"]
    return {
        "total": total,
        "expected_OK": labels["OK"],
        "expected_NG": labels["NG"],
        "cnn_scoped": available,
        "cnn_not_available": statuses[UNSUPPORTED],
        "passed": passed,
        "regression": statuses["REGRESSAO"],
        "review": statuses["REVISAO_OBRIGATORIA"],
        "invalid": statuses["INVALIDO"],
        "unsupported": statuses[UNSUPPORTED],
        "status_counts": dict(sorted(statuses.items())),
        "historical_full_archive_match_rate": (
            round(passed / total, 6) if total else None
        ),
        "historical_supported_match_rate": (
            round(passed / available, 6) if available else None
        ),
        "raw_binary_correct": raw_correct,
        "raw_binary_match_rate_full_archive": (
            round(raw_correct / total, 6) if total else None
        ),
        "expected_NG_scoped": len(supported_ng),
        "NG_as_NG": correct_ng,
        "NG_as_OK": sum(
            r["verdict"] == "FALHA FALSA" for r in supported_ng
        ),
        "NG_to_review": sum(
            r["verdict"] == "REVISÃO OBRIGATÓRIA" for r in supported_ng
        ),
        "NG_invalid": sum(r["status"] == "INVALIDO" for r in supported_ng),
        "expected_OK_scoped": len(supported_ok),
        "OK_as_OK": correct_ok,
        "OK_as_NG": sum(
            r["verdict"] == "DEFEITO REAL" for r in supported_ok
        ),
        "OK_to_review": sum(
            r["verdict"] == "REVISÃO OBRIGATÓRIA" for r in supported_ok
        ),
        "OK_invalid": sum(r["status"] == "INVALIDO" for r in supported_ok),
        "scope_fully_covered": available == total,
        "historical_98pct_target_met": (
            total > 0 and available == total
            and passed / total >= MIN_RETENTION
        ),
    }


def _event_report(rows: list[dict]):
    # Apenas manifestos explícitos e íntegros podem unir TOP/MID/SIDE
    # em um evento. Nunca deduzir pelo minuto ou nome do screenshot.
    groups = defaultdict(list)
    for row in rows:
        if row["event_id"] and row["manifest_links"]:
            groups[row["event_id"]].append(row)
    events = []
    for event_id, group in sorted(groups.items()):
        lights = {r["lighting_mode"] for r in group}
        labels = {r["expected_label"] for r in group}
        outcomes = {r["status"] for r in group}
        if len(group) != 3 or lights != {"SIDE", "TOP", "MID"} or len(labels) != 1:
            status = "MANIFESTO_INCOMPLETO_OU_CONFLITANTE"
        elif outcomes == {"PASSOU"}:
            status = "PASSOU_3_LUZES"
        elif any(r["status"] in ("REGRESSAO", "INVALIDO", UNSUPPORTED) for r in group):
            status = "FALHOU_3_LUZES"
        else:
            status = "REVISAO_3_LUZES"
        events.append({
            "event_id": event_id,
            "source_paths": sorted(r["source_path"] for r in group),
            "expected_label": next(iter(labels)) if len(labels) == 1 else None,
            "lighting_modes": sorted(lights),
            "status": status,
        })
    return {
        "events_with_explicit_manifest": len(events),
        "status_counts": dict(sorted(Counter(r["status"] for r in events).items())),
        "events": events,
        "side_only_or_unlinked_are_not_inferred_as_three_light": True,
    }


def replay_full_cnn_history(
    root: Path, *, inventory: dict | None = None,
    extractor: Callable | None = None,
    predictors: dict | None = None,
    progress: Callable | None = None,
) -> dict:
    root = Path(root).expanduser().resolve()
    inv = inventory if inventory is not None else inventory_archives(root)
    items = inv.get("images")
    if (inv.get("schema") != "visionx.archive_inventory.v1"
            or not isinstance(items, list) or not items
            or len(items) != inv.get("summary", {}).get("png_count")):
        raise ValueError("Inventário inválido/incompleto; não avaliar amostra parcial")
    if extractor is None:
        from src.services.faltando_neural_dataset import AOIPairExtractor
        extractor = AOIPairExtractor()
    model_map = {} if predictors is None else dict(predictors)
    if MODEL_ABSENCE not in model_map:
        from src.core.neural.faltando_live import FaltandoCNNLive
        # FaltandoCNNLive verifica SHA-256 de checkpoint; NÃO usa KNN.
        model_map[MODEL_ABSENCE] = FaltandoCNNLive(online_root=root)
    conflicts = {
        path for group in inv.get("cross_label_conflicts", [])
        for path in group.get("paths", [])
    }
    rows = []
    for i, item in enumerate(items, 1):
        row = _row(item)
        try:
            if item.get("expected_label") not in ("OK", "NG"):
                raise ValueError("Rótulo do acervo inválido")
            if item.get("status") != "VALID_PNG":
                raise ValueError("Imagem inválida no inventário")
            if row["source_path"] in conflicts:
                raise ValueError("Mesmo PNG rotulado OK e NG")
            category_hint = canonical_memory_category(row["category_hint"])
            row["model"] = _classify_model(category_hint)
            if row["model"] is None:
                row["status"] = UNSUPPORTED
            else:
                frame = _archive_image(root, item)
                reference, test, info = extractor(frame)
                if not isinstance(info, dict):
                    raise ValueError("OCR não devolveu metadados")
                category, normalized_value = normalize_aoi_text(
                    info.get("value", "")
                )
                detected = canonical_memory_category(category)
                row["ocr_observed_category"] = detected
                if (not detected or detected == "UNKNOWN"
                        or detected != category_hint):
                    raise ValueError(
                        f"Categoria OCR {detected!r} não corresponde "
                        f"ao arquivo {category_hint!r}"
                    )
                if not any(str(info.get(k, "") or "").strip() for k in (
                    "board", "parts", "value"
                )):
                    raise ValueError("OCR vazio para identificação da inspeção")
                predictor = model_map.get(row["model"])
                if predictor is None:
                    raise ValueError("CNN do escopo indisponível")
                output = predictor.inspect(reference, test, row["lighting_mode"])
                score, verdict, checksum = _validate_cnn_output(output)
                row.update({
                    "verdict": verdict, "checkpoint_sha256": checksum,
                    "ng_score_uncalibrated": score,
                    "raw_binary_prediction": "NG" if score >= .5 else "OK",
                    "raw_binary_correct": (
                        (score >= .5) == (row["expected_label"] == "NG")
                    ),
                    "status": (
                        "REVISAO_OBRIGATORIA" if verdict == "REVISÃO OBRIGATÓRIA"
                        else "PASSOU" if verdict == VERDICT_FOR_LABEL[row["expected_label"]]
                        else "REGRESSAO"
                    ),
                })
        except Exception as exc:
            row["status"] = "INVALIDO"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if progress is not None:
            progress(i, len(items), row["source_path"])
    by_category = {}
    for category in sorted({canonical_memory_category(r["category_hint"]) for r in rows}):
        selected = [
            r for r in rows
            if canonical_memory_category(r["category_hint"]) == category
        ]
        by_category[category] = _summarize(selected)
    by_light = {}
    for light in sorted({r["lighting_mode"] for r in rows}):
        by_light[light] = _summarize([r for r in rows if r["lighting_mode"] == light])
    total = _summarize(rows)
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": "CNN_ONLY_READ_ONLY_HISTORICAL_REPLAY",
        "expected_retention_rate": MIN_RETENTION,
        "inventory_png_count": len(items),
        "model_scope": {
            MODEL_ABSENCE: [
                "FALTANDO", "EMBORCADO", "INVERTIDO", "DESLOCADO",
            ],
            "NO_CNN_AVAILABLE": ["MUITOADESIVO"],
        },
        "overall": total,
        "by_category": by_category,
        "by_lighting": by_light,
        "multilight_explicit_events": _event_report(rows),
        "cases": rows,
        "knn_used": False,
        "dataset_modified": False,
        "cnn_training_performed": False,
        "startup_gate_enabled": False,
        "production_approved": False,
        "historical_replay_is_not_independent_validation": True,
        "legacy_952_json_signatures_are_not_pngs": True,
        "note": (
            "98% significa >=98% dos PNGs totais classificados OK/NG "
            "conforme rótulos do arquivo; revisão, imagem inválida e "
            "categoria sem CNN não são acertos. O limiar aplicado ao "
            "score bruto de CNN é 0.5 apenas diagnóstico. Não retreinar "
            "com os casos de teste sem separação de eventos/placas. "
            "952 JSONs de memória incluem 815 sem imagens associadas e "
            "NÃO podem ser reinferidos como imagens por CNN."
        ),
    }


__all__ = [
    "SCHEMA", "MODEL_ABSENCE", "UNSUPPORTED",
    "replay_full_cnn_history",
]
