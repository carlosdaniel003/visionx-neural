"""Auditoria offline: CNN FALTANDO v2 como hipótese VISUAL para TODAS as categorias.

Não muda o roteador, não consulta KNN, não treina e nunca emite 0/1.
O rótulo da pasta é evidência de arquivo; não prova validação humana/NG físico.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

import cv2
import numpy as np

from src.services.startup_regression.archive_inventory import inventory_archives

SCHEMA = "visionx.faltando_cross_category_audit.v1"
VALID_VERDICTS = {"FALHA FALSA", "DEFEITO REAL", "REVISÃO OBRIGATÓRIA"}


def _archive_image(root: Path, item: dict) -> np.ndarray:
    """Validate the path, label, PNG bytes and declared source SHA before inference."""
    relative = item.get("path")
    label = item.get("expected_label")
    if not isinstance(relative, str) or label not in {"OK", "NG"}:
        raise ValueError("Origem/rótulo do arquivo inválido")
    relative_path = Path(relative)
    if (
        relative_path.is_absolute() or ".." in relative_path.parts
        or relative_path.parts[:2] != ("public", "ok_archive" if label == "OK" else "ng_archive")
        or relative_path.suffix.lower() != ".png"
    ):
        raise ValueError("PNG fora do arquivo OK/NG correspondente")
    base = (root / "public" / ("ok_archive" if label == "OK" else "ng_archive")).resolve()
    path = root / relative_path
    if not path.is_file() or path.is_symlink() or path.resolve() == base or base not in path.resolve().parents:
        raise ValueError("Arquivo inválido ou link simbólico")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != item.get("file_sha256"):
        raise ValueError("Imagem original mudou após inventário")
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Imagem não decodificada em RGB/BGR 3 canais")
    if [img.shape[1], img.shape[0]] != [item.get("width"), item.get("height")]:
        raise ValueError("Dimensões do PNG incompatíveis com inventário")
    return img


def _scored_result(output: dict) -> tuple[str, str | None, float | None]:
    if not isinstance(output, dict):
        raise ValueError("Inferência não devolveu diagnóstico estruturado")
    verdict = str(output.get("verdict", "") or "").strip().upper()
    if verdict not in VALID_VERDICTS:
        raise ValueError("Veredito fora do contrato CNN")
    detail = output.get("detail", {})
    if not isinstance(detail, dict) or not detail.get("cnn_v2_active", False):
        raise ValueError("Apenas a CNN FALTANDO v2 pode ser usada nesta auditoria")
    if verdict != "REVISÃO OBRIGATÓRIA" and (
        not detail.get("cnn_v2_checkpoint_verified", False)
        or not detail.get("cnn_v2_experimental", False)
    ):
        raise ValueError("Checkpoint/modelo experimental não confirmado")
    raw_score = detail.get("cnn_v2_ng_score_uncalibrated")
    if raw_score is None:
        return verdict, None, None
    try:
        score = float(raw_score)
    except (TypeError, ValueError) as exc:
        raise ValueError("Score NG inválido") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("Score NG não finito ou fora de [0,1]")
    # Proxy informativo somente; caso em revisão nunca vira OK/NG
    return verdict, "NG" if score >= .5 else "OK", round(score, 8)


def audit_cross_category(
    root: Path, *, inventory: dict | None = None,
    extractor=None, predictor=None,
) -> tuple[dict, Path]:
    root = Path(root).expanduser().resolve()
    evidence = inventory if inventory is not None else inventory_archives(root)
    if evidence.get("schema") != "visionx.archive_inventory.v1":
        raise ValueError("Inventário de arquivo incompatível")
    items = evidence.get("images", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Nenhuma imagem inventariada")
    if int(evidence.get("summary", {}).get("png_count", -1)) != len(items):
        raise ValueError("Cobertura do inventário incompatível")

    # Conservative: no inference from records whose label is contradicted by
    # identical pixels in the opposite archive.
    contradicted = {
        path for group in evidence.get("cross_label_conflicts", [])
        for path in group.get("paths", [])
    }
    results = []
    if extractor is None:
        from src.services.faltando_neural_dataset import AOIPairExtractor
        extractor = AOIPairExtractor()
    if predictor is None:
        from src.core.neural.faltando_live import FaltandoCNNLive
        predictor = FaltandoCNNLive()

    for item in items:
        original_label = item.get("expected_label")
        original_category = item.get("category_hint") or "UNKNOWN"
        row = {
            "source_path": item.get("path"),
            "source_sha256": item.get("file_sha256"),
            "archive_label": original_label,
            "archive_label_human_verified": False,
            "aoi_category_hint": original_category,
            "lighting_mode": item.get("lighting_mode", "SIDE"),
            "lighting_source": item.get("lighting_source"),
            "verified_event_id": item.get("event_id"),
            "evaluation_status": "NOT_EVALUATED",
            "cnn_verdict": None,
            "cnn_ng_score_uncalibrated": None,
            "raw_score_at_half_only": None,
            "label_agrees_with_cnn_decision": None,
            "issues": [],
        }
        try:
            if item.get("status") != "VALID_PNG":
                row["evaluation_status"] = "INVALID_PNG"
                row["issues"].append("Inventário rejeitou PNG")
            elif row["source_path"] in contradicted:
                row["evaluation_status"] = "CONTRADICTORY_LABEL"
                row["issues"].append("Mesmo conteúdo visual rotulado como OK e NG")
            elif original_label not in {"OK", "NG"}:
                row["evaluation_status"] = "INVALID_LABEL"
            else:
                frame = _archive_image(root, item)
                reference, test, observed = extractor(frame)
                if not isinstance(observed, dict):
                    raise ValueError("Metadados do par AOI inválidos")
                row["ocr_observed"] = {
                    key: str(observed.get(key, "") or "")
                    for key in ("category", "board", "parts", "value")
                }
                observed_category = row["ocr_observed"]["category"].strip().upper()
                if original_category not in {"UNKNOWN", ""} and observed_category not in {
                    "", "UNKNOWN", original_category,
                }:
                    row["issues"].append("CATEGORIA_OCR_DIFERE_DO_NOME")
                verdict, rough, score = _scored_result(
                    predictor.inspect(reference, test, row["lighting_mode"])
                )
                row["cnn_verdict"] = verdict
                row["cnn_ng_score_uncalibrated"] = score
                row["raw_score_at_half_only"] = rough
                if verdict == "REVISÃO OBRIGATÓRIA":
                    row["evaluation_status"] = "CNN_REVIEW"
                else:
                    judged = "NG" if verdict == "DEFEITO REAL" else "OK"
                    row["evaluation_status"] = "EVALUATED"
                    row["cnn_decision_for_diagnostics"] = judged
                    row["label_agrees_with_cnn_decision"] = judged == original_label
        except Exception as exc:
            row["evaluation_status"] = "INVALID_OR_FAILED"
            row["issues"].append(f"{type(exc).__name__}: {exc}")
        results.append(row)

    assert len(results) == len(items)
    def summarize(rows):
        decided = [r for r in rows if r["evaluation_status"] == "EVALUATED"]
        archive_ng = [r for r in decided if r["archive_label"] == "NG"]
        archive_ok = [r for r in decided if r["archive_label"] == "OK"]
        return {
            "count": len(rows),
            "status_counts": dict(Counter(r["evaluation_status"] for r in rows)),
            "archive_label_counts": dict(Counter(r["archive_label"] for r in rows)),
            "cnn_decisions": len(decided),
            "archived_ng_called_ok": sum(
                r["cnn_decision_for_diagnostics"] == "OK" for r in archive_ng
            ),
            "archived_ok_called_ng": sum(
                r["cnn_decision_for_diagnostics"] == "NG" for r in archive_ok
            ),
            "correct_against_archive_label": sum(
                bool(r["label_agrees_with_cnn_decision"]) for r in decided
            ),
            "cnn_review": sum(r["evaluation_status"] == "CNN_REVIEW" for r in rows),
        }
    categories = defaultdict(list)
    lights = defaultdict(list)
    for result in results:
        categories[str(result["aoi_category_hint"])].append(result)
        lights[str(result["lighting_mode"])].append(result)
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "OFFLINE_CROSS_CATEGORY_SHADOW_AUDIT",
        "method": "FALTANDO_CNN_V2_SINGLE_LIGHT_NO_KNN",
        "total_images": len(results),
        "summary": summarize(results),
        "by_category": {c: summarize(rows) for c, rows in sorted(categories.items())},
        "by_light": {c: summarize(rows) for c, rows in sorted(lights.items())},
        "inventory_issue_count": evidence.get("summary", {}).get("issue_count"),
        "cross_label_conflict_groups": len(evidence.get("cross_label_conflicts", [])),
        "category_is_original_aoi_hint_not_verified_defect_type": True,
        "archive_folder_is_not_proof_of_independent_human_truth": True,
        "legacy_and_multilight_are_evaluated_as_frames_not_independent_events": True,
        "includes_adhesive_in_audit_but_does_not_change_specialist": True,
        "images_may_have_been_seen_during_faltando_training": True,
        "checkpoints_and_training_modified": False,
        "knn_used": False,
        "production_modified": False,
        "cnn_cross_category_approved": False,
        "automatic_zero_or_one_enabled": False,
        "ng_generalization_proven": False,
        "warning": (
            "Comparação descritiva com pasta de arquivo. Erros NG->OK são "
            "críticos; nenhuma categoria nem decisão 0/1 está liberada."
        ),
        "images": results,
    }
    folder = (
        root / "reports" / "faltando_neural" / "cross_category_audit"
        / ("audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    )
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "cross_category_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "ODIN — CNN FALTANDO v2 APLICADA EM SOMBRA ÀS CATEGORIAS AOI",
        "NÃO TREINA | NÃO CONSULTA KNN | NÃO ALTERA PRODUÇÃO | NÃO ENVIA 0/1",
        "Arquivo OK/NG não prova rótulo humano nem generalização independente.",
        f"TOTAL: {len(results)}",
        f"GERAL: {report['summary']}", "",
        "POR CATEGORIA ORIGINAL AOI:",
    ]
    lines += [f"{name}: {value}" for name, value in report["by_category"].items()]
    lines += ["", "DIVERGÊNCIAS E CASOS NÃO DECIDIDOS:"]
    for item in results:
        if item["evaluation_status"] != "EVALUATED" or not item["label_agrees_with_cnn_decision"]:
            lines.append(
                f"{item['source_path']} | AOI={item['aoi_category_hint']} "
                f"| arquivo={item['archive_label']} | CNN={item['cnn_verdict']} "
                f"| status={item['evaluation_status']} | {item['issues']}"
            )
    (folder / "cross_category_audit.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report, folder
