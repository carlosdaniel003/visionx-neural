"""Auditoria INTEGRAL, offline, dos PNGs OK/NG contra a CNN V3 ativa.

Rótulos de pasta são referências provisórias; o teste mede concordância,
não demonstra verdade física nem acurácia cega em peças futuras.

Separa o classificador dos mecanismos de recuperação:
1. CNN V3 usa os dois recortes originais do mesmo extrator operacional.
2. Identidade de TELA é comparada por SHA-256 dos pixels.
3. Recuperação visual leave-one-out é somente uma sonda de vizinhança:
   não conta a própria imagem, não altera o voto e não libera 0/1 ao XP.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.services.image_archive_dedup import (
    archive_visual_descriptor, image_fingerprint, near_duplicate_visual,
)
from src.services.startup_regression.archive_inventory import inventory_archives

SCHEMA = "visionx.aoi_v3_full_archive_audit.v1"
PRIMARY_CATEGORIES = {"FALTANDO", "DESLOCADO", "INVERTIDO", "EMBORCADO"}
_LABELS = ("OK", "NG")
_LIGHTS = ("SIDE", "TOP", "MID")


def _read(path: Path) -> tuple[np.ndarray, str]:
    data = path.read_bytes()
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3):
        raise ValueError("PNG não pode ser lido em três canais BGR")
    return frame, sha256(data).hexdigest()


def _scope(item: dict) -> str:
    return (str(item.get("category_hint", "") or "").upper() + "_" +
            str(item.get("lighting", item.get("lighting_mode", "")) or "").upper())


def _coarse_vector(descriptor) -> np.ndarray | None:
    if descriptor is None:
        return None
    # Menos pixels para pré-selecionar, mas verificação visual posterior
    # utiliza TODOS os quatro mapas da mesma rotina de deduplicação.
    pieces = [
        cv2.resize(part, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
        for part in descriptor[1]
    ]
    vector = np.concatenate(pieces).astype(np.float32) / 255.0
    return vector


def _probe_memory(rows: list[dict], signatures: dict[str, tuple]) -> dict:
    """Leave-one-out em todas as categorias e luzes.

    Consulta de arquivo idêntico dentro do acervo e recuperação de
    nova imagem por similaridade são experimentos diferentes. Nenhum
    resultado desta sonda pode substituir uma correção humana.
    """
    exact_index = defaultdict(list)
    scoped = defaultdict(list)
    for row in rows:
        if row["status"] != "EVALUATED":
            continue
        path = row["path"]
        if row.get("pixel_sha256"):
            exact_index[row["pixel_sha256"]].append(row)
        if path in signatures:
            scoped[_scope(row)].append(row)

    vectors = {}
    for path, desc in signatures.items():
        vec = _coarse_vector(desc)
        if vec is not None:
            vectors[path] = vec

    for scope, group in scoped.items():
        group = [row for row in group if row["path"] in vectors]
        if len(group) < 2:
            for item in group:
                item["leave_one_out"] = {
                    "status": "NO_OTHER_SOURCE",
                    "own_image_excluded": True, "scope": scope,
                    "label_authorized": False,
                }
            continue
        stack = np.stack([vectors[row["path"]] for row in group])
        sq = np.einsum("ij,ij->i", stack, stack)
        for i, item in enumerate(group):
            distances = sq + sq[i] - 2 * (stack @ stack[i])
            distances[i] = np.inf
            nearest = np.argsort(distances)[:min(12, len(group)-1)]
            candidates = []
            for j in nearest:
                other = group[int(j)]
                matched = near_duplicate_visual(
                    signatures[item["path"]], signatures[other["path"]]
                )
                candidates.append({
                    "path": other["path"],
                    "folder_label": other["archive_label"],
                    "visual_near_duplicate": bool(matched),
                    "coarse_distance": round(float(max(0.0, distances[j])), 6),
                })
            strong = [c for c in candidates if c["visual_near_duplicate"]]
            # Mesmo se nenhum vizinho passar, registrar os candidatos.
            # Um top-k aproximado NÃO pode atestar ausência no acervo.
            strong_labels = sorted({c["folder_label"] for c in strong})
            status = (
                "NEAR_LABEL_CONFLICT" if len(strong_labels) > 1
                else "NEAR_OTHER_SOURCE" if strong_labels else
                "CANDIDATES_NOT_VERIFIED"
            )
            item["leave_one_out"] = {
                "status": status,
                "scope": scope,
                "own_image_excluded": True,
                "label_authorized": False,
                "source_count_in_scope": len(group),
                "checked_candidates": len(candidates),
                "candidates": candidates,
                "verified_near_labels": strong_labels,
                "same_label_retrieved": (
                    bool(strong_labels)
                    and len(strong_labels) == 1
                    and strong_labels[0] == item["archive_label"]
                ),
            }

    for row in rows:
        if row["status"] != "EVALUATED":
            continue
        siblings = [
            r for r in exact_index.get(row["pixel_sha256"], [])
            if r["path"] != row["path"]
        ]
        row["full_frame_exact"] = {
            "own_source_in_index": True,
            "own_source_is_not_independent_validation": True,
            "other_paths": [r["path"] for r in siblings],
            "other_labels": sorted({r["archive_label"] for r in siblings}),
            "conflict": len({r["archive_label"] for r in
                             exact_index[row["pixel_sha256"]]}) > 1,
        }
        if "leave_one_out" not in row:
            row["leave_one_out"] = {
                "status": "NOT_INDEXED",
                "scope": _scope(row), "own_image_excluded": True,
                "label_authorized": False,
            }
    return {
        "total_probed": sum(x["status"] == "EVALUATED" for x in rows),
        "same_label_near": sum(
            x.get("leave_one_out", {}).get("same_label_retrieved", False)
            for x in rows
        ),
        "conflicting_near": sum(
            x.get("leave_one_out", {}).get("status") == "NEAR_LABEL_CONFLICT"
            for x in rows
        ),
        "exact_cross_label_conflicts": sum(
            x.get("full_frame_exact", {}).get("conflict", False)
            for x in rows
        ),
        "probe_is_diagnostic_only": True,
        "not_an_out_of_sample_accuracy_estimate": True,
    }


def _metrics(rows: list[dict]) -> dict:
    considered = [x for x in rows if x["status"] == "EVALUATED"]
    counts = Counter((x["archive_label"], x["predicted"]) for x in considered)
    total = len(considered)
    right = counts[("OK", "OK")] + counts[("NG", "NG")]
    ok_total = sum(x["archive_label"] == "OK" for x in considered)
    ng_total = total - ok_total
    return {
        "evaluated": total,
        "correct_against_folder": right,
        "accuracy_against_folder": round(right / total, 6) if total else None,
        "OK_total": ok_total,
        "OK_correct": counts[("OK", "OK")],
        "OK_false_NG": counts[("OK", "NG")],
        "NG_total": ng_total,
        "NG_correct": counts[("NG", "NG")],
        "NG_false_OK": counts[("NG", "OK")],
        "balanced_accuracy_against_folder": round(
            (counts[("OK", "OK")]/ok_total + counts[("NG", "NG")]/ng_total)/2, 6
        ) if ok_total and ng_total else None,
        "failed_or_unavailable": len(rows) - total,
    }


def audit_full_archive(
    root: Path, *, extractor=None, predictor=None,
    progress: Callable[[int, int, dict], None] | None = None,
    inventory: dict | None = None,
) -> dict:
    """Reavalia todos os PNGs do inventário; não faz treino ou writes.

    Funções injetáveis permitem testar a cobertura sem Torch ou máquina XP.
    """
    root = Path(root).resolve()
    if predictor is None:
        from src.core.neural.aoi_quality_cnn_v3_primary import AOIQualityV3Primary
        predictor = AOIQualityV3Primary(root=root)
    warmup = getattr(predictor, "warmup", None)
    if callable(warmup):
        checked = warmup()
        if not isinstance(checked, dict) or checked.get("state") != "ready":
            raise ValueError("Checkpoint CNN V3 indisponível: " + str(checked))
    if inventory is None:
        inventory = inventory_archives(root)
    if inventory.get("schema") != "visionx.archive_inventory.v1":
        raise ValueError("Inventário de acervo incompatível")
    items = inventory.get("images", [])
    if not isinstance(items, list) or not items:
        raise ValueError("Acervo vazio ou sem imagens inventariadas")
    if len(items) != inventory.get("summary", {}).get("png_count"):
        raise ValueError("Inventário não cobre todos os PNGs")
    if extractor is None:
        from src.services.faltando_neural_dataset import AOIPairExtractor
        extractor = AOIPairExtractor()
    rows = []
    signatures = {}
    initial_hashes = {}
    for number, item in enumerate(items, start=1):
        path = str(item.get("path", ""))
        source = (root / path).resolve()
        expected = item.get("expected_label")
        light = str(item.get("lighting_mode", "") or "").upper()
        category = str(item.get("category_hint", "") or "").upper()
        result = {
            "path": path,
            "archive_label": expected,
            "label_provenance": "ARCHIVE_FOLDER_PROVISIONAL",
            "archive_label_physically_verified": False,
            "category_hint": category,
            "lighting": light,
            "model_role": (
                "PRIMARY_RUNTIME" if category in PRIMARY_CATEGORIES
                else "V3_COMPARATIVE_ONLY"
            ),
            "status": "NOT_EVALUATED",
            "predicted": None,
            "v3_ng_score_uncalibrated": None,
            "source_file_sha256": item.get("file_sha256", ""),
            "pixel_sha256": item.get("pixel_sha256", ""),
            "issues": [],
        }
        try:
            if expected not in _LABELS or light not in _LIGHTS:
                raise ValueError("Rótulo/luz do inventário inválido")
            if source.parent not in {
                (root / "public/ok_archive").resolve(),
                (root / "public/ng_archive").resolve(),
            }:
                raise ValueError("PNG não está na raiz do acervo válido")
            if item.get("status") != "VALID_PNG":
                raise ValueError("PNG inválido no inventário")
            frame, file_hash = _read(source)
            if file_hash != item.get("file_sha256"):
                raise ValueError("PNG alterado após o inventário inicial")
            fingerprint = image_fingerprint(frame)
            # Para PNG 3 canais (padrão operacional), exige hash coerente.
            if int(item.get("channels") or 0) == 3 and (
                fingerprint != item.get("pixel_sha256")
            ):
                raise ValueError("Pixels mudaram após o inventário")
            reference, test, info = extractor(frame)
            if any(
                not isinstance(im, np.ndarray) or im.size == 0 or
                im.ndim != 3 or im.shape[-1] != 3
                for im in (reference, test)
            ):
                raise ValueError("Extração AOI gerou par inválido")
            result.update(
                reference_shape=list(reference.shape),
                test_shape=list(test.shape),
                reference_sha256=image_fingerprint(reference),
                test_sha256=image_fingerprint(test),
                extraction_category=str((info or {}).get("category", "") or ""),
                extraction_geometry=dict(
                    getattr(getattr(extractor, "monitor", None),
                            "last_crop_geometry", {}) or {}
                ),
            )
            vote = predictor.inspect(reference, test, light)
            detail = vote.get("detail", {}) if isinstance(vote, dict) else {}
            status = detail.get("cnn_v3_status")
            score = detail.get("cnn_v3_ng_score_uncalibrated")
            predicted = (
                "NG" if vote.get("verdict") == "DEFEITO REAL"
                else "OK" if vote.get("verdict") == "FALHA FALSA"
                else None
            )
            if (
                status != "INFERENCE_OK" or predicted is None
                or isinstance(score, bool) or not isinstance(score, (float, int))
                or not math.isfinite(score) or not 0 <= score <= 1
                or (score >= .5) != (predicted == "NG")
                or detail.get("cnn_v3_checkpoint_verified") is not True
            ):
                raise ValueError(
                    "CNN V3 não retornou inferência e checkpoint verificados"
                )
            result.update(
                status="EVALUATED",
                predicted=predicted,
                v3_ng_score_uncalibrated=float(score),
                v3_checkpoint_sha256=detail.get("cnn_v3_checkpoint_sha256"),
                agrees_with_folder=predicted == expected,
            )
            desc = archive_visual_descriptor(frame)
            if desc is not None:
                signatures[path] = desc
            initial_hashes[path] = file_hash
        except (OSError, ValueError, TypeError, cv2.error) as exc:
            result["status"] = "UNAVAILABLE"
            result["issues"].append(type(exc).__name__ + ": " + str(exc))
        rows.append(result)
        if progress:
            progress(number, len(items), result)

    probes = _probe_memory(rows, signatures)
    # Não publicar relatório como se fosse conjunto imutável se algo mudou
    # durante a execução longa: rechecagem de bytes das fontes processadas.
    changed = []
    for path, expected_hash in initial_hashes.items():
        try:
            current_hash = sha256((root / path).read_bytes()).hexdigest()
            if current_hash != expected_hash:
                changed.append(path)
        except OSError:
            changed.append(path)

    category_rows = defaultdict(list)
    light_rows = defaultdict(list)
    group_rows = defaultdict(list)
    for result in rows:
        category_rows[result["category_hint"]].append(result)
        light_rows[result["lighting"]].append(result)
        group_rows[_scope(result)].append(result)
    stats = _metrics(rows)
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "production_decision_authorized": False,
        "archive_label_physically_verified": False,
        "blind_holdout": False,
        "folder_labels_are_provisional": True,
        "all_files_evaluated": (
            len(changed) == 0 and stats["failed_or_unavailable"] == 0
        ),
        "root": str(root),
        "coverage": {
            "png_in_inventory": len(items),
            "rows_in_report": len(rows),
            "source_changed_during_run": changed,
            "invalid_inventory_pngs": inventory["summary"].get("invalid_png", 0),
            "cross_label_pixel_conflicts": inventory.get("cross_label_conflicts", []),
            "by_folder": inventory["summary"].get("by_label", {}),
        },
        "classifier": {
            "name": "AOIQualityV3Primary",
            "v3_checkpoint_sha256": sorted({
                r.get("v3_checkpoint_sha256")
                for r in rows if r.get("v3_checkpoint_sha256")
            }),
            "measured_on_original_png_not_mss": True,
            "threshold": 0.5,
            "all_categories_v3_comparison_includes_non_primary_runtime": True,
        },
        "summary": stats,
        "by_category": {key: _metrics(value)
                        for key, value in sorted(category_rows.items())},
        "by_light": {key: _metrics(value)
                     for key, value in sorted(light_rows.items())},
        "by_category_light": {key: _metrics(value)
                              for key, value in sorted(group_rows.items())},
        "leave_one_out_archive_probe": probes,
        "cases": rows,
    }


__all__ = ["audit_full_archive", "SCHEMA"]
