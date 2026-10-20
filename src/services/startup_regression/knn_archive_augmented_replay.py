"""Teste OFFLINE de robustez OK/NG com variações leves dos PNGs do acervo.

As imagens originais formam a referência KNN temporária. Apenas CONSULTAS
sintéticas (nunca gravadas) são perturbadas. A origem da imagem/mesmo evento
é excluída dos vizinhos, prevenindo o acerto trivial de uma cópia.
Categoria OCR é só metadado: jamais decide rótulo, vizinhança ou validade.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import math

import cv2
import numpy as np

from src.core.experts.knn_expert import KNNExpert
from src.core.knn_similarity_shadow_candidate import (
    select_shadow_candidate, summarize_shadow_candidates,
)
from src.services.faltando_cross_category_audit import _archive_image
from src.services.startup_regression.archive_inventory import inventory_archives
from src.services.startup_regression.knn_archive_concordance_experiment import (
    select_concordance_candidate,
)
from src.services.startup_regression.knn_archive_focus_signature import (
    _green_mask, build_focus_signature, compare_focus_signatures,
    valid_focus_signature,
)
from src.services.startup_regression.knn_archive_geometry_experiment import (
    build_geometry_signature, compare_geometry_hybrid,
)
from src.services.startup_regression.knn_archive_visual_replay import (
    _capture_group, _excluded, _measure,
)

SCHEMA = "visionx.knn_archive_augmented_robustness.v1"
TRANSFORMS = ("exposure_and_sensor_noise", "shared_one_pixel_shift_and_soft_focus")


def perturb_aoi_pair(reference: np.ndarray, capture: np.ndarray,
                     source_path: str, variant: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Alterações pequenas e reproduzíveis, sem fabricar NG nem mudar gabarito.

    Transformações de um arquivo OK continuam apenas *hipóteses OK*. Não se
    adiciona/remova componente, nem se desloca só a captura criando NG falso.
    """
    if variant not in (0, 1):
        raise ValueError("Somente as duas perturbações leves suportadas")
    if (not isinstance(reference, np.ndarray)
            or not isinstance(capture, np.ndarray)
            or reference.dtype != np.uint8 or capture.dtype != np.uint8
            or reference.ndim != 3 or capture.ndim != 3
            or reference.shape != capture.shape
            or reference.shape[2] != 3
            or min(reference.shape[:2]) < 12):
        raise ValueError("Par AOI inválido para variação de captura")
    seed = int.from_bytes(
        sha256(f"{source_path}|{variant}|knn_aug_v1".encode()).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(seed)
    ref, test = reference.copy(), capture.copy()
    if variant == 0:
        # Variação de exposição e ruído leve da câmera na captura somente.
        gain = float(rng.choice([0.985, 1.015]))
        bias = int(rng.choice([-3, 3]))
        noise = rng.normal(0, 1.25, test.shape)
        altered = np.clip(test.astype(np.float32) * gain + bias + noise, 0, 255)
        test = altered.astype(np.uint8)
        # A anotação verde da AOI não é iluminação física nem defeito.
        test[_green_mask(capture) > 0] = capture[_green_mask(capture) > 0]
        settings = {"gain": gain, "bias": bias, "noise_sigma": 1.25}
    else:
        # Jitter de 1 px aplicado AO MESMO TEMPO aos dois lados.
        # Não altera o alinhamento relativo de gabarito e captura.
        dx = int(rng.choice([-1, 1]))
        dy = int(rng.choice([-1, 1]))
        affine = np.float32([[1, 0, dx], [0, 1, dy]])
        h, w = reference.shape[:2]
        ref = cv2.warpAffine(
            reference, affine, (w, h), borderMode=cv2.BORDER_REFLECT_101,
        )
        test = cv2.warpAffine(
            capture, affine, (w, h), borderMode=cv2.BORDER_REFLECT_101,
        )
        original_marks = _green_mask(test) > 0
        blurred = cv2.GaussianBlur(test, (3, 3), .55)
        blurred[original_marks] = test[original_marks]
        test = blurred
        settings = {"dx_both": dx, "dy_both": dy, "blur_sigma_capture": .55}
    return ref, test, {"name": TRANSFORMS[variant], **settings}


def _neighbors(query_focus: dict, query_geometry: dict,
               eligible: list[dict], k: int, *, geometry: bool) -> list[dict]:
    scores = []
    for example in eligible:
        if geometry:
            sim, _ = compare_geometry_hybrid(
                query_focus, example["focus_signature"],
                query_geometry, example["geometry_signature"],
            )
        else:
            sim, _ = compare_focus_signatures(
                query_focus, example["focus_signature"]
            )
        if math.isfinite(sim):
            scores.append((float(sim), example))
    scores.sort(key=lambda pair: (-pair[0], pair[1]["source_path"]))
    seen = set()
    selected = []
    for similarity, other in scores:
        digest = other["source_pixel_sha256"] or other["source_file_sha256"]
        if digest and digest in seen:
            continue
        if digest:
            seen.add(digest)
        selected.append({
            "source_path": other["source_path"],
            "source_pixel_sha256": digest,
            "label": other["expected_label"],
            "similarity": round(similarity, 8),
        })
        if len(selected) >= k:
            break
    return selected


def _judge(neighbors: list[dict], lighting: str,
           reference_counts: dict[str, int]) -> dict:
    score = None
    prediction = None
    if neighbors:
        score = KNNExpert._weighted_vote([
            (1.0 - n["similarity"], n["label"]) for n in neighbors
        ])
        if math.isfinite(score):
            prediction = "NG" if score > 0.5 else "OK"
    selective = select_shadow_candidate(
        neighbors, category="OK_NG_VISUAL_SEM_CATEGORIA_AOI",
        lighting=lighting, scope_examples=reference_counts,
    )
    return {
        "prediction": prediction,
        "score_ng": round(float(score), 8) if score is not None and math.isfinite(score) else None,
        "shadow_candidate": selective,
        "neighbors": neighbors,
    }


def replay_augmented_archive(
    root: Path, *, variations: int = 2, k: int = 5,
    inventory: dict | None = None, extractor=None, frame_loader=None,
    progress=None, cancelled=None,
) -> dict:
    """Testa variações com referência fixa: NÃO cria memória nem dataset.

    IMPORTANTE: a referência contém os outros eventos originais; proíbe
    a origem da consulta e seu prefixo/event_id, inclusive demais luzes.
    Categoria AOI é intencionalmente ignorada. Mantém a iluminação para
    não comparar câmeras com distribuições distintas.
    """
    if type(variations) is not int or variations not in (1, 2):
        raise ValueError("variations precisa ser 1 ou 2")
    if type(k) is not int or not 1 <= k <= 25:
        raise ValueError("k precisa ser inteiro entre 1 e 25")
    root = Path(root).expanduser().resolve()
    inv = inventory if inventory is not None else inventory_archives(root)
    items = inv.get("images")
    if (inv.get("schema") != "visionx.archive_inventory.v1"
            or not isinstance(items, list) or not items
            or inv.get("summary", {}).get("png_count") != len(items)):
        raise ValueError("Inventário PNG incompleto")
    if extractor is None:
        from src.services.faltando_neural_dataset import AOIPairExtractor
        extractor = AOIPairExtractor()
    if frame_loader is None:
        frame_loader = _archive_image

    conflict_paths = {
        path for conflict in inv.get("cross_label_conflicts", [])
        for path in conflict.get("paths", [])
    }
    originals = []
    failed_originals = {}
    count = len(items)
    interrupted = False
    for n, item in enumerate(items, 1):
        if cancelled is not None and cancelled():
            interrupted = True
            break
        path = item.get("path", "")
        try:
            if item.get("expected_label") not in ("OK", "NG"):
                raise ValueError("Rótulo OK/NG ausente")
            if item.get("status") != "VALID_PNG" or path in conflict_paths:
                raise ValueError("PNG inválido ou hash OK/NG conflitante")
            reference, capture, _info = extractor(frame_loader(root, item))
            # Nenhum gate por categoria da AOI: metadado pode estar errado.
            focus = build_focus_signature(reference, capture)
            if not valid_focus_signature(focus):
                raise ValueError("Assinatura AOI inválida")
            geometry = build_geometry_signature(reference, capture, focus["focus"])
            originals.append({
                "source_path": path,
                "source_file_sha256": item.get("file_sha256", ""),
                "source_pixel_sha256": item.get("pixel_sha256", ""),
                "expected_label": item["expected_label"],
                "lighting_mode": item.get("lighting_mode", "SIDE"),
                "event_id": item.get("event_id"),
                "capture_prefix_hint": _capture_group(path),
                "category_hint": item.get("category_hint"),  # relatório APENAS
                "reference": reference,
                "capture": capture,
                "focus_signature": focus,
                "geometry_signature": geometry,
            })
        except Exception as error:
            failed_originals[path] = f"{type(error).__name__}: {error}"
        if progress is not None:
            progress("extract", n, count, path)

    pools = defaultdict(list)
    for row in originals:
        pools[row["lighting_mode"]].append(row)
    rows = []
    completed_queries = 0
    for n, item in enumerate(items, 1):
        src = next((r for r in originals if r["source_path"] == item.get("path")), None)
        for variant in range(variations):
            if cancelled is not None and cancelled():
                interrupted = True
            row = {
                "source_path": item.get("path", ""),
                "expected_label": item.get("expected_label"),
                "lighting_mode": item.get("lighting_mode", "SIDE"),
                "category_hint": item.get("category_hint"),
                "variation": TRANSFORMS[variant],
                "variation_details": None,
                "status": "NAO_ANALISADO" if interrupted else "INVALIDO",
                "prediction": None,
                "error": failed_originals.get(item.get("path", "")),
                "shadow_candidate": {"status": "REVISAO"},
                "geometry_experiment": {
                    "prediction": None,
                    "shadow_candidate": {"status": "REVISAO"},
                },
                "geometry_consensus_candidate": {"status": "REVISAO"},
            }
            if src is None or interrupted:
                rows.append(row)
                continue
            try:
                q_ref, q_test, details = perturb_aoi_pair(
                    src["reference"], src["capture"], src["source_path"], variant
                )
                q_focus = build_focus_signature(q_ref, q_test)
                q_geo = build_geometry_signature(q_ref, q_test, q_focus["focus"])
                if not valid_focus_signature(q_focus):
                    raise ValueError("Variação sem assinatura visual válida")
                eligible = [
                    ex for ex in pools[src["lighting_mode"]]
                    if not _excluded(src, ex)
                ]
                refs = {"OK": set(), "NG": set()}
                for ex in eligible:
                    digest = ex["source_pixel_sha256"] or ex["source_file_sha256"]
                    if digest and ex["expected_label"] in refs:
                        refs[ex["expected_label"]].add(digest)
                support = {name: len(ids) for name, ids in refs.items()}
                aa = _judge(_neighbors(
                    q_focus, q_geo, eligible, k, geometry=False
                ), src["lighting_mode"], support)
                bb = _judge(_neighbors(
                    q_focus, q_geo, eligible, k, geometry=True
                ), src["lighting_mode"], support)
                cc = select_concordance_candidate(
                    aa["shadow_candidate"], bb["shadow_candidate"],
                    visual_prediction=aa["prediction"],
                    geometry_prediction=bb["prediction"],
                )
                row.update({
                    "status": "PASSOU" if aa["prediction"] == row["expected_label"]
                    else "REGRESSAO" if aa["prediction"] else "SEM_COBERTURA",
                    "prediction": aa["prediction"],
                    "variation_details": details,
                    "shadow_candidate": aa["shadow_candidate"],
                    "neighbors": aa["neighbors"],
                    "geometry_experiment": bb,
                    "geometry_consensus_candidate": cc,
                })
            except Exception as error:
                row["status"] = "INVALIDO"
                row["error"] = f"{type(error).__name__}: {error}"
            completed_queries += 1
            rows.append(row)
        if progress is not None:
            progress("test", n, count, item.get("path", ""))

    # Consultas apenas: descarta imagens das métricas, não grava PNGs.
    visual = _measure(rows)
    geometry_rows = [{
        **row,
        "prediction": row["geometry_experiment"]["prediction"],
        "status": (
            "PASSOU" if row["geometry_experiment"]["prediction"] == row["expected_label"]
            else "REGRESSAO" if row["geometry_experiment"]["prediction"] else row["status"]
        ),
        "shadow_candidate": row["geometry_experiment"]["shadow_candidate"],
    } for row in rows]
    consensus_rows = [{
        **row, "shadow_candidate": row["geometry_consensus_candidate"],
    } for row in rows]
    complete = not interrupted and not failed_originals and (
        completed_queries == len(items) * variations
    ) and all(r["status"] not in {"INVALIDO", "NAO_ANALISADO"} for r in rows)
    if not complete:
        visual["target_98pct_met"] = False
    geometry_metrics = _measure(geometry_rows)
    if not complete:
        geometry_metrics["target_98pct_met"] = False
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "complete": complete,
        "canceled": interrupted,
        "inventory_png_count": count,
        "originals_decoded": len(originals),
        "original_errors": failed_originals,
        "variations_per_original": variations,
        "expected_query_count": count * variations,
        "queries_completed": completed_queries,
        "category_used_for_selection": False,
        "same_lighting_only": True,
        "excludes_source_and_same_event": True,
        "originals_as_neighbors_only": True,
        "synthetic_images_written": False,
        "writes_memory_or_dataset": False,
        "is_production_approved": False,
        "results": visual,
        "shadow_candidate": summarize_shadow_candidates(rows, complete=complete),
        "geometry_experiment": {
            "results": geometry_metrics,
            "shadow_candidate": summarize_shadow_candidates(
                geometry_rows, complete=complete
            ),
            "consensus_candidate": summarize_shadow_candidates(
                consensus_rows, complete=complete
            ),
        },
        "by_variant": {
            TRANSFORMS[variant]: {
                "results": _measure([
                    r for r in rows if r["variation"] == TRANSFORMS[variant]
                ]),
                "consensus_candidate": summarize_shadow_candidates(
                    [r for r in consensus_rows
                     if r["variation"] == TRANSFORMS[variant]],
                    complete=complete,
                ),
            }
            for variant in range(variations)
        },
        "cases": rows,
        "limitations": (
            "Variações artificiais leves preservam os rótulos presumidos do "
            "arquivo, mas NÃO garantem que falhas limítrofes continuariam OK/NG "
            "para auditoria humana. Mesmos componentes/lotes e iluminação "
            "podem estar correlacionados. Não fabrica defeitos NG a partir de OK. "
            "Teste de robustez sem treinamento não melhora a KNN de produção "
            "e não comprova precisão de peças novas."
        ),
    }


__all__ = ["SCHEMA", "TRANSFORMS", "perturb_aoi_pair", "replay_augmented_archive"]
