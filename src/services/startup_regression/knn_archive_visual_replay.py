"""Replay KNN visual EXCLUSIVAMENTE em public/ok_archive e ng_archive.

Protocolo visual_holdout: cada PNG é consultado e comparado com OUTROS
pares AOI da mesma categoria e iluminação, sem qualquer regra de data.
O comparador pesa o gabarito vs. captura em três escalas: imagem global,
moldura maior e epicentro menor (quando marcados pela AOI).
Mesmo evento/prefixo e conteúdo duplicado não são vizinhos.
O voto ponderado usa KNNExpert._weighted_vote (k=5).

Isto mede generalização de KNN construída TEMPORARIAMENTE sobre os PNGs
arquivados. NÃO é o lookup de pares exatos da VerifiedKNNMemory em
produção, NÃO lê os 952 JSONs, NÃO importa, altera ou treina memória.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import math
import re
from typing import Callable

from src.services.startup_regression.knn_archive_focus_signature import (
    build_focus_signature, compare_focus_signatures, valid_focus_signature,
)
from src.services.startup_regression.knn_archive_concordance_experiment import (
    select_concordance_candidate,
)
from src.services.startup_regression.knn_archive_geometry_experiment import (
    build_geometry_signature, compare_geometry_hybrid,
)
from src.core.experts.knn_expert import KNNExpert
from src.core.knn_similarity_shadow_candidate import (
    select_shadow_candidate, summarize_shadow_candidates,
)
from src.core.strict_category_memory import canonical_memory_category
from src.services.faltando_cross_category_audit import _archive_image
from src.services.archive_aoi_category import resolve_archive_aoi_category
from src.services.startup_regression.archive_inventory import inventory_archives

SCHEMA = "visionx.knn_archive_visual_replay.v1"
VALID_PROTOCOLS = ("visual_holdout",)
TRAILING_COPY = re.compile(r"_(?:[2-9]|[1-9]\d+)$", re.IGNORECASE)
TRAILING_LIGHT = re.compile(r"_(SIDE|TOP|MID)$", re.IGNORECASE)


def _capture_group(path: str) -> str:
    """Conservative prefix grouping; NOT proof of a multilight event."""
    stem = Path(path).stem.upper()
    stem = TRAILING_COPY.sub("", stem)
    return TRAILING_LIGHT.sub("", stem)


def _row(item: dict) -> dict:
    path = item.get("path", "")
    return {
        "source_path": path,
        "source_file_sha256": item.get("file_sha256", ""),
        "source_pixel_sha256": item.get("pixel_sha256", ""),
        "expected_label": item.get("expected_label"),
        "category_hint": item.get("category_hint"),
        "lighting_mode": item.get("lighting_mode", "SIDE"),
        "lighting_source": item.get("lighting_source"),
        "event_id": item.get("event_id"),
        "capture_prefix_hint": _capture_group(path),
        "ocr_observed_category": None,
        "focus": None,
        "signature_map_peak": None,
        "knn_score_ng": None,
        "prediction": None,
        "neighbors_count": 0,
        "neighbors_ok": 0,
        "neighbors_ng": 0,
        "best_similarity": None,
        "neighbors": [],
        "status": "NAO_ANALISADO",
        "error": None,
    }


def _measure(rows: list[dict]) -> dict:
    counts = Counter((r.get("expected_label"), r.get("prediction")) for r in rows)
    by_label = {}
    for label in ("OK", "NG"):
        subset = [r for r in rows if r.get("expected_label") == label]
        correct = sum(r.get("prediction") == label for r in subset)
        n = len(subset)
        rate = correct / n if n else 0.0
        by_label[label] = {
            "total": n, "correct": correct,
            "rate": round(rate, 6),
            "as_OK": counts[(label, "OK")],
            "as_NG": counts[(label, "NG")],
            "no_decision": n - counts[(label, "OK")] - counts[(label, "NG")],
        }
    return {
        "total": len(rows),
        "correct": sum(r.get("prediction") == r.get("expected_label")
                       for r in rows if r.get("prediction") in ("OK", "NG")),
        "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
        "by_label": by_label,
        "OK_as_NG": by_label["OK"]["as_NG"],
        "NG_as_OK": by_label["NG"]["as_OK"],
        "target_98pct_met": (
            all(by_label[label]["total"] > 0
                and by_label[label]["correct"] / by_label[label]["total"] >= 0.98
                for label in ("OK", "NG"))
        ),
    }


def _excluded(query: dict, candidate: dict, protocol: str = "visual_holdout") -> bool:
    if query["source_path"] == candidate["source_path"]:
        return True
    if (query["source_file_sha256"]
            and query["source_file_sha256"] == candidate["source_file_sha256"]):
        return True
    if (query["source_pixel_sha256"]
            and query["source_pixel_sha256"] == candidate["source_pixel_sha256"]):
        return True
    if query.get("event_id") and query["event_id"] == candidate.get("event_id"):
        return True
    if (query["capture_prefix_hint"]
            and query["capture_prefix_hint"] == candidate["capture_prefix_hint"]):
        return True
    return False


def replay_knn_archive_visual(
    root: Path, *, k: int = 5, protocol: str = "visual_holdout",
    inventory: dict | None = None,
    extractor: Callable | None = None,
    frame_loader: Callable | None = None,
    geometry_experiment: bool = False,
    progress: Callable | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """Avalia todos os PNGs, com vizinhos SEM a própria consulta.

    Injeções de inventory/extractor/frame_loader são para testes unitários;
    a execução padrão sempre usa inventário, hash e extração AOI reais.
    """
    root = Path(root).expanduser().resolve()
    if type(k) is not int or not 1 <= k <= 25:
        raise ValueError("k precisa ser inteiro entre 1 e 25")
    if protocol not in VALID_PROTOCOLS:
        raise ValueError("Protocolo KNN desconhecido")

    inv = inventory if inventory is not None else inventory_archives(root)
    items = inv.get("images")
    if (inv.get("schema") != "visionx.archive_inventory.v1"
            or not isinstance(items, list) or not items
            or len(items) != inv.get("summary", {}).get("png_count")):
        raise ValueError("Inventário PNG incompleto/ausente")

    if extractor is None:
        from src.services.faltando_neural_dataset import AOIPairExtractor
        extractor = AOIPairExtractor()
    if frame_loader is None:
        frame_loader = _archive_image

    conflicting = {
        path for conflict in inv.get("cross_label_conflicts", [])
        for path in conflict.get("paths", [])
    }
    # Inicializa os denominadores antes do replay; cancelamento não permite
    # que imagens restantes desapareçam das estatísticas como acertos.
    rows = [_row(item) for item in items]
    signatures = {}
    geometry_signatures = {}
    canceled = False
    for i, (item, row) in enumerate(zip(items, rows), 1):
        if cancelled is not None and cancelled():
            canceled = True
            break
        try:
            if row["expected_label"] not in ("OK", "NG"):
                raise ValueError("Rótulo fora das pastas OK/NG")
            if item.get("status") != "VALID_PNG":
                raise ValueError("PNG inválido no inventário")
            if row["source_path"] in conflicting:
                raise ValueError("Pixels idênticos com rótulos OK e NG")
            frame = frame_loader(root, item)
            reference, test, info = extractor(frame)
            if not isinstance(info, dict):
                raise ValueError("Extrator AOI não informou metadados")
            category, normalized_value = resolve_archive_aoi_category(
                info, row["category_hint"],
            )
            row["ocr_observed_category"] = category
            signature = build_focus_signature(reference, test)
            if not valid_focus_signature(signature):
                raise ValueError("Assinatura visual de gabarito/epicentro inválida")
            row["focus"] = signature["focus"]
            row["signature_map_peak"] = max(
                max(r) for r in signature["global_map"]
            )
            signatures[row["source_path"]] = signature
            if geometry_experiment:
                try:
                    geometry_signatures[row["source_path"]] = (
                        build_geometry_signature(reference, test, signature["focus"])
                    )
                except Exception as geometry_error:
                    # Erros do experimento não invalidam o controle v3.
                    row["geometry_error"] = (
                        f"{type(geometry_error).__name__}: {geometry_error}"
                    )
            row["status"] = "PRONTO_PARA_KNN"
        except Exception as exc:
            row["status"] = "INVALIDO"
            row["error"] = f"{type(exc).__name__}: {exc}"
        if progress is not None:
            progress(i, len(items), row["source_path"])

    pools = defaultdict(list)
    for row in rows:
        if row["status"] == "PRONTO_PARA_KNN":
            pools[(
                canonical_memory_category(row["ocr_observed_category"]),
                row["lighting_mode"],
            )].append(row)

    for row in rows:
        if cancelled is not None and cancelled():
            canceled = True
            break
        if row["status"] != "PRONTO_PARA_KNN":
            continue
        selected = []
        geometry_selected = []
        for other in pools[(
            canonical_memory_category(row["ocr_observed_category"]),
            row["lighting_mode"],
        )]:
            if _excluded(row, other, protocol):
                continue
            similarity, _breakdown = compare_focus_signatures(
                signatures[row["source_path"]], signatures[other["source_path"]]
            )
            if not math.isfinite(similarity):
                row["status"] = "INVALIDO"
                row["error"] = "Similaridade não finita"
                break
            selected.append((1.0 - similarity, other["expected_label"], other))
            if (geometry_experiment
                    and row["source_path"] in geometry_signatures
                    and other["source_path"] in geometry_signatures):
                hybrid, _parts = compare_geometry_hybrid(
                    signatures[row["source_path"]], signatures[other["source_path"]],
                    geometry_signatures[row["source_path"]],
                    geometry_signatures[other["source_path"]],
                )
                if math.isfinite(hybrid):
                    geometry_selected.append(
                        (1.0 - hybrid, other["expected_label"], other)
                    )
        if row["status"] == "INVALIDO":
            continue
        selected.sort(key=lambda item: (item[0], item[2]["source_path"]))
        # Colapsa vizinhos de mesmo conteúdo: uma imagem copiada não pode
        # multiplicar votos. Hash de pixels é independente do nome do arquivo.
        neighbors = []
        seen_hashes = set()
        for distance, label, other in selected:
            digest = other["source_pixel_sha256"] or other["source_file_sha256"]
            if digest and digest in seen_hashes:
                continue
            if digest:
                seen_hashes.add(digest)
            neighbors.append((distance, label, other))
            if len(neighbors) >= k:
                break
        row["neighbors_count"] = len(neighbors)
        row["neighbors_ok"] = sum(n[1] == "OK" for n in neighbors)
        row["neighbors_ng"] = sum(n[1] == "NG" for n in neighbors)
        row["neighbors"] = [
            {"source_path": n[2]["source_path"], "label": n[1],
             "source_pixel_sha256": n[2]["source_pixel_sha256"],
             "similarity": round(1.0 - n[0], 8)}
            for n in neighbors
        ]
        if not neighbors:
            row["status"] = "SEM_COBERTURA"
            continue
        row["best_similarity"] = round(1.0 - neighbors[0][0], 8)
        score = KNNExpert._weighted_vote(
            [(distance, label) for distance, label, _other in neighbors]
        )
        if not math.isfinite(score):
            row["status"] = "INVALIDO"
            row["error"] = "Voto KNN não finito"
            continue
        row["knn_score_ng"] = round(float(score), 8)
        row["prediction"] = "NG" if score > 0.5 else "OK"
        row["status"] = (
            "PASSOU" if row["prediction"] == row["expected_label"]
            else "REGRESSAO"
        )
        if geometry_experiment:
            # Ranking B independente. Nunca substitui o ranking/resultado A.
            geometry_selected.sort(key=lambda v: (v[0], v[2]["source_path"]))
            seen_geo = set()
            neighbors_geo = []
            for distance, label, other in geometry_selected:
                identity = other["source_pixel_sha256"] or other["source_file_sha256"]
                if identity and identity in seen_geo:
                    continue
                if identity:
                    seen_geo.add(identity)
                neighbors_geo.append((distance, label, other))
                if len(neighbors_geo) >= k:
                    break
            geo = {
                "status": "SEM_COBERTURA", "prediction": None,
                "knn_score_ng": None, "neighbors": [],
                "best_similarity": None,
                "shadow_candidate": {
                    "status": "REVISAO", "reason": "SEM_COBERTURA_GEOMETRICA",
                    "diagnostic_only": True, "auto_decision_authorized": False,
                },
            }
            if neighbors_geo:
                score_geo = KNNExpert._weighted_vote([
                    (dist, label) for dist, label, _other in neighbors_geo
                ])
                if math.isfinite(score_geo):
                    geo["knn_score_ng"] = round(float(score_geo), 8)
                    geo["prediction"] = "NG" if score_geo > 0.5 else "OK"
                    geo["status"] = (
                        "PASSOU" if geo["prediction"] == row["expected_label"]
                        else "REGRESSAO"
                    )
                    geo["best_similarity"] = round(1 - neighbors_geo[0][0], 8)
                    geo["neighbors"] = [
                        {"source_path": other["source_path"],
                         "source_pixel_sha256": other["source_pixel_sha256"],
                         "label": label, "similarity": round(1 - distance, 8)}
                        for distance, label, other in neighbors_geo
                    ]
            row["geometry_experiment"] = geo

    if canceled:
        # Não alegar que uma execução incompleta atingiu a meta.
        for row in rows:
            if row["status"] == "PRONTO_PARA_KNN":
                row["status"] = "NAO_ANALISADO"
                row["prediction"] = None

    # O candidato só usa vizinhos admissíveis da MESMA categoria/luz.
    # A cobertura de contraprova também exclui o próprio par, duplicatas
    # e eventos conhecidos, e nunca usa o rótulo da imagem consultada.
    for row in rows:
        if row["status"] in {"INVALIDO", "NAO_ANALISADO"}:
            row["shadow_candidate"] = {
                "status": "REVISAO", "reason": "IMAGEM_NAO_ELEGIVEL",
                "diagnostic_only": True, "auto_decision_authorized": False,
            }
            continue
        eligible = (
            other for other in pools[(
                canonical_memory_category(row["ocr_observed_category"]),
                row["lighting_mode"],
            )]
            if not _excluded(row, other, protocol)
        )
        support = {"OK": set(), "NG": set()}
        for other in eligible:
            ident = other["source_pixel_sha256"] or other["source_file_sha256"]
            if ident and other["expected_label"] in support:
                support[other["expected_label"]].add(ident)
        row["shadow_candidate"] = select_shadow_candidate(
            row["neighbors"], category=row["ocr_observed_category"],
            lighting=row["lighting_mode"],
            scope_examples={label: len(ids) for label, ids in support.items()},
        )
        if geometry_experiment:
            geo = row.get("geometry_experiment")
            if geo is None:
                geo = row["geometry_experiment"] = {
                    "status": "NAO_ANALISADO", "prediction": None, "neighbors": [],
                }
            if geo["status"] in {"NAO_ANALISADO", "INVALIDO", "SEM_COBERTURA"}:
                geo["shadow_candidate"] = {
                    "status": "REVISAO", "reason": "SEM_COBERTURA_GEOMETRICA",
                    "diagnostic_only": True, "auto_decision_authorized": False,
                }
            else:
                geo["shadow_candidate"] = select_shadow_candidate(
                    geo["neighbors"], category=row["ocr_observed_category"],
                    lighting=row["lighting_mode"],
                    scope_examples={label: len(ids) for label, ids in support.items()},
                )
            row["geometry_consensus_candidate"] = select_concordance_candidate(
                row["shadow_candidate"], geo["shadow_candidate"],
                visual_prediction=row["prediction"],
                geometry_prediction=geo.get("prediction"),
            )


    # Agrupamento conservador para DIAGNÓSTICO de cobertura: manifesto
    # autenticado quando existe, senão prefixo apenas como pista. Não usar
    # o agrupamento inferido como comprovação de peça/event_id real.
    groups = defaultdict(list)
    for row in rows:
        key = (
            ("manifest", row["event_id"])
            if row.get("event_id") else
            ("prefix_hint", row["capture_prefix_hint"])
        )
        groups[key].append(row)
    cohorts = {"side_only": [], "three_lights": [], "partial_other": []}
    verified_groups = 0
    inferred_groups = 0
    counts = {"side_only": 0, "three_lights": 0, "partial_other": 0}
    for (kind, _group), entries in groups.items():
        verified_groups += int(kind == "manifest")
        inferred_groups += int(kind == "prefix_hint")
        lights = {entry["lighting_mode"] for entry in entries}
        if lights == {"SIDE"}:
            cohort = "side_only"
        elif lights == {"SIDE", "TOP", "MID"}:
            cohort = "three_lights"
        else:
            cohort = "partial_other"
        cohorts[cohort].extend(entries)
        counts[cohort] += 1

    result = _measure(rows)
    geometry_report = None
    if geometry_experiment:
        # O mesmo conjunto de consultas e rótulos é utilizado no A/B.
        # Sem JPEG/PNG de teste externos, não demonstrar generalização.
        geo_rows = []
        for row in rows:
            geo = row.get("geometry_experiment") or {}
            geo_rows.append({
                "expected_label": row["expected_label"],
                "category_hint": row["category_hint"],
                "lighting_mode": row["lighting_mode"],
                "prediction": geo.get("prediction"),
                "status": geo.get("status", "NAO_ANALISADO"),
                "shadow_candidate": geo.get("shadow_candidate") or {
                    "status": "REVISAO",
                    "reason": "IMAGEM_NAO_ELEGIVEL",
                },
            })
        # C compara A e B no MESMO acervo: nunca usa o esperado na escolha.
        # Rótulo esperado é copiado SOMENTE para métricas após a decisão.
        consensus_rows = [
            {
                "expected_label": row["expected_label"],
                "category_hint": row["category_hint"],
                "lighting_mode": row["lighting_mode"],
                "shadow_candidate": row.get("geometry_consensus_candidate")
                or {
                    "status": "REVISAO", "reason": "SEM_DOIS_VOTOS_KNN",
                },
            }
            for row in rows
        ]
        geometry_metrics = _measure(geo_rows)
        if canceled:
            geometry_metrics["target_98pct_met"] = False
        geometry_report = {
            "experimental_only": True, "production_approved": False,
            "complete": not canceled,
            "name": "comparacao_geometrica_arestas_ausentes_e_adicionadas",
            "visual_control": "knn_archive_aoi_focus.v1",
            "geometry_weight": 0.35,
            "results": geometry_metrics,
            "consensus_candidate": summarize_shadow_candidates(
                consensus_rows, complete=not canceled,
            ),
            "consensus_source_counts": dict(Counter(
                (row.get("geometry_consensus_candidate") or {}).get(
                    "source") or "REVISAO"
                for row in rows
            )),
            "consensus_by_category": {
                name: summarize_shadow_candidates(
                    [v for v in consensus_rows
                     if canonical_memory_category(v["category_hint"]) == name],
                    complete=not canceled,
                )
                for name in sorted({
                    canonical_memory_category(v["category_hint"])
                    for v in consensus_rows
                })
            },
            "consensus_by_lighting": {
                light: summarize_shadow_candidates(
                    [v for v in consensus_rows if v["lighting_mode"] == light],
                    complete=not canceled,
                )
                for light in ("SIDE", "TOP", "MID")
            },
            "shadow_candidate": summarize_shadow_candidates(
                geo_rows, complete=not canceled,
            ),
            "by_category": {
                name: {
                    "results": _measure([
                        v for v in geo_rows
                        if canonical_memory_category(v["category_hint"]) == name
                    ]),
                    "shadow_candidate": summarize_shadow_candidates(
                        [v for v in geo_rows
                         if canonical_memory_category(v["category_hint"]) == name],
                        complete=not canceled,
                    ),
                }
                for name in sorted({
                    canonical_memory_category(v["category_hint"])
                    for v in geo_rows
                })
            },
            "by_lighting": {
                light: {
                    "results": _measure([
                        v for v in geo_rows if v["lighting_mode"] == light
                    ]),
                    "shadow_candidate": summarize_shadow_candidates(
                        [v for v in geo_rows if v["lighting_mode"] == light],
                        complete=not canceled,
                    ),
                }
                for light in ("SIDE", "TOP", "MID")
            },
            "geometry_failures": sum(
                bool(r.get("geometry_error")) for r in rows
            ),
            "case_count": len(geo_rows),
        }
    if canceled:
        result["target_98pct_met"] = False
    return {
        "complete": not canceled,
        "geometry_experiment": geometry_report,
        "shadow_candidate": summarize_shadow_candidates(rows, complete=not canceled),
        "shadow_candidate_by_lighting": {
            mode: summarize_shadow_candidates([r for r in rows
                  if r["lighting_mode"] == mode], complete=not canceled)
            for mode in ("SIDE", "TOP", "MID")
        },
        "shadow_candidate_by_category": {
            name: summarize_shadow_candidates(
                [r for r in rows
                 if canonical_memory_category(r["category_hint"]) == name],
                complete=not canceled,
            )
            for name in sorted({canonical_memory_category(r["category_hint"])
                                for r in rows})
        },
        "canceled": canceled,
        "by_capture_cohort": {
            cohort: {
                "capture_groups": counts[cohort],
                "results": _measure(entries),
                "shadow_candidate": summarize_shadow_candidates(entries, complete=not canceled),
            }
            for cohort, entries in cohorts.items()
        },
        "capture_grouping": {
            "manifest_verified_groups": verified_groups,
            "prefix_hint_groups_unverified": inferred_groups,
            "interpretation": (
                "Prefixos apenas estimam conjunto de iluminação; "
                "não comprovam mesma peça, event_id nem consenso."
            ),
        },
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": "ARCHIVE_ONLY_TEMPORARY_KNN_VISUAL",
        "protocol": protocol,
        "similarity_basis": "reference_vs_capture_global_outer_inner_epicenter",
        "date_used_for_neighbor_selection": False,
        "k": k,
        "expected_retention_rate_by_label": 0.98,
        "candidate_target_conditional_precision": 0.97,
        "inventory_png_count": len(items),
        "memory_source": ["public/ok_archive", "public/ng_archive"],
        "uses_legacy_memory_json": False,
        "uses_cnn": False,
        "writes_memory_or_dataset": False,
        "is_production_verified_exact_knn_test": False,
        "is_production_approved": False,
        "shadow_candidate_uses_only_archive_not_runtime_memory": True,
        "results": result,
        "focus_detection": {
            "total": len(rows),
            "outer_detected": sum(bool((r.get("focus") or {}).get(
                "outer_detected")) for r in rows),
            "inner_detected": sum(bool((r.get("focus") or {}).get(
                "inner_detected")) for r in rows),
            "global_only": sum(
                r["status"] not in {"INVALIDO", "NAO_ANALISADO"}
                and not (r.get("focus") or {}).get("outer_detected")
                for r in rows
            ),
        },
        "by_category": {
            c: _measure([r for r in rows if canonical_memory_category(
                r["category_hint"]) == c])
            for c in sorted({canonical_memory_category(
                r["category_hint"]) for r in rows})
        },
        "by_lighting": {
            mode: _measure([r for r in rows if r["lighting_mode"] == mode])
            for mode in sorted({r["lighting_mode"] for r in rows})
        },
        "cases": rows,
        "limitations": (
            "Avaliação apenas dos PNGs do arquivo. Cada consulta usa gabarito "
            "e captura reais, com regiões global, caixa maior e epicentro menor "
            "quando os contornos verdes aninhados estão presentes. Não há "
            "qualquer filtro por data. Sem epicentro menor comprovado, o "
            "relatório informa ausência e compara apenas regiões existentes. "
            "A consulta nunca usa seu próprio rótulo; vizinhos usam rótulos "
            "humanos somente como referências. O mesmo arquivo, mesmo hash "
            "e mesmo evento/prefixo são excluídos, mas outras imagens muito "
            "semelhantes do mesmo lote ainda podem ser correlacionadas. "
            "Reconhecimento de exemplos desse acervo não demonstra "
            "generalização independente a novos componentes. Sem PNGs "
            "locais não é possível medir novamente os resultados. "
            "Os grupos multilight baseados só em prefixo NÃO são uma prova "
            "de mesmo event_id. Não comanda XP nem escreve na memória."
        ),
    }


__all__ = ["replay_knn_archive_visual", "SCHEMA", "VALID_PROTOCOLS"]
