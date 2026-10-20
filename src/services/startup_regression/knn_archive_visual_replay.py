"""Replay KNN visual EXCLUSIVAMENTE em public/ok_archive e ng_archive.

Protocolo padrão: leave-one-DAY-out. Cada PNG é consultado, sem usar seu
rótulo na predição; vizinhos vêm somente de OUTROS DIAS do mesmo arquivo
visual, categoria e iluminação. O recorte AOI é extraído do PNG real.
A assinatura e a comparação usam src.core.anomaly_signature, e o voto
ponderado usa KNNExpert._weighted_vote (k=5, congelado).

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

from src.core.anomaly_signature import (
    build_anomaly_signature, compare_anomaly_signatures, valid_anomaly_signature,
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
VALID_PROTOCOLS = ("leave_day_out", "leave_one_out")
DATE_PATTERN = re.compile(r"(?<!\d)(20\d{2}-[01]\d-[0-3]\d)(?!\d)")
TRAILING_COPY = re.compile(r"_(?:[2-9]|[1-9]\d+)$", re.IGNORECASE)
TRAILING_LIGHT = re.compile(r"_(SIDE|TOP|MID)$", re.IGNORECASE)


def _capture_group(path: str) -> str:
    """Conservative prefix grouping; NOT proof of a multilight event."""
    stem = Path(path).stem.upper()
    stem = TRAILING_COPY.sub("", stem)
    return TRAILING_LIGHT.sub("", stem)


def _day(path: str) -> str | None:
    hit = DATE_PATTERN.search(Path(path).stem)
    return hit.group(1) if hit else None


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
        "session_day_hint": _day(path),
        "capture_prefix_hint": _capture_group(path),
        "ocr_observed_category": None,
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


def _excluded(query: dict, candidate: dict, protocol: str) -> bool:
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
    if (protocol == "leave_day_out" and query["session_day_hint"]
            and query["session_day_hint"] == candidate["session_day_hint"]):
        return True
    return False


def replay_knn_archive_visual(
    root: Path, *, k: int = 5, protocol: str = "leave_day_out",
    inventory: dict | None = None,
    extractor: Callable | None = None,
    frame_loader: Callable | None = None,
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
            if protocol == "leave_day_out" and not row["session_day_hint"]:
                raise ValueError("Data ausente no nome: sem isolamento por dia")
            frame = frame_loader(root, item)
            reference, test, info = extractor(frame)
            if not isinstance(info, dict):
                raise ValueError("Extrator AOI não informou metadados")
            category, normalized_value = resolve_archive_aoi_category(
                info, row["category_hint"],
            )
            row["ocr_observed_category"] = category
            signature = build_anomaly_signature(
                reference, test, {}, {
                    **info, "category": category, "value": normalized_value,
                },
            )
            if not valid_anomaly_signature(signature):
                raise ValueError("Assinatura KNN inválida")
            row["signature_map_peak"] = signature["summary"]["map_peak"]
            signatures[row["source_path"]] = signature
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
        for other in pools[(
            canonical_memory_category(row["ocr_observed_category"]),
            row["lighting_mode"],
        )]:
            if _excluded(row, other, protocol):
                continue
            similarity, _breakdown = compare_anomaly_signatures(
                signatures[row["source_path"]], signatures[other["source_path"]]
            )
            if not math.isfinite(similarity):
                row["status"] = "INVALIDO"
                row["error"] = "Similaridade não finita"
                break
            selected.append((1.0 - similarity, other["expected_label"], other))
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

    if canceled:
        # Não alegar que uma execução incompleta atingiu a meta.
        for row in rows:
            if row["status"] == "PRONTO_PARA_KNN":
                row["status"] = "NAO_ANALISADO"
                row["prediction"] = None

    # O candidato de shadow não altera o voto baseline nem o rótulo do acervo.
    # Um NG/OK sem exemplares de OUTROS DIAS na mesma categoria/luz
    # provoca abstenção, nunca sugestão OK por ausência de exemplos NG.
    support_days = defaultdict(lambda: {"OK": set(), "NG": set()})
    for row in rows:
        if row["status"] not in {"INVALIDO", "NAO_ANALISADO"}:
            support_days[(canonical_memory_category(row["ocr_observed_category"]),
                          row["lighting_mode"])][row["expected_label"]].add(
                              row["session_day_hint"]
                          )
    for row in rows:
        if row["status"] in {"INVALIDO", "NAO_ANALISADO"}:
            row["shadow_candidate"] = {
                "status": "REVISAO", "reason": "IMAGEM_NAO_ELEGIVEL",
                "diagnostic_only": True, "auto_decision_authorized": False,
            }
            continue
        scope = support_days[(canonical_memory_category(row["ocr_observed_category"]),
                              row["lighting_mode"])]
        # Também retira o dia da própria consulta do denominador de cobertura.
        days = {
            label: len(values - {row["session_day_hint"]})
            for label, values in scope.items()
        }
        row["shadow_candidate"] = select_shadow_candidate(
            row["neighbors"], category=row["ocr_observed_category"],
            lighting=row["lighting_mode"], scope_days=days,
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
    if canceled:
        result["target_98pct_met"] = False
    return {
        "complete": not canceled,
        "shadow_candidate": summarize_shadow_candidates(rows),
        "shadow_candidate_by_lighting": {
            mode: summarize_shadow_candidates([r for r in rows
                  if r["lighting_mode"] == mode])
            for mode in ("SIDE", "TOP", "MID")
        },
        "shadow_candidate_by_category": {
            name: summarize_shadow_candidates([r for r in rows
                  if canonical_memory_category(r["category_hint"]) == name])
            for name in sorted({canonical_memory_category(r["category_hint"])
                                for r in rows})
        },
        "canceled": canceled,
        "by_capture_cohort": {
            cohort: {
                "capture_groups": counts[cohort],
                "results": _measure(entries),
                "shadow_candidate": summarize_shadow_candidates(entries),
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
        "k": k,
        "expected_retention_rate_by_label": 0.98,
        "inventory_png_count": len(items),
        "memory_source": ["public/ok_archive", "public/ng_archive"],
        "uses_legacy_memory_json": False,
        "uses_cnn": False,
        "writes_memory_or_dataset": False,
        "is_production_verified_exact_knn_test": False,
        "is_production_approved": False,
        "shadow_candidate_uses_only_archive_not_runtime_memory": True,
        "results": result,
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
            "Avaliação da generalização da KNN com assinaturas da divergência "
            "extraídas exclusivamente dos PNGs do arquivo: não equivale à "
            "memória KNN verificada de produção, que exige par AOI humano "
            "comprovado. O rótulo da pasta só é usado nos vizinhos e na "
            "avaliação pós-predição; nunca na consulta. Na opção "
            "leave_one_out pode haver correlação temporal; o protocolo "
            "padrão leave_day_out remove do treino todas as imagens do "
            "mesmo dia. Sem PNGs locais não é possível medir resultados. "
            "Um acervo fechado com >98% não prova generalização futura. "
            "Subgrupos SIDE apenas e 3 luzes são contagens por manifesto "
            "verificado ou prefixo como pista, nunca confirmação automática "
            "de event_id. Ausência de independência física entre placas ou "
            "lotes impede inferir taxa real de futuras peças."
        ),
    }


__all__ = ["replay_knn_archive_visual", "SCHEMA", "VALID_PROTOCOLS"]
