"""Comparador A/B de roteamento CNN + KNN visual; relatório offline não decisório.

Executar os MESMOS PNGs com CNN e KNN-visual sem auto-memorizar a consulta,
cruzando arquivo e hash original. As rotas NÃO são produção, nem demonstram
desempenho por peça, event_id ou generalização a lotes independentes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import time

from src.core.strict_category_memory import canonical_memory_category
from src.services.startup_regression.archive_inventory_report import _atomic_text

SCHEMA = "visionx.knn_cnn_arbitration_shadow.v1"
CNN_SCHEMA = "visionx.interactive_archive_validation.v1"
KNN_SCHEMA = "visionx.knn_archive_visual_replay.v1"
LABELS = ("OK", "NG")
CANDIDATE = ("SUGERE_OK", "SUGERE_NG")


def _join(cnn_report: dict, knn_report: dict) -> tuple[list[dict], dict]:
    if (cnn_report.get("schema") != CNN_SCHEMA
            or knn_report.get("schema") != KNN_SCHEMA
            or cnn_report.get("complete") is not True
            or knn_report.get("complete") is not True
            or cnn_report.get("canceled") is not False
            or knn_report.get("canceled") is not False):
        raise ValueError("Exige relatórios completos, não cancelados, dos dois modelos")
    if (Path(cnn_report.get("root", "")).resolve()
            != Path(knn_report.get("root", "")).resolve()):
        raise ValueError("Relatórios de instalações diferentes")
    if (cnn_report.get("archive_png_count") != knn_report.get("inventory_png_count")
            or cnn_report.get("archive_png_count", 0) <= 0):
        raise ValueError("Inventários com quantidades divergentes")
    if (knn_report.get("protocol") != "visual_holdout"
            or knn_report.get("uses_cnn") is not False
            or knn_report.get("is_production_verified_exact_knn_test") is not False):
        raise ValueError("Replay KNN não independente por exclusão de consulta")
    originals = {}
    for case in cnn_report.get("cases", ()):
        if case.get("model") != "CNN_FALTANDO_V2":
            continue
        key = case.get("source_path")
        if not key or key in originals:
            raise ValueError("Caminho CNN ausente ou duplicado")
        originals[key] = case
    if len(originals) != cnn_report["archive_png_count"]:
        raise ValueError("Validação CNN sem todos os registros do inventário")
    visuals = {}
    for case in knn_report.get("cases", ()):
        key = case.get("source_path")
        if not key or key in visuals:
            raise ValueError("Caminho KNN ausente ou duplicado")
        visuals[key] = case
    if len(visuals) != knn_report["inventory_png_count"]:
        raise ValueError("Replay KNN sem todos os registros do inventário")
    if set(originals) != set(visuals):
        raise ValueError("Arquivos dos relatórios não coincidem")
    out = []
    excluded_invalid = []
    for key in sorted(originals):
        c, k = originals[key], visuals[key]
        if (c.get("expected_label") not in LABELS
                or k.get("expected_label") != c["expected_label"]
                or c.get("lighting_mode") != k.get("lighting_mode")
                or canonical_memory_category(c.get("category"))
                   != canonical_memory_category(k.get("category_hint"))):
            raise ValueError(f"Rótulo/categoria/luz divergente: {key}")
        if (c.get("status") == "INVALIDO" and k.get("status") == "INVALIDO"
                and not c.get("source_file_sha256")
                and not k.get("source_file_sha256")):
            # O inventário pode conter um PNG ilegível: não inventar hash.
            # A ocorrência permanece visível nas métricas de cobertura.
            excluded_invalid.append(key)
            continue
        for field in ("source_file_sha256", "source_pixel_sha256"):
            a, b = c.get(field), k.get(field)
            if (not isinstance(a, str) or len(a) != 64
                    or not isinstance(b, str) or len(b) != 64 or a != b):
                raise ValueError(
                    "Hashes ausentes ou diferentes; rode novamente ambos os "
                    f"diagnósticos sobre o mesmo inventário: {key}"
                )
        if c.get("status") == "NAO_ANALISADO" or k.get("status") == "NAO_ANALISADO":
            raise ValueError("Há consultas não avaliadas; resultado incompleto")
        suggested = k.get("shadow_candidate") or {}
        if suggested.get("auto_decision_authorized") is True:
            raise ValueError("Candidato KNN contém autorização de produção indevida")
        status = suggested.get("status")
        if status not in CANDIDATE:
            status = "REVISAO"
        # O KNN tem de excluir a própria consulta e duplicatas; a estrutura
        # da origem garante isso, mas checamos também o vizinho informado.
        for neighbor in k.get("neighbors", ()):
            if (neighbor.get("source_path") == key
                    or neighbor.get("source_pixel_sha256") == c["source_pixel_sha256"]):
                raise ValueError(f"Vazamento do próprio PNG nos vizinhos: {key}")
        cnn_label = (
            c.get("predicted_label")
            if c.get("status") in {"PASSOU", "ERRO_CLASSIFICACAO"}
            and c.get("predicted_label") in LABELS
            else None
        )
        suggestion = status.removeprefix("SUGERE_") if status in CANDIDATE else None
        # Se a CNN não votou, KNN aproximada NÃO cria voto conclusivo.
        # Se discordarem, o candidato seguro é revisão (nunca substitui
        # NG por OK ou OK por NG sem prova física).
        guarded = (
            cnn_label if suggestion is None or suggestion == cnn_label
            else None
        )
        optimistic = (
            suggestion if cnn_label and suggestion else cnn_label
        )
        out.append({
            "source_path": key,
            "source_file_sha256": c["source_file_sha256"],
            "expected_label": c["expected_label"],
            "category": c.get("category"), "lighting": c.get("lighting_mode"),
            "cnn": {
                "status": c.get("status"), "label": cnn_label,
                "ng_score_uncalibrated": c.get("cnn_score_ng"),
                "checkpoint_sha256": c.get("checkpoint_sha256"),
            },
            "knn_visual": {
                "status": status, "suggested_label": suggestion,
                "visual_prediction": k.get("prediction"),
                "best_similarity": k.get("best_similarity"),
                "neighbor_count": k.get("neighbors_count"),
                "neighbors_ok": k.get("neighbors_ok"),
                "neighbors_ng": k.get("neighbors_ng"),
                "neighbors": [
                    {
                        "source_path": n.get("source_path"),
                        "label": n.get("label"),
                        "similarity": n.get("similarity"),
                    }
                    for n in (k.get("neighbors") or [])[:5]
                    if isinstance(n, dict)
                ],
                "diagnostic_only": True,
            },
            "guarded_label": guarded,
            "optimistic_label": optimistic,
            "conflict": bool(cnn_label and suggestion and cnn_label != suggestion),
            "rescue_of_false_ng_for_review": (
                c["expected_label"] == "OK" and cnn_label == "NG"
                and suggestion == "OK"
            ),
            "ng_blocked_from_auto_ok": (
                c["expected_label"] == "NG" and cnn_label == "OK"
                and suggestion == "NG"
            ),
        })
    return out, {
        "archives_same_files": True,
        "invalid_pngs_excluded_by_path": excluded_invalid,
        "original_png_hashes_verified_between_reports": True,
        "knn_own_capture_excluded": True,
        "independent_event_holdout": False,
        "candidate_diagnostic_only": True,
    }


def _metrics(rows: list[dict], key: str) -> dict:
    out = {"total": len(rows), "classified": 0, "correct": 0,
           "ok_as_ng": 0, "ng_as_ok": 0, "review": 0,
           "by_label": {}}
    for expected in LABELS:
        bucket = [r for r in rows if r["expected_label"] == expected]
        correct = sum(r.get(key) == expected for r in bucket)
        wrong = sum(r.get(key) in LABELS and r.get(key) != expected for r in bucket)
        review = len(bucket) - correct - wrong
        out["by_label"][expected] = {
            "total": len(bucket), "correct": correct,
            "incorrect": wrong, "review": review,
        }
    out["classified"] = sum(r.get(key) in LABELS for r in rows)
    out["correct"] = sum(r.get(key) == r["expected_label"] for r in rows)
    out["ok_as_ng"] = sum(r["expected_label"] == "OK"
                         and r.get(key) == "NG" for r in rows)
    out["ng_as_ok"] = sum(r["expected_label"] == "NG"
                         and r.get(key) == "OK" for r in rows)
    out["review"] = len(rows) - out["classified"]
    return out


def compare(cnn_report: dict, knn_report: dict) -> dict:
    started = time.monotonic()
    rows, evidence = _join(cnn_report, knn_report)
    eligible = [r for r in rows if r["cnn"]["status"] != "FORA_ESCOPO"]
    # "Sem escopo CNN" NÃO é acerto e não entra no denominador CNN.
    for r in eligible:
        r["cnn_baseline_label"] = r["cnn"]["label"]
    def category_group(key):
        groups = defaultdict(list)
        for r in eligible:
            groups[str(r[key])].append(r)
        return {name: {
            "cnn": _metrics(
                [{**item, "baseline": item["cnn"]["label"]} for item in group],
                "baseline"
            ),
            "guarded": _metrics(group, "guarded_label"),
            "optimistic_hypothetical": _metrics(group, "optimistic_label"),
        } for name, group in sorted(groups.items())}
    results = {
        "cnn_isolated_baseline": _metrics(eligible, "cnn_baseline_label"),
        "guarded_conflict_to_review": _metrics(eligible, "guarded_label"),
        "optimistic_knn_override_hypothetical": _metrics(eligible, "optimistic_label"),
    }
    # A métrica do baseline foi calculada ANTES de remover o campo interno.
    for r in eligible:
        r.pop("cnn_baseline_label", None)
    conflicts = [r for r in eligible if r["conflict"]]
    # Drill-down: locais dos OK que a CNN considerou NG, sem converter
    # sugestão aproximada em autorização operacional.
    false_ng = [
        {
            "source_path": r["source_path"],
            "category": r["category"],
            "lighting": r["lighting"],
            "cnn_ng_score_uncalibrated": r["cnn"]["ng_score_uncalibrated"],
            "knn_suggested_label": r["knn_visual"]["suggested_label"],
            "knn_visual_prediction": r["knn_visual"]["visual_prediction"],
            "best_similarity": r["knn_visual"]["best_similarity"],
            "neighbors": r["knn_visual"]["neighbors"],
            "possible_review_rescue": r["rescue_of_false_ng_for_review"],
        }
        for r in eligible
        if r["expected_label"] == "OK" and r["cnn"]["label"] == "NG"
    ]
    groups = defaultdict(list)
    for case in false_ng:
        # Agrupamento exploratório por vizinho principal, NUNCA por event_id.
        first = case["neighbors"][0]["source_path"] if case["neighbors"] else ""
        groups[first or "SEM_VIZINHOS"].append(case["source_path"])
    return {
        "schema": SCHEMA, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": cnn_report["root"],
        "mode": "OFFLINE_PNG_KNN_CNN_AB_ONLY",
        "complete": True, "production_approved": False,
        "xp_command_authorized": False,
        "weights_changed": False, "memory_changed": False,
        "archive_pngs": cnn_report["archive_png_count"],
        "valid_crosschecked_pngs": len(rows),
        "invalid_pngs": len(evidence["invalid_pngs_excluded_by_path"]),
        "cnn_eligible_pngs": len(eligible),
        "cnn_out_of_scope_pngs": len(rows)-len(eligible),
        "evidence_contract": evidence,
        "baseline_isolated_cnn_only": True,
        "production_route_reproduced": False,
        "results": results,
        "conflicts": {
            "total": len(conflicts),
            "cnn_ng_knn_suggests_ok": sum(
                r["cnn"]["label"]=="NG" and r["knn_visual"]["suggested_label"]=="OK"
                for r in conflicts
            ),
            "cnn_ok_knn_suggests_ng": sum(
                r["cnn"]["label"]=="OK" and r["knn_visual"]["suggested_label"]=="NG"
                for r in conflicts
            ),
            "false_ng_possible_review_rescue": sum(
                r["rescue_of_false_ng_for_review"] for r in eligible
            ),
            "false_ok_blocked_from_auto_release": sum(
                r["ng_blocked_from_auto_ok"] for r in eligible
            ),
        },
        "false_ng_investigation": {
            "scope": "PNG_HISTORICO_IN_SAMPLE_RISCO_NG_OK",
            "count": len(false_ng),
            "possible_review_rescues": sum(
                c["possible_review_rescue"] for c in false_ng
            ),
            "grouping_notice": (
                "Agrupamento apenas por vizinho KNN mais próximo; "
                "não identifica peças ou eventos físicos."
            ),
            "nearest_neighbor_groups": {
                key: sorted(values) for key, values in sorted(groups.items())
            },
            "cases": false_ng,
            "auto_ok_authorized": False,
        },
        "by_category": category_group("category"),
        "by_lighting": category_group("lighting"),
        "checkpoint_sha256_values": sorted({
            r["cnn"]["checkpoint_sha256"] for r in eligible
            if r["cnn"]["checkpoint_sha256"]
        }),
        "duration_seconds": round(time.monotonic()-started,3),
        "rows": rows,
        "cautions": [
            "PNG, não peça/evento. Resultados não reproduzem fusão nem política de Produção.",
            "KNN visual temporária leave-one-out, não o atalho KNN exato em Produção.",
            "Embora exclua arquivo, hash e prefixo, exemplos correlacionados do mesmo lote podem vazar.",
            "Score da CNN e similaridade KNN não são probabilidades calibradas.",
            "Rótulo da pasta pode não ser verdade humana verificada.",
            "Conflito da estratégia guarded vira REVISÃO, não OK/NG automaticamente.",
            "Cenário optimistic é contrafactual de risco; NUNCA controla XP.",
            "Não aprova checkpoint, limiares, memórias, roteamento de Produção ou 0/1.",
        ],
    }


def save_report(report: dict, root: Path) -> tuple[Path, Path]:
    root = Path(root).resolve()
    dest = root / "reports" / "startup_regression"
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    jp = dest / f"knn_cnn_shadow_{stamp}.json"
    tp = dest / f"knn_cnn_shadow_{stamp}.txt"
    _atomic_text(jp, json.dumps(report, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n")
    metrics = report["results"]
    t = ["ODIN | KNN + CNN — ARBITRAGEM EXPERIMENTAL OFFLINE",
         "Somente PNGs com hashes cruzados; nenhuma autorização XP.",
         f"Arquivo: {report['archive_pngs']} PNGs; CNN elegíveis: {report['cnn_eligible_pngs']}",
         f"Conflitos: {report['conflicts']['total']}", ""]
    for name, m in metrics.items():
        t.extend([name,
                  f"Corretos: {m['correct']}/{m['total']}",
                  f"NG->OK: {m['ng_as_ok']} | OK->NG: {m['ok_as_ng']}",
                  f"Revisões: {m['review']} | Classificados: {m['classified']}", ""])
    t.extend(["POSSÍVEIS MELHORIAS SEM APROVAR NG AUTOMATICAMENTE:",
              f"Falsos NG enviados a revisão: {report['conflicts']['false_ng_possible_review_rescue']}",
              f"Falsos OK bloqueados: {report['conflicts']['false_ok_blocked_from_auto_release']}",
              "", "FALSOS NG HISTÓRICOS POR PNG (CNN NG, ARQUIVO OK):",
              f"Total: {report['false_ng_investigation']['count']}; "
              f"enviáveis a revisão: {report['false_ng_investigation']['possible_review_rescues']}",
              "O vizinho não valida a verdade física. Nenhum OK automático.",
              *[
                  f"{case['source_path']} | CNN_NG={case['cnn_ng_score_uncalibrated']} "
                  f"| KNN={case['knn_suggested_label']} "
                  f"| próximo={case['neighbors'][0]['source_path'] if case['neighbors'] else '-'}"
                  for case in report["false_ng_investigation"]["cases"]
              ],
              "", "LIMITAÇÕES:", *report["cautions"]])
    _atomic_text(tp, "\n".join(t)+"\n")
    return jp,tp


__all__ = ["SCHEMA","compare","save_report"]
