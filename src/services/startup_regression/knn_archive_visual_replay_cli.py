"""CLI: KNN visual apenas com os PNGs arquivados.

    python -m src.services.startup_regression.knn_archive_visual_replay_cli

Não consulta JSONs de memória nem CNN, e não altera produção, memória ou
os arquivos de imagens. Salva JSON/TXT em reports/startup_regression.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .archive_inventory_report import _atomic_text
from .knn_archive_visual_replay import replay_knn_archive_visual, VALID_PROTOCOLS


def write_knn_archive_reports(report: dict, output_dir: Path) -> tuple[Path, Path]:
    root = Path(report["root"]).resolve()
    dest = Path(output_dir).resolve()
    reports = (root / "reports").resolve()
    if reports not in dest.parents:
        raise ValueError("Saída de relatórios deve estar em subpasta de reports/")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = dest / f"knn_archive_visual_{stamp}.json"
    txt_path = dest / f"knn_archive_visual_{stamp}.txt"
    _atomic_text(json_path, json.dumps(
        report, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n")

    metrics = report["results"]
    lines = [
        "ODIN — REPLAY VISUAL DA MEMÓRIA KNN (SOMENTE PNG OK/NG)",
        "ARQUIVO-ONLY | SEM CNN | SEM JSON ANTIGO | SEM MODIFICAR PRODUÇÃO",
        "KNN experimental por vizinhos de assinaturas visuais arquivadas.",
        "NÃO é o roteador KNN verificado por par exato da produção.",
        f"Protocolo: {report['protocol']} | k={report['k']}",
        "Critério: visual gabarito vs. captura, macro AOI e epicentro menor",
        "Filtro por datas: NENHUM",
        f"Inventário total: {report['inventory_png_count']}",
        f"Acertos: {metrics['correct']}/{metrics['total']}",
        f"Meta KNN >=98% em OK E em NG: {metrics['target_98pct_met']}",
        f"Status: {metrics['status_counts']}",
        f"Completo: {report.get('complete', True)} | Cancelado: {report.get('canceled', False)}",
        "As mesmas datas são aceitas; mesmo arquivo/duplicata/evento não.",
        "",
        "CLASSES SEPARADAS:",
    ]
    for label in ("OK", "NG"):
        stats = metrics["by_label"][label]
        lines.append(
            f"{label}: {stats['correct']}/{stats['total']} "
            f"({100 * stats['rate']:.2f}%) | "
            f"como OK={stats['as_OK']} | como NG={stats['as_NG']} "
            f"| sem decisão={stats['no_decision']}"
        )
    lines.extend([
        f"NG perigosamente como OK: {metrics['NG_as_OK']}",
        f"OK incorretamente como NG: {metrics['OK_as_NG']}",
        "",
        "POR CATEGORIA:",
    ])
    for category, stats in report["by_category"].items():
        lines.append(
            f"{category}: OK={stats['by_label']['OK']['correct']}/"
            f"{stats['by_label']['OK']['total']}; "
            f"NG={stats['by_label']['NG']['correct']}/"
            f"{stats['by_label']['NG']['total']}; "
            f"status={stats['status_counts']}"
        )
    cohort = report.get("by_capture_cohort") or {}
    if cohort:
        lines.extend(["", "COBERTURA DE CAPTURAS SIDE / TRÊS LUZES:"])
        for key, title in (
            ("side_only", "SOMENTE SIDE"),
            ("three_lights", "SIDE+TOP+MID"),
            ("partial_other", "PARCIAL/OUTROS"),
        ):
            data = cohort.get(key) or {}
            stats = data.get("results") or {}
            by = stats.get("by_label") or {}
            ok = by.get("OK") or {}
            ng = by.get("NG") or {}
            lines.append(
                f"{title}: grupos={data.get('capture_groups', 0)} "
                f"OK={ok.get('correct', 0)}/{ok.get('total', 0)} "
                f"NG={ng.get('correct', 0)}/{ng.get('total', 0)} "
                f"NG->OK={stats.get('NG_as_OK', 0)}"
            )
        grouping = report.get("capture_grouping") or {}
        lines.append(
            "Manifestos verificados: "
            + str(grouping.get("manifest_verified_groups", 0))
            + "; prefixos não verificados: "
            + str(grouping.get("prefix_hint_groups_unverified", 0))
        )
        lines.append(
            "Sem manifesto, grupos são estimados por nome; "
            "não comprovaram vínculo físico de evento."
        )
    shadow = report.get("shadow_candidate") or {}
    if shadow:
        ok_c = (shadow.get("by_label") or {}).get("OK") or {}
        ng_c = (shadow.get("by_label") or {}).get("NG") or {}
        lines.extend([
            "",
            "CANDIDATO KNN POR EPICENTRO (NAO CONTROLA XP):",
            "Meta >=97% de PRECISAO ENTRE SUGESTOES (nao de todo o acervo).",
            "Cobertura conta TODAS as imagens, inclusive as enviadas a revisao.",
            "Similaridade experimental >=0.90; cinco vizinhos distintos. "
            "OK exige cinco OK e contraprova NG; NG exige >=3 NG e voto "
            "ponderado NG >0.5, sem divergencia entre voto e maioria.",
            f"Precisao condicional: {100*shadow.get('conditional_precision', 0):.2f}% "
            f"({shadow.get('correct_suggestions', 0)}/{shadow.get('suggestions', 0)} sugestoes)",
            f"Cobertura: {100*shadow.get('coverage', 0):.2f}% "
            f"({shadow.get('suggestions', 0)}/{shadow.get('total', 0)} imagens)",
            f"Meta 97% + zero NG->OK observados: {shadow.get('target_97pct_met', False)}",
            f"OK sugerido corretamente: {ok_c.get('SUGERE_OK', 0)}/{ok_c.get('total', 0)}",
            f"NG sugerido corretamente: {ng_c.get('SUGERE_NG', 0)}/{ng_c.get('total', 0)}",
            f"NG sugeridos como OK (perigoso): {shadow.get('NG_suggested_OK', 0)}",
            f"OK sugeridos como NG: {shadow.get('OK_suggested_NG', 0)}",
            f"REVISAO/SEM COBERTURA: {shadow.get('reviews', 0)}",
            f"Erros entre sugestoes: {shadow.get('incorrect_suggestions', 0)}",
            "AVISO: comparação visual independente do dia, mas imagens do "
            "mesmo lote ainda podem ter correlação; não aprovado para Produção.",
        ])
        for category, bucket in (report.get("shadow_candidate_by_category") or {}).items():
            lines.append(
                f"  {category}: OK certo="
                f"{bucket['by_label']['OK']['SUGERE_OK']} "
                f"NG certo={bucket['by_label']['NG']['SUGERE_NG']} "
                f"NG->OK={bucket['NG_suggested_OK']} revisoes={bucket['reviews']}"
            )
    geo = report.get("geometry_experiment")
    if geo:
        gm, gc = geo["results"], geo["shadow_candidate"]
        lines.extend([
            "",
            "COMPARAÇÃO A/B DE GEOMETRIA (EXPERIMENTAL; SEM PROMOÇÃO):",
            "A = KNN v3 original; B = bordas desaparecidas/adicionadas "
            "nas regiões AOI (peso geométrico de 35%).",
            f"A acertos: {metrics['correct']}/{metrics['total']} | "
            f"NG->OK {metrics['NG_as_OK']}",
            f"B acertos: {gm['correct']}/{gm['total']} | "
            f"NG->OK {gm['NG_as_OK']}",
            f"A conservadora: {shadow.get('correct_suggestions', 0)}/"
            f"{shadow.get('suggestions', 0)} | cobertura "
            f"{100 * shadow.get('coverage', 0):.2f}%",
            f"B conservadora: {gc['correct_suggestions']}/"
            f"{gc['suggestions']} | cobertura "
            f"{100 * gc['coverage']:.2f}% | "
            f"NG->OK {gc['NG_suggested_OK']} | "
            f"meta 97% {gc['target_97pct_met']}",
            f"Falhas na extração geométrica: {geo['geometry_failures']}",
            "Métricas do mesmo acervo; NÃO comprova generalização.",
        ])
        for category, entry in geo["by_category"].items():
            summary = entry["shadow_candidate"]
            lines.append(
                f"  B {category}: sugestões={summary['suggestions']} "
                f"acertos={summary['correct_suggestions']} "
                f"revisões={summary['reviews']} "
                f"NG->OK={summary['NG_suggested_OK']}"
            )
    focus = report.get("focus_detection") or {}
    lines.extend([
        "", "DETECÇÃO DAS REGIÕES AOI:",
        f"Caixa maior detectada: {focus.get('outer_detected', 0)}",
        f"Epicentro menor detectado: {focus.get('inner_detected', 0)}",
        f"Somente gabarito/teste global: {focus.get('global_only', 0)}",
        "Molduras verdes removidas das máscaras de diferença visual.",
    ])
    lines.extend(["", "POR ILUMINAÇÃO:"])
    for mode, stats in report["by_lighting"].items():
        lines.append(
            f"{mode}: OK={stats['by_label']['OK']['correct']}/"
            f"{stats['by_label']['OK']['total']}; "
            f"NG={stats['by_label']['NG']['correct']}/"
            f"{stats['by_label']['NG']['total']}; "
            f"status={stats['status_counts']}"
        )
    lines.extend(["", "CASOS QUE NÃO PASSARAM:"])
    for case in report["cases"]:
        if case["status"] == "PASSOU":
            continue
        lines.append(
            f"- {case['status']} | {case['source_path']} | "
            f"esperado={case['expected_label']} | obtido="
            f"{case['prediction'] or 'SEM_DECISAO'} | "
            f"votoNG={case['knn_score_ng']} | "
            f"sim={case['best_similarity']} | "
            f"vizinhosOK={case['neighbors_ok']} NG={case['neighbors_ng']} | "
            f"candidato={case.get('shadow_candidate', {}).get('status', 'N/D')} | "
            f"motivo={case.get('shadow_candidate', {}).get('reason', 'N/D')} | "
            f"{case['error'] or ''}"
        )
    lines.extend([
        "",
        "LIMITAÇÕES:",
        report["limitations"],
        "Sem treinamento, sem importação, sem escrita no dataset, sem gate.",
    ])
    _atomic_text(txt_path, "\n".join(lines) + "\n")
    return json_path, txt_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Avaliação KNN só com as imagens OK/NG, >=98% por classe"
    )
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--geometry-experiment", action="store_true",
        help="A/B: descritor visual v3 versus geometria do gabarito e captura.",
    )
    parser.add_argument(
        "--protocol", choices=VALID_PROTOCOLS, default="visual_holdout",
        help="visual_holdout: compara imagens do acervo pelo aspecto visual",
    )
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    print(
        "KNN arquivo visual: extração AOI e comparação KNN, sem CNN/JSONs...",
        flush=True,
    )

    def progress(n, total, path):
        if n == 1 or n % 25 == 0 or n == total:
            print(f"[{n}/{total}] {path}", flush=True)

    report = replay_knn_archive_visual(
        root, k=args.k, protocol=args.protocol,
        geometry_experiment=args.geometry_experiment, progress=progress
    )
    j, t = write_knn_archive_reports(
        report, root / "reports" / "startup_regression"
    )
    stats = report["results"]
    ok, ng = (stats["by_label"][label] for label in ("OK", "NG"))
    print(
        f"KNN OK: {ok['correct']}/{ok['total']} ({ok['rate']:.2%}) | "
        f"NG: {ng['correct']}/{ng['total']} ({ng['rate']:.2%}) | "
        f"NG->OK: {stats['NG_as_OK']}",
        flush=True,
    )
    candidate = report["shadow_candidate"]
    print(
        "CANDIDATO CONSERVADOR: "
        f"precisão entre sugestões {candidate['conditional_precision']:.2%} "
        f"({candidate['correct_suggestions']}/{candidate['suggestions']}) | "
        f"cobertura {candidate['coverage']:.2%} "
        f"({candidate['suggestions']}/{candidate['total']}) | "
        f"revisões {candidate['reviews']} | "
        f"NG->OK {candidate['NG_suggested_OK']}",
        flush=True,
    )
    print(
        f"META CANDIDATO >=97% CONDICIONAL: {candidate['target_97pct_met']} "
        "(somente diagnóstico, não aprova produção)",
        flush=True,
    )
    print(
        f"BASELINE: meta anterior de 98% por classe: {stats['target_98pct_met']}",
        flush=True,
    )
    geom = report.get("geometry_experiment")
    if geom:
        gm, gc = geom["results"], geom["shadow_candidate"]
        print(
            f"GEOMETRIA B: acertos {gm['correct']}/{gm['total']} "
            f"NG->OK {gm['NG_as_OK']} | conservadora "
            f"{gc['correct_suggestions']}/{gc['suggestions']} "
            f"| cobertura {gc['coverage']:.2%} "
            f"| meta 97% {gc['target_97pct_met']}",
            flush=True,
        )
    print(f"JSON: {j}\nTXT: {t}", flush=True)
    print("Sem qualquer alteração na memória ou política produtiva.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
