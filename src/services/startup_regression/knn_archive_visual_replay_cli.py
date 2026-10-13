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
        f"Inventário total: {report['inventory_png_count']}",
        f"Acertos: {metrics['correct']}/{metrics['total']}",
        f"Meta KNN >=98% em OK E em NG: {metrics['target_98pct_met']}",
        f"Status: {metrics['status_counts']}",
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
        "--protocol", choices=VALID_PROTOCOLS, default="leave_day_out",
        help="leave_day_out (padrão, mais rigoroso) ou leave_one_out (diagnóstico)",
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
        root, k=args.k, protocol=args.protocol, progress=progress
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
    print(
        f"META >=98% POR CLASSE: {stats['target_98pct_met']}",
        flush=True,
    )
    print(f"JSON: {j}\nTXT: {t}", flush=True)
    print("Sem qualquer alteração na memória ou política produtiva.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
