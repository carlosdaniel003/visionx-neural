"""CLI de robustez KNN: variações LEVES sobre PNGs OK/NG arquivados.

python -m src.services.startup_regression.knn_archive_augmented_replay_cli

Somente consulta original-alterada: não escreve imagens, treina nem julga XP.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.services.startup_regression.archive_inventory_report import _atomic_text
from src.services.startup_regression.knn_archive_augmented_replay import (
    replay_augmented_archive,
)


def write_augmented_reports(report: dict, destination: Path) -> tuple[Path, Path]:
    root = Path(report["root"]).resolve()
    output = Path(destination).resolve()
    if (root / "reports").resolve() not in output.parents:
        raise ValueError("Relatórios somente em reports/ do projeto")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = output / f"knn_archive_augmented_{stamp}.json"
    txt_path = output / f"knn_archive_augmented_{stamp}.txt"
    _atomic_text(json_path, json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False,
    ) + "\n")
    lines = [
        "ODIN — TESTE DE ROBUSTEZ KNN COM PNGS VARIADOS (OFFLINE)",
        "Consultas levemente variadas vs. OUTROS eventos originais do acervo.",
        "NÃO é treinamento, não cria imagens, não altera a produção/XP/memória.",
        "A categoria AOI não participa da seleção ou da verdade OK/NG.",
        "O rótulo esperado é o da pasta original; nenhum NG é inventado.",
        f"Imagens originais: {report['inventory_png_count']}",
        f"Variações por PNG: {report['variations_per_original']}",
        f"Consultas: {report['expected_query_count']}",
        f"Concluído: {report['complete']} | interrompido: {report['canceled']}",
        f"Falhas de extração: {len(report['original_errors'])}",
    ]
    def describe(name: str, basic: dict, selective: dict) -> None:
        rows = basic["by_label"]
        lines.extend([
            "",
            f"{name}:",
            f"  KNN base: {basic['correct']}/{basic['total']}; "
            f"OK {rows['OK']['correct']}/{rows['OK']['total']}; "
            f"NG {rows['NG']['correct']}/{rows['NG']['total']}",
            f"  FALSOS OK (NG->OK): {basic['NG_as_OK']} | "
            f"FALSOS NG (OK->NG): {basic['OK_as_NG']}",
            f"  Seletiva: {selective['correct_suggestions']}/"
            f"{selective['suggestions']} corretas | "
            f"precisão {100*selective['conditional_precision']:.2f}% | "
            f"cobertura {100*selective['coverage']:.2f}% | "
            f"revisões {selective['reviews']} | "
            f"NG->OK {selective['NG_suggested_OK']} | "
            f"meta experimental >=97% {selective['target_97pct_met']}",
        ])
    describe("A — visual", report["results"], report["shadow_candidate"])
    geo = report["geometry_experiment"]
    describe("B — visual + geometria", geo["results"], geo["shadow_candidate"])
    c = geo["consensus_candidate"]
    lines.extend([
        "",
        "C — concordância de A e B (somente sugestões):",
        f"  {c['correct_suggestions']}/{c['suggestions']} corretas | "
        f"precisão {100*c['conditional_precision']:.2f}% | "
        f"cobertura {100*c['coverage']:.2f}% | "
        f"revisões {c['reviews']} | NG->OK {c['NG_suggested_OK']} | "
        f"meta >=97% {c['target_97pct_met']}",
    ])
    for name, data in report["by_variant"].items():
        basic = data["results"]
        c = data["consensus_candidate"]
        lines.append(
            f"  VARIAÇÃO {name}: A {basic['correct']}/{basic['total']} "
            f"| C {c['correct_suggestions']}/{c['suggestions']} "
            f"| NG->OK C {c['NG_suggested_OK']}"
        )
    lines.extend([
        "",
        "LIMITAÇÕES:",
        report["limitations"],
        "Somente relatório offline: nenhuma KNN produtiva foi retreinada.",
        "Não comparar 100% seletivo com acurácia de todas as imagens.",
        "Nenhuma conclusão de desempenho em eventos realmente independentes.",
    ])
    _atomic_text(txt_path, "\n".join(lines) + "\n")
    return json_path, txt_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Variações leves OK/NG temporárias, KNN experimental A/B/C"
    )
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument(
        "--variations", type=int, choices=(1, 2), default=2,
        help="1: iluminação/ruído; 2: também variação óptica e jitter comum.",
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()

    def progress(stage, completed, total, _path):
        if completed == 1 or completed % 50 == 0 or completed == total:
            print(f"{stage}: {completed}/{total}", flush=True)

    print("Teste KNN em variações sintéticas; nenhuma imagem será salva.", flush=True)
    report = replay_augmented_archive(
        root, variations=args.variations, k=args.k, progress=progress,
    )
    json_path, txt_path = write_augmented_reports(
        report, root / "reports" / "startup_regression",
    )
    print(
        f"Completo: {report['complete']} | "
        f"A: {report['results']['correct']}/{report['results']['total']} | "
        f"NG->OK A: {report['results']['NG_as_OK']}",
        flush=True,
    )
    a = report["shadow_candidate"]
    b = report["geometry_experiment"]["shadow_candidate"]
    c = report["geometry_experiment"]["consensus_candidate"]
    for name, metric in (("A", a), ("B", b), ("C", c)):
        print(
            f"{name} seletiva: {metric['correct_suggestions']}/"
            f"{metric['suggestions']} | cobertura {metric['coverage']:.2%} | "
            f"NG->OK {metric['NG_suggested_OK']} | "
            f"meta 97% {metric['target_97pct_met']}",
            flush=True,
        )
    print(f"JSON: {json_path}\nTXT: {txt_path}", flush=True)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
