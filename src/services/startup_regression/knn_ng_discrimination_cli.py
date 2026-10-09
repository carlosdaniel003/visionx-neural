"""CLI de auditoria NG KNN (somente simulação offline).

    python -m src.services.startup_regression.knn_ng_discrimination_cli

Relatórios somente em reports/startup_regression. Nenhuma gravação no
dataset; KNN de produção, CNN e gate de startup permanecem inalterados.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.services.startup_regression.archive_inventory_report import _atomic_text
from .knn_ng_discrimination_audit import audit_knn_ng_discrimination


def write_discrimination_reports(report: dict, output_dir: Path):
    root = Path(report["root"]).resolve()
    output = Path(output_dir).resolve()
    reports_root = (root / "reports").resolve()
    if output == reports_root or reports_root not in output.parents:
        raise ValueError("Relatórios permitidos apenas em subpasta de reports/")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = output / f"knn_ng_audit_{stamp}.json"
    txt_path = output / f"knn_ng_audit_{stamp}.txt"
    _atomic_text(
        json_path,
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    lines = [
        "ODIN — AUDITORIA DE DISCRIMINAÇÃO NG DA MEMÓRIA KNN",
        "READ-ONLY | SEM ALTERAÇÃO DE PRODUÇÃO | NÃO LIBERA STARTUP",
        f"JSONs: {report['original_json_records']}",
        f"Assinaturas elegíveis: {report['evaluated_records']}",
        f"Grupos de assinatura duplicada: {report['duplicate_signature_groups']}",
        f"Registros em grupos duplicados: {report['records_inside_duplicate_groups']}",
        f"Comparações visuais de assinaturas: {report['pair_similarity_comparisons']}",
        f"Limite experimental de similaridade: {report['review_similarity_min_experimental']}",
        f"Margem experimental de revisão: {report['review_margin_experimental']}",
        "",
        "COMPARAÇÃO — MESMOS REGISTROS, CINCO MODOS DE VOTO:",
    ]
    for mode, stats in report["comparisons"].items():
        lines.extend([
            f"\n{mode}",
            f"  NG detectados automaticamente: {stats['correct_NG']}/{stats['expected_NG']}",
            f"  NG liberados incorretamente como OK: {stats['missed_NG_as_OK']}",
            f"  NG enviados para revisão: {stats['review_NG']}",
            f"  OK reconhecidos automaticamente: {stats['correct_OK']}/{stats['expected_OK']}",
            f"  OK enviados para revisão: {stats['review_OK']}",
            f"  OK acusados incorretamente NG: {stats['false_NG_on_OK']}",
            f"  Concordância somente entre decisões automáticas: "
            f"{stats['automatic_accuracy_on_decided_only']}",
            f"  Percentual total de revisões: {stats['review_fraction_total']}",
        ])
        lines.append("  Por categoria:")
        for category, values in stats["by_category"].items():
            lines.append(
                f"    {category}: "
                f"NG seguros={values.get('correct_NG', 0)}, "
                f"NG falsamente OK={values.get('missed_NG_as_OK', 0)}, "
                f"NG revisão={values.get('review_NG', 0)}, "
                f"OK falsamente NG={values.get('false_NG_on_OK', 0)}, "
                f"OK revisão={values.get('review_OK', 0)}"
            )
    lines.extend([
        "", "DEFEITOS NG COM LIBERAÇÃO INCORRETA OU REVISÃO PENDENTE:",
    ])
    for item in report["cases"]:
        if item["expected_human_label"] != "NG":
            continue
        baseline = item["modes"]["BASELINE_TOP5"]
        tested = item["modes"]["BALANCEADO_COM_REVISAO"]
        if baseline["prediction"] != "OK" and tested["prediction"] == "NG":
            continue
        lines.append(
            f"  {item['path']} | baseline={baseline['status']} "
            f"(votoNG={baseline['vote_ng']}) | "
            f"experimental={tested['status']} "
            f"(votoNG={tested['vote_ng']}) | "
            f"duplicatas excluídas={tested['excluded_query_signature_copies']}"
        )
    lines.extend([
        "", "AVISO:", report["limitations"],
        "Não converter revisão em acerto, nem alterar o motor de produção "
        "apenas para elevar as métricas.",
    ])
    _atomic_text(txt_path, "\n".join(lines) + "\n")
    return json_path, txt_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audita desequilíbrio OK/NG, duplicatas e revisão na KNN"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--review-margin", type=float, default=.10)
    parser.add_argument("--min-similarity", type=float, default=.80)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()

    print("Auditoria offline KNN NG — sem modificar dataset ou motor", flush=True)

    def progress(done, total, path):
        if done == 1 or done % 50 == 0 or done == total:
            print(f"[{done}/{total}] {path}", flush=True)

    report = audit_knn_ng_discrimination(
        root, k=args.top_k, per_label=args.per_class,
        min_similarity=args.min_similarity,
        review_margin=args.review_margin,
        progress=progress,
    )
    json_path, txt_path = write_discrimination_reports(
        report, root / "reports" / "startup_regression"
    )
    for name, stats in report["comparisons"].items():
        print(
            f"{name}: NG detectados={stats['correct_NG']}, "
            f"NG liberados como OK={stats['missed_NG_as_OK']}, "
            f"NG revisão={stats['review_NG']}, "
            f"OK falsamente NG={stats['false_NG_on_OK']}, "
            f"OK revisão={stats['review_OK']}",
            flush=True,
        )
    print(f"JSON: {json_path}\nTXT: {txt_path}", flush=True)
    print("Nenhuma mudança de memória, inferência operacional ou gate.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
