"""Avaliação seletiva KNN separada por dia, read-only e sem aprovação de produção.

    python -m src.services.startup_regression.knn_selective_holdout_cli

Saída restrita a reports/startup_regression. Sem retreinamento,
migração, edição dos pesos, KNN runtime ou ativação de startup gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.services.startup_regression.archive_inventory_report import _atomic_text
from .knn_selective_holdout import selective_knn_holdout


def write_holdout_reports(report: dict, target: Path):
    root = Path(report["root"]).resolve()
    target = Path(target).resolve()
    reports_root = (root / "reports").resolve()
    if target == reports_root or reports_root not in target.parents:
        raise ValueError("Relatórios só podem ser gravados abaixo de reports/")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = target / f"knn_selective_holdout_{stamp}.json"
    txt_path = target / f"knn_selective_holdout_{stamp}.txt"
    _atomic_text(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    lines = [
        "ODIN — KNN SELETIVA COM VALIDAÇÃO AGRUPADA POR DIA",
        "READ-ONLY | NUNCA LIBERA A PRODUÇÃO | SEM ALTERAR KNN/CNN",
        "",
        f"STATUS: {report['status']}",
        f"Registros de assinatura elegíveis: {report['eligible_records']}",
        f"Dias identificados: {report['groups_total']}",
        f"JSONs sem data verificável: {len(report['ungrouped_paths'])}",
        f"Registros excluídos por sobreposição entre partições: "
        f"{len(report['excluded_cross_partition'])}",
        "Agrupamento por DATA INFERIDA DO NOME, NÃO por placa/lote físico.",
    ]
    for part, days in report["groups"].items():
        lines.append(
            f"  {part}: {report['after_leakage_exclusion'][part]} "
            f"registros, classes={report['class_counts_after_exclusion'][part]}, "
            f"dias={', '.join(days)}"
        )
    lines.extend([
        "",
        "POLÍTICA SELECIONADA APENAS NA CALIBRAÇÃO:",
        str(report["calibrated_policy"] or "NENHUMA — ABSTER POR PADRÃO"),
        f"Políticas candidatas examinadas: {len(report['calibration_grid'])}",
    ])
    for title, key in (
        ("CALIBRAÇÃO", "calibration"), ("TESTE RESERVADO", "heldout_test"),
    ):
        lines.append("")
        lines.append(title)
        stats = report[key]
        if not stats:
            lines.append("  NÃO AVALIADO — partições insuficientes")
            continue
        for metric in (
            "evaluated", "actual_NG", "actual_OK", "NG_released_as_OK",
            "NG_auto_NG", "NG_review", "OK_auto_OK", "OK_false_NG",
            "OK_review", "total_reviews",
        ):
            lines.append(f"  {metric}: {stats[metric]}")
        lines.append("  Por categoria:")
        for category, values in stats["by_category"].items():
            lines.append(f"    {category}: {values}")
    lines.extend([
        "",
        "Limite superior unilateral 95% do risco NG->OK, se 0 falhas: "
        f"{report['heldout_zero_ng_miss_upper_bound_95']}",
        "",
        "CASOS DE TESTE QUE NÃO RECEBERAM LIBERAÇÃO AUTOMÁTICA CORRETA:",
    ])
    if report["heldout_test"]:
        for case in report["heldout_test"]["cases"]:
            if case["decision"] != case["expected_human_label"]:
                lines.append(
                    f"  {case['path']} | esperado={case['expected_human_label']} "
                    f"| decisão={case['decision']} | {case['reason']} "
                    f"| scoreNG={case['vote_ng']}"
                )
    lines.extend(["", report["note"]])
    _atomic_text(txt_path, "\n".join(lines) + "\n")
    return json_path, txt_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ajusta a política só na calibração, mede em dias separados"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()

    print("ODIN — KNN seletiva: teste agrupado em datas sem escrita", flush=True)

    def progress(n, total, path):
        if n == 1 or n % 50 == 0 or n == total:
            print(f"[{n}/{total}] {path}", flush=True)

    report = selective_knn_holdout(root, progress=progress)
    output = root / "reports" / "startup_regression"
    json_path, txt_path = write_holdout_reports(report, output)
    print(f"STATUS: {report['status']}", flush=True)
    for part in ("memory_train", "calibration", "heldout_test"):
        print(f"  {part}: {report['class_counts_after_exclusion'][part]}", flush=True)
    if report["heldout_test"]:
        print(
            "Teste reservado — NG liberados incorretamente como OK:",
            report["heldout_test"]["NG_released_as_OK"], flush=True
        )
        print(
            "Teste reservado — OK liberados automaticamente:",
            report["heldout_test"]["OK_auto_OK"], flush=True
        )
    print(f"JSON: {json_path}\nTXT: {txt_path}", flush=True)
    print("Nenhum motor de produção alterado. Gate continua desativado.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
