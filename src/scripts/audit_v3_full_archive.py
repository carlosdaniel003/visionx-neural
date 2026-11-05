"""CLI offline: testar TODOS os PNGs do acervo contra CNN V3 e memória visual.

Saída única em reports/aoi_v3_full_archive_audit/<run>/:
  - full_archive_v3.json: todos os exemplos, scores e provas
  - full_archive_v3.csv: 1 linha por PNG
  - full_archive_v3.txt: diagnóstico agregado + divergências
Nenhum PNG, checkpoint, arquivo KNN ou comando XP é modificado.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path

from src.config.settings import BASE_DIR
from src.services.aoi_v3_archive_batch import audit_full_archive


def _summary_text(report: dict) -> str:
    summary = report["summary"]
    coverage = report["coverage"]
    lines = [
        "ODIN • CNN V3 / ACERVO COMPLETO • SOMENTE LEITURA",
        "=" * 75,
        "Este relatório compara a rede com RÓTULOS DE PASTA PROVISÓRIOS.",
        "Não é avaliação cega nem prova de qualidade física de cada arquivo.",
        "",
        f"PNG inventariados: {coverage['png_in_inventory']}",
        f"Exemplos relatados: {coverage['rows_in_report']}",
        f"Inferências V3 válidas: {summary['evaluated']}",
        f"Sem inferência válida: {summary['failed_or_unavailable']}",
        f"OK na pasta: {summary['OK_total']}",
        f"  V3 OK: {summary['OK_correct']} | V3 NG (falso alarme relativo à pasta): {summary['OK_false_NG']}",
        f"NG na pasta: {summary['NG_total']}",
        f"  V3 NG: {summary['NG_correct']} | V3 OK (risco de escape relativo à pasta): {summary['NG_false_OK']}",
        f"Concordância global: {summary['accuracy_against_folder']}",
        f"Concordância balanceada: {summary['balanced_accuracy_against_folder']}",
        f"Todos avaliados sem falhas: {report['all_files_evaluated']}",
        f"PNG alterados durante o teste: {len(coverage['source_changed_during_run'])}",
        f"Conflitos exatos OK/NG no inventário: {len(coverage['cross_label_pixel_conflicts'])}",
        "",
        "RECUPERAÇÃO DE MEMÓRIA: FONTE EXATA vs VIZINHO",
        "-" * 75,
        f"Outras fontes quase iguais, mesmo rótulo: {report['leave_one_out_archive_probe']['same_label_near']}",
        f"Quase iguais com conflito de rótulos: {report['leave_one_out_archive_probe']['conflicting_near']}",
        "A própria imagem não conta como exemplo novo no leave-one-out.",
        "A assinatura visual NÃO valida um rótulo nem autoriza um comando 0/1.",
        "",
        "POR CATEGORIA AOI / ILUMINAÇÃO",
        "-" * 75,
    ]
    for key, data in report["by_category_light"].items():
        lines.append(
            f"{key}: {data['evaluated']} avaliados | "
            f"OK->NG={data['OK_false_NG']} | NG->OK={data['NG_false_OK']} | "
            f"indisponível={data['failed_or_unavailable']}"
        )
    lines.extend(["", "DIVERGÊNCIAS E FALHAS (ARQUIVOS INDIVIDUAIS)", "-" * 75])
    for row in report["cases"]:
        if row["status"] != "EVALUATED" or not row.get("agrees_with_folder"):
            lines.append(
                f"{row['path']} | pasta={row['archive_label']} "
                f"| V3={row['predicted']} | NG={row['v3_ng_score_uncalibrated']} "
                f"| estado={row['status']} "
                f"| geometria={row.get('extraction_geometry', {}).get('status', '-')}"
                f"| erro={'; '.join(row['issues'])}"
            )
    if not any(
        r["status"] != "EVALUATED" or not r.get("agrees_with_folder")
        for r in report["cases"]
    ):
        lines.append("Nenhuma divergência contra os rótulos de pasta.")
    return "\n".join(lines) + "\n"


def _write_reports(report: dict, output_base: Path) -> Path:
    output_base = Path(output_base).resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    run = output_base / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run.mkdir(exist_ok=False)
    json_target = run / "full_archive_v3.json"
    json_target.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    columns = [
        "path", "archive_label", "label_provenance",
        "category_hint", "lighting", "model_role", "status",
        "predicted", "v3_ng_score_uncalibrated", "agrees_with_folder",
        "extraction_category", "reference_shape", "test_shape",
        "extraction_geometry", "source_file_sha256", "pixel_sha256",
        "v3_checkpoint_sha256", "leave_one_out", "full_frame_exact",
        "issues",
    ]
    with (run / "full_archive_v3.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in report["cases"]:
            writer.writerow({
                name: (
                    json.dumps(row[name], ensure_ascii=False)
                    if isinstance(row.get(name), (dict, list)) else row.get(name, "")
                ) for name in columns
            })
    (run / "full_archive_v3.txt").write_text(
        _summary_text(report), encoding="utf-8",
    )
    return run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validar CNN V3 em todo acervo OK/NG original, sem MSS",
    )
    parser.add_argument("--root", type=Path, default=BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    output = (
        arguments.output_dir.resolve()
        if arguments.output_dir is not None
        else root / "reports" / "aoi_v3_full_archive_audit"
    )
    # Não produzir relatórios dentro de qualquer arquivo operacional.
    for source in ("public/ok_archive", "public/ng_archive"):
        folder = (root / source).resolve()
        if output == folder or folder in output.parents:
            parser.error("Relatórios não podem ser escritos no acervo de imagens")

    def progress(count, total, row):
        status = row["status"]
        pred = row.get("predicted") or "INDEFINIDO"
        score = row.get("v3_ng_score_uncalibrated")
        score_str = f" NG_score={score:.6f}" if score is not None else ""
        print(
            f"CNN V3 ACERVO {count}/{total}: {status} "
            f"pasta={row['archive_label']} V3={pred}{score_str} "
            f"{row['path']}",
            flush=True,
        )

    report = audit_full_archive(root, progress=progress)
    run_dir = _write_reports(report, output)
    print(_summary_text(report))
    print("Relatórios salvos em: " + str(run_dir))
    return 0 if report["all_files_evaluated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
