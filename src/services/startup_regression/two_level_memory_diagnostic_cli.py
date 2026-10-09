"""Diagnóstico separado: KNN por assinatura vs plano de KNN visual exata.

    python -m src.services.startup_regression.two_level_memory_diagnostic_cli

Não modifica dataset, KNNExpert, CNN, imagens, treinamento ou main.py.
Saída em reports/startup_regression, nunca dentro de public/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.services.startup_regression.archive_inventory_report import _atomic_text

from .archive_exact_memory_plan import plan_archive_exact_memory
from .legacy_knn_signature_audit import audit_signature_knn


def write_dual_level_reports(signatures: dict, archive: dict, target: Path):
    root_a = Path(signatures["root"]).resolve()
    root_b = Path(archive["root"]).resolve()
    if root_a != root_b:
        raise ValueError("Relatórios não vieram da mesma estação")
    target = Path(target).resolve()
    root_reports = (root_a / "reports").resolve()
    if target == root_reports or root_reports not in target.parents:
        raise ValueError("Relatórios só podem ser salvos em reports/ e subpasta")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = target / f"memory_levels_{stamp}.json"
    txt_path = target / f"memory_levels_{stamp}.txt"
    report = {
        "schema": "visionx.memory_two_levels_diagnostic.v1",
        "mode": "READ_ONLY_NO_MIGRATION",
        "signature_memory": signatures,
        "exact_archive_plan": archive,
        "modifies_production": False,
        "startup_gate_enabled": False,
        "two_results_must_not_be_merged": True,
    }
    _atomic_text(
        json_path, json.dumps(
            report, indent=2, ensure_ascii=False, allow_nan=False
        ) + "\n"
    )
    lines = [
        "ODIN — DOIS NÍVEIS DE MEMÓRIA, SOMENTE SIMULAÇÃO",
        "SEM MIGRAÇÃO | SEM TREINO | SEM GATE | NÃO APROVA 100% AUTOMÁTICO",
        "",
        "MEMÓRIA HISTÓRICA POR ASSINATURA — TESTE LEAVE-ONE-RECORD-OUT",
        f"JSONs examinados: {signatures['json_records_total']}",
        f"Assinaturas elegíveis: {signatures['eligible_signatures']}",
    ]
    for k, v in signatures["status_counts"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("REGISTROS EXCLUÍDOS DO TESTE DE ASSINATURA:")
    for k, v in signatures["rejected_records"].items():
        lines.append(f"  {k}: {v}")
    lines.extend([
        "",
        "ARQUIVO HISTÓRICO — PROPOSTA DE ÍNDICE VISUAL EXATO",
        f"PNGs examinados: {archive['png_count']}",
    ])
    for k, v in archive["status_counts"].items():
        lines.append(f"  {k}: {v}")
    lines.extend([
        "",
        f"Verificados v3: {archive['verified_existing']}",
        f"Legados exatos ainda pendentes: {archive['legacy_exact_pending']}",
        f"Pendentes de procedência humana: {archive['pending_human_evidence']}",
        "Autorizados para migração: 0",
        "",
        "DIVERGÊNCIAS E PENDÊNCIAS — ASSINATURAS:",
    ])
    for row in signatures["cases"]:
        if row["status"] not in {"CONCORDA"}:
            lines.append(
                f"  {row['status']} | {row['label'] if 'label' in row else row['expected_human_label']}"
                f" | {row['path']} | vizinhos={row['neighbors_used']}"
            )
    lines.append("")
    lines.append("DIVERGÊNCIAS E PENDÊNCIAS — ARQUIVO EXATO:")
    for row in archive["cases"]:
        if row["status"] not in {"JA_VERIFICADO_EXATO_V3"}:
            lines.append(f"  {row['status']} | {row['source_path']}")
    lines.extend([
        "", signatures["note"], archive["note"],
    ])
    _atomic_text(txt_path, "\n".join(lines) + "\n")
    return json_path, txt_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audita assinaturas KNN e prepara memória exata, sem gravação"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()

    def progress_sig(done, total, path):
        if done == 1 or done % 50 == 0 or done == total:
            print(f"Assinaturas [{done}/{total}]: {path}", flush=True)

    print("Auditoria KNN histórica sem consulta ao próprio registro...", flush=True)
    signatures = audit_signature_knn(
        root, top_k=args.top_k, progress=progress_sig
    )

    def progress_archive(done, total, path):
        if done == 1 or done % 25 == 0 or done == total:
            print(f"Arquivo [{done}/{total}]: {path}", flush=True)

    print("Construindo plano de memória exata sem salvar imagens...", flush=True)
    archive = plan_archive_exact_memory(
        root, progress=progress_archive
    )
    paths = write_dual_level_reports(
        signatures, archive, root / "reports" / "startup_regression"
    )
    print("Assinaturas:", signatures["status_counts"], flush=True)
    print("Arquivo:", archive["status_counts"], flush=True)
    print(f"JSON: {paths[0]}\nTXT: {paths[1]}", flush=True)
    print("KNN e CNN de produção não foram alteradas.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
