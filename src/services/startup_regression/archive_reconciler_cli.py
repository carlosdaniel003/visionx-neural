"""CLI de reconciliação read-only do dataset KNN e do acervo visual.

    python -m src.services.startup_regression.archive_reconciler_cli

Somente diagnóstico. Saída fora dos arquivos originais; nenhuma CNN é
iniciada e nenhum bloqueio de operação é adicionado ao main.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .archive_reconciler import reconcile_archive, write_reconciliation_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audita a compatibilidade de arquivos OK/NG com memória KNN"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None
        else root / "reports" / "startup_regression"
    )

    print("Reconciliação de acervo x MEMÓRIA KNN — somente leitura", flush=True)

    def progress(done: int, total: int, path: str) -> None:
        if done == 1 or done % 25 == 0 or done == total:
            print(f"[{done}/{total}] {path}", flush=True)

    report = reconcile_archive(root, progress=progress)
    json_path, txt_path = write_reconciliation_report(report, output)
    print(f"PNG inspecionados: {report['archive_png_count']}", flush=True)
    print(f"JSONs auditados: {report['memory_json_count']}", flush=True)
    for state, count in report["case_status_counts"].items():
        print(f"  {state}: {count}", flush=True)
    print(f"JSON: {json_path}\nTXT: {txt_path}", flush=True)
    print("Nenhuma memória modificada; o gate permanece desativado.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
