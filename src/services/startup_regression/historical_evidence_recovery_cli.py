"""Comando seguro para procurar evidências de imagens legadas no ODIN.

    python -m src.services.startup_regression.historical_evidence_recovery_cli

Varre apenas public/dataset, public/ok_archive e public/ng_archive.
Relatórios em reports/startup_regression. Sem escrita de memória, CNN ou KNN.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .historical_evidence_recovery import (
    inspect_historical_evidence, write_evidence_report,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Recupera somente evidências históricas, nunca altera memória"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()

    print("ODIN — busca read-only de fontes, gabaritos e testes por hash", flush=True)

    def progress(done, total, path):
        if done == 1 or done % 50 == 0 or done == total:
            print(f"[{done}/{total}] {path}", flush=True)

    report = inspect_historical_evidence(root, progress=progress)
    target = root / "reports" / "startup_regression"
    json_path, text_path = write_evidence_report(report, target)
    print(
        f"JSONs: {report['scanned_jsons']} | v2/legados: "
        f"{report['legacy_jsons']} | PNGs: {report['scanned_pngs']}",
        flush=True,
    )
    for name, count in report["status_counts"].items():
        print(f"  {name}: {count}", flush=True)
    print(f"JSON: {json_path}\nTXT: {text_path}", flush=True)
    print(
        "Diagnóstico somente leitura. Não habilita startup gate nem "
        "migração de memórias.", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
