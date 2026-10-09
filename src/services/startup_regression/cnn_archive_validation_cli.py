"""Diagnóstico das CNNs; ainda não é o bloqueio de startup do ODIN.

Uso:
    python -m src.services.startup_regression.cnn_archive_validation_cli
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .cnn_archive_validation import validate_archive_cnns, write_cnn_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validação CNN FALTANDO v2 e CNN MEMÓRIA sobre acervo OK/NG. "
            "A CNN MEMÓRIA precisa ser conectada antes do gate em main.py."
        )
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    out = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "reports" / "startup_regression"
    )

    print("Validando CNNs em todos os PNGs do acervo (sem KNN)...", flush=True)

    def progress(done: int, total: int, current: str) -> None:
        if done == 1 or done % 25 == 0 or done == total:
            print(f"{done}/{total} — {current}", flush=True)

    report = validate_archive_cnns(root, progress=progress)
    json_path, txt_path = write_cnn_report(report, out)
    for name, count in report["models"].items():
        print(
            f"{name}: {count['passed']}/{count['eligible']} aprovados; "
            f"{count['regressions']} regressões; "
            f"{count['invalid']} inválidos; "
            f"{count['model_unavailable']} sem modelo",
            flush=True,
        )
    print(f"JSON: {json_path}\nTXT: {txt_path}", flush=True)
    print("Gate do ODIN ainda não foi ativado.", flush=True)
    return 0 if report["both_cnns_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
