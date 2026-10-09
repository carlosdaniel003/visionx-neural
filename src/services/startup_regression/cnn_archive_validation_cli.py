"""Diagnóstico CNN FALTANDO V2 e MEMÓRIA KNN; não bloqueia o ODIN.

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
            "Validação CNN FALTANDO V2 e MEMÓRIA KNN sobre OK/NG. "
            "Memória KNN exige par exato humano verificado, em todas categorias."
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

    print("Validando CNN FALTANDO V2 e MEMÓRIA KNN no acervo...", flush=True)

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
            f"{count['model_unavailable']} indisponíveis; "
            f"{count['without_memory_coverage']} sem cobertura",
            flush=True,
        )
    print(f"JSON: {json_path}\nTXT: {txt_path}", flush=True)
    print("Gate do ODIN ainda não foi ativado.", flush=True)
    return 0 if report["cnn_and_knn_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
