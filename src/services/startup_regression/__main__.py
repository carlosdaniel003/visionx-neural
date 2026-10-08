"""Executar: python -m src.services.startup_regression [--root PASTA]."""

from __future__ import annotations

import argparse
from pathlib import Path

from .archive_inventory import inventory_archives
from .archive_inventory_report import write_reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inventaria arquivos visuais OK/NG, sem reclassificar nem alterar fotos."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="Raiz do repositório; padrão: raiz detectada automaticamente",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Relatórios fora dos arquivos/dataset; padrão: reports/startup_regression",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "reports" / "startup_regression"
    )

    print(f"Inventariando arquivos da AOI em: {root / 'public'}", flush=True)

    def progress(current: int, total: int, filename: str) -> None:
        if current == 1 or current == total or current % 25 == 0:
            print(f"Inventário: {current}/{total} — {filename}", flush=True)

    report = inventory_archives(root, progress=progress)
    json_path, text_path = write_reports(report, output)
    summary = report["summary"]
    print(
        f"Concluído: {summary['png_count']} PNG, "
        f"{summary['valid_png']} legíveis, "
        f"{summary['invalid_png']} inválidos, "
        f"{summary['issue_count']} pendências.",
        flush=True,
    )
    print(f"Relatório JSON: {json_path}", flush=True)
    print(f"Relatório TXT:  {text_path}", flush=True)
    print("Etapa 1 concluída. Nenhuma imagem foi julgada OK/NG pela IA.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
