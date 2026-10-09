"""Executar CNN FALTANDO v2 em SOMBRA nas categorias reais do arquivo AOI.

    python -m src.scripts.audit_faltando_cross_category

NÃO envia 0/1 ao XP, NÃO altera Produção, KNN, checkpoints ou datasets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.services.faltando_cross_category_audit import audit_cross_category


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Auditar todas as categorias AOI com CNN FALTANDO v2 offline"
    )
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    result, folder = audit_cross_category(args.root)
    print("CNN FALTANDO v2 — auditoria SOMBRA de todas as categorias AOI")
    print("Total PNG:", result["total_images"])
    print("Resumo:", result["summary"])
    print("Por categoria:")
    for category, counts in result["by_category"].items():
        print(" ", category, ":", counts)
    print("JSON:", folder / "cross_category_audit.json")
    print("TXT:", folder / "cross_category_audit.txt")
    print("Nenhum comando 0/1, treinamento ou alteração produtiva efetuada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
