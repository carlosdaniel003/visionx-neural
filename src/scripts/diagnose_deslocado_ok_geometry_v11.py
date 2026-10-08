"""Reavaliar DESLOCADO OK: ORB/AKAZE + RANSAC, sem desenho/treino.

    python -m src.scripts.diagnose_deslocado_ok_geometry_v11
    python -m src.scripts.diagnose_deslocado_ok_geometry_v11 --manifest "...\manifest.json"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.services.deslocado_ok_geometry_v11 import diagnose_ok_geometry_v11


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnóstico geométrico v1.1 por ORB/AKAZE, sem KNN nem máscaras"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    report, folder = diagnose_ok_geometry_v11(args.root, args.manifest)
    print("DESLOCADO GEOMETRIA v1.1 (sem treino/KNN)")
    print("Imagens:", report["total_ok_images"])
    print("Eventos candidatos:", report["total_events_candidate"])
    print("Antes / depois:", report["comparison"])
    print("Por iluminação:", report["by_lighting"])
    print("Motivos de insuficiência:", report["reason_codes"])
    print("JSON:", folder / "deslocado_ok_geometry_v11.json")
    print("TXT:", folder / "deslocado_ok_geometry_v11.txt")
    print("Registro visual nao e prova de alinhamento fisico. Producao inalterada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
