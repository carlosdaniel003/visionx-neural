"""Executar diagnóstico DESLOCADO OK sem desenho e sem treinar modelo.

No PC da fábrica:
    python -m src.scripts.diagnose_deslocado_ok_geometry
    python -m src.scripts.diagnose_deslocado_ok_geometry --manifest "...\manifest.json"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.services.deslocado_ok_geometry import diagnose_ok_geometry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspecionar geometricamente os pares OK DESLOCADO sem desenho/KNN"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Manifesto original já extraído; sem parâmetro usa o último",
    )
    args = parser.parse_args(argv)
    report, directory = diagnose_ok_geometry(args.root, args.manifest)
    print("DESLOCADO OK — diagnóstico offline concluído")
    print("Imagens OK:", report["total_ok_images"])
    print("Eventos candidatos:", report["total_candidate_events"])
    print("Por iluminação:", report["source_lights"])
    print("Qualidade das evidências:", report["evidence_statuses"])
    print("Falsos NG anteriores localizados:", report["historical_false_ng_cases_present"])
    print("JSON:", directory / "deslocado_ok_geometry.json")
    print("TXT:", directory / "deslocado_ok_geometry.txt")
    print("SEM CNN treinada, SEM KNN, SEM alteração operacional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
