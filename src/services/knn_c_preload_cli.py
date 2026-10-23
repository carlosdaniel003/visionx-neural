"""Preparar cache KNN C fora do ODIN, com o programa fechado.

Exemplo Windows: python -m src.services.knn_c_preload_cli
Sem comandos 0/1, CNN, captura XP nem alteração de PNGs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config.settings import BASE_DIR
from src.core.knn_c_live import LiveKNNStrategyC


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        "Prepara somente assinaturas de imagens OK/NG já existentes para "
        "a KNN C, sem abrir interface ou conectar-se ao XP."
    ))
    parser.add_argument("--root", type=Path, default=Path(BASE_DIR),
                        help="Pasta do projeto ODIN (default: BASE_DIR)")
    args = parser.parse_args(argv)
    bank = LiveKNNStrategyC(root=args.root)
    result = bank.warmup()
    print(json.dumps({
        "status": result.get("status"),
        "result": result,
        "progress": bank.progress(),
        "diagnostic_only": True,
        "xp_commands_sent": 0,
        "archive_pngs_modified": False,
        "classifier_weights_changed": False,
    }, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "READY" else 2


if __name__ == "__main__":
    sys.exit(main())
