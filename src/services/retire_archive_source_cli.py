"""Retirar do snapshot KNN uma fonte NG já excluída do arquivo local.

Uso:
  python -m src.services.retire_archive_source_cli --source public/ng_archive/ARQUIVO.png
  python -m src.services.retire_archive_source_cli --source public/ng_archive/ARQUIVO.png --apply

Executar com ODIN fechado. Não remove imagens nem julgamentos humanos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.settings import BASE_DIR
from src.core.archive_persistent_memory import PersistentArchiveKNN


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Exclui apenas fontes NG previamente retiradas do acervo; "
                    "sem --apply realiza somente simulação."
    )
    parser.add_argument("--root", type=Path, default=Path(BASE_DIR))
    parser.add_argument("--source", required=True,
                        help="Caminho relativo public/ng_archive/NOME.png")
    parser.add_argument("--apply", action="store_true",
                        help="Confirma a retirada do snapshot e cria memory.json.bak")
    args = parser.parse_args(argv)
    store = PersistentArchiveKNN(root=args.root)
    try:
        result = store.retire_deleted_ng_archive_source(
            args.source, apply=args.apply,
        )
    except (ValueError, OSError) as exc:
        print(json.dumps({
            "status": "BLOCKED",
            "error": str(exc),
            "source_path": args.source,
            "changed": False,
            "xp_commands_sent": 0,
        }, ensure_ascii=False, indent=2))
        return 2

    print(json.dumps({
        "status": "APPLIED" if args.apply and result["changed"] else (
            "NO_SOURCE_IN_SNAPSHOT" if not result["archive_sources_retired"]
            else "DRY_RUN"),
        **result,
        "xp_commands_sent": 0,
        "images_modified": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
