"""Limpeza segura da memória local usada pelo dataset do VisionX Neural."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from src.config.settings import settings


def _count_dataset_entries(root: Path) -> tuple[int, int]:
    """Conta arquivos e subpastas sem seguir links simbólicos."""
    file_count = 0
    directory_count = 0

    for _, directory_names, file_names in os.walk(root, followlinks=False):
        file_count += len(file_names)
        directory_count += len(directory_names)

    return file_count, directory_count


def clear_local_dataset() -> dict:
    """
    Exclui o conteúdo das pastas OK/NG, preservando as pastas raiz.

    A validação impede que um caminho configurado fora de DATASET_DIR seja
    apagado acidentalmente.
    """
    dataset_root = Path(settings.DATASET_DIR).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    targets = [Path(settings.ANOMALY_DIR), Path(settings.NORMAL_DIR)]
    deleted_files = 0
    deleted_directories = 0
    errors: list[str] = []

    for target in targets:
        try:
            target.mkdir(parents=True, exist_ok=True)
            resolved_target = target.resolve()

            if resolved_target == dataset_root or dataset_root not in resolved_target.parents:
                errors.append(
                    f"Caminho recusado por segurança: {resolved_target}"
                )
                continue

            target_files, target_directories = _count_dataset_entries(target)

            for child in list(target.iterdir()):
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)

            deleted_files += target_files
            deleted_directories += target_directories
        except Exception as exc:
            errors.append(f"Falha ao limpar {target}: {exc}")

    return {
        "success": not errors,
        "deleted_files": deleted_files,
        "deleted_directories": deleted_directories,
        "errors": errors,
    }
