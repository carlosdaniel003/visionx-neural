"""Ativação local reversível da CNN V3 época 5, com consentimento do operador.

Apenas para o AOI/VisionX. Preserva a época 3 como rollback e NÃO habilita
0/OK somente porque a rede votou OK. Exige ambos os artefatos locais reais.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from src.core.neural.aoi_v3_epoch5_contract import (
    EPOCH5_POINTER, SNAPSHOT_SHA256, epoch5_pointer_payload,
    epoch5_pointer_enabled, validate_epoch5_snapshot,
)


def _preflight_model(root: Path) -> None:
    checkpoint, _ = validate_epoch5_snapshot(root)
    import torch
    from src.core.neural.aoi_quality_cnn_v3 import AOIQualityCNNV3
    model = AOIQualityCNNV3()
    model.load_state_dict(checkpoint["model_weights_not_production_state_dict"],
                          strict=True)
    model.eval()
    with torch.inference_mode():
        images = [torch.zeros((1, 3, 3, 96, 96)) for _ in range(4)]
        for index in range(3):
            mask = torch.zeros((1, 3))
            mask[0, index] = 1.
            logits, lights = model(*images, mask)
            if (
                tuple(logits.shape) != (1,)
                or tuple(lights.shape) != (1, 3)
                or not bool(torch.isfinite(logits).all().item())
            ):
                raise ValueError("CNN V3 época 5 falhou no preflight de inferência")


def activate(root: Path) -> dict:
    root = Path(root).resolve()
    _preflight_model(root)   # Erros não deixam nenhum ponteiro novo.
    target = root / EPOCH5_POINTER
    payload = epoch5_pointer_payload()
    if epoch5_pointer_enabled(root):
        return payload
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".tmp",
            prefix=".v3_epoch5_", dir=target.parent, delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        # Apenas 1 ponteiro novo é escrito. A época 3 segue guardada.
        os.replace(temp_path, target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return payload


def deactivate(root: Path) -> bool:
    root = Path(root).resolve()
    if not epoch5_pointer_enabled(root):
        return False
    target = root / EPOCH5_POINTER
    target.unlink()
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--activate", action="store_true")
    action.add_argument("--deactivate", action="store_true")
    args = parser.parse_args(argv)
    if args.activate:
        payload = activate(args.root)
        print(json.dumps({
            "status": "EPOCH5_ACTIVE",
            "checkpoint_sha256": SNAPSHOT_SHA256,
            "epoch": 5,
            "human_NG_training_pairs": "24/24",
            "validation_OK_false_NG": 2,
            "validation_NG_false_OK": 0,
            "production_approval": "OPERATOR_OVERRIDE_NOT_INDEPENDENT_VALIDATION",
            "automatic_OK_from_V3_only": False,
            "v2_runtime_reenabled": False,
            "pointer": str(Path(args.root).resolve() / EPOCH5_POINTER),
        }, ensure_ascii=False, indent=2))
    else:
        removed = deactivate(args.root)
        print(json.dumps({
            "status": "EPOCH5_DEACTIVATED",
            "pointer_removed": removed,
            "fallback": "OLDER_V3_EPOCH3_IF_ITS_OWN_POINTER_IS_VALID",
            "automatic_v2_fallback": False,
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
