"""Process-boundary for CNN FALTANDO v2 visual explanations.

The ODIN/Qt parent NEVER imports PyTorch from this module. All native Torch /
OpenMP / model-loading failures occur inside a disposable child interpreter,
not in the factory GUI process. Results are only diagnostic; NEVER routed to
classification, KNN memory, training, or XP key actions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import BoundedSemaphore

import cv2
import numpy as np

LIGHT_KEYS = ("large_reference", "large", "small_reference", "small")
MAPS = ("major", "minor")
MAX_INPUT_EDGE = 2048
MAX_OUTPUT_EDGE = 640
TIMEOUT_SECONDS = 45

# One CPU-heavy child at a time across all SIDE/TOP/MID Qt worker threads.
_PROBE_SLOT = BoundedSemaphore(1)


def _safe_inputs(crops: dict) -> dict:
    if not isinstance(crops, dict):
        raise ValueError("Pares da inspeção indisponíveis")
    out = {}
    for key in LIGHT_KEYS:
        image = crops.get(key)
        if image is None:
            continue
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or min(image.shape[:2]) < 12
            or max(image.shape[:2]) > MAX_INPUT_EDGE
        ):
            raise ValueError(f"Recorte AOI inválido: {key}")
        out[key] = np.ascontiguousarray(image)
    if not (
        {"large_reference", "large"}.issubset(out)
        or {"small_reference", "small"}.issubset(out)
    ):
        raise ValueError("Nenhum par completo gabarito/teste")
    return out


def _child(request_path: Path, result_path: Path) -> int:
    """Only executed with --child. No Qt application is imported/created."""
    import torch
    from src.core.neural.faltando_explainability import FaltandoExplainability

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    with np.load(request_path, allow_pickle=False) as archive:
        crops = {key: archive[key].copy() for key in archive.files}
    result = FaltandoExplainability().explain_epicenters(crops)

    output = {}
    metadata = {}
    for epicenter in MAPS:
        data = result.get(epicenter)
        if data is None:
            metadata[epicenter] = None
            continue
        if data.get("neural") is not True or len(data.get("images", [])) != 3:
            raise ValueError("Resposta não veio de features neurais verificadas")
        properties = {key: value for key, value in data.items() if key != "images"}
        metadata[epicenter] = properties
        for i, image in enumerate(data["images"]):
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("Mapa neural inválido")
            h, w = image.shape[:2]
            factor = min(1.0, MAX_OUTPUT_EDGE/max(h, w))
            if factor < 1:
                image = cv2.resize(
                    image,
                    (max(1, round(w*factor)), max(1, round(h*factor))),
                    interpolation=cv2.INTER_AREA,
                )
            output[f"{epicenter}_{i}"] = image
    # No pickle, no arbitrary deserialization, no hidden model state.
    np.savez_compressed(result_path, **output)
    result_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    return 0


def _read_response(result_path: Path) -> dict:
    meta_path = result_path.with_suffix(".json")
    if not (result_path.is_file() and meta_path.is_file()):
        raise RuntimeError("Processo CNN não produziu todos os mapas")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Metadados CNN inválidos")
    out = {}
    with np.load(result_path, allow_pickle=False) as packed:
        for region in MAPS:
            info = metadata.get(region)
            if info is None:
                out[region] = None
                continue
            if not isinstance(info, dict) or info.get("neural") is not True:
                raise ValueError("A saída não foi derivada da CNN")
            frames = []
            for n in range(3):
                key = f"{region}_{n}"
                if key not in packed:
                    raise ValueError(f"Mapa CNN ausente: {key}")
                frame = packed[key]
                if (
                    frame.dtype != np.uint8 or frame.ndim != 3
                    or frame.shape[2] != 3
                    or min(frame.shape[:2]) == 0
                    or max(frame.shape[:2]) > MAX_OUTPUT_EDGE
                ):
                    raise ValueError("Dimensão ou tipo de mapa CNN inválido")
                frames.append(frame.copy())
            out[region] = {**info, "images": tuple(frames)}
    return out


def explain_in_isolated_process(
    crops: dict, *, python_executable: str | None = None,
    timeout_seconds: int = TIMEOUT_SECONDS,
) -> dict:
    """Never crashes GUI if Torch process exits with a native fatal error."""
    if os.environ.get("VISIONX_DISABLE_NEURAL_MAPS", "") == "1":
        raise RuntimeError("Mapas neurais desativados por VISIONX_DISABLE_NEURAL_MAPS=1")
    safe = _safe_inputs(crops)
    if not 1 <= timeout_seconds <= 180:
        raise ValueError("Timeout fora do contrato")

    with _PROBE_SLOT:
        with tempfile.TemporaryDirectory(prefix="visionx_cnn_probe_") as temp:
            folder = Path(temp)
            request = folder / "request.npz"
            result = folder / "result.npz"
            np.savez_compressed(request, **safe)
            env = os.environ.copy()
            env.update({
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
            })
            command = [
                python_executable or sys.executable, "-m",
                "src.core.neural.faltando_explainability_runner", "--child",
                str(request), str(result),
            ]
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                proc = subprocess.run(
                    command, cwd=str(Path(__file__).resolve().parents[3]),
                    env=env, creationflags=flags, timeout=timeout_seconds,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Tempo limite de {timeout_seconds}s na sonda CNN; "
                    "o julgamento operacional continua inalterado"
                ) from exc
            if proc.returncode != 0:
                # Windows access violation 0xC0000005, OOM etc. are child-only.
                stderr = proc.stderr.decode("utf-8", errors="replace")
                last_line = stderr.strip().splitlines()[-1:] or ["Falha nativa sem mensagem"]
                raise RuntimeError(
                    f"Processo neural terminou (código {proc.returncode}): "
                    f"{last_line[0][:175]}"
                )
            return _read_response(result)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3 or args[0] != "--child":
        print("Uso: python -m src.core.neural.faltando_explainability_runner "
              "--child request.npz result.npz", file=sys.stderr)
        return 2
    try:
        return _child(Path(args[1]), Path(args[2]))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
