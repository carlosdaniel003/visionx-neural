"""ABBA read-only AOI OCR trial: baseline Tesseract vs OMP_THREAD_LIMIT=1.

Both arms execute the *unchanged production* ScreenMonitor parser and directed
Parts recovery on original AOI frames. Separate worker processes avoid changing
the ODIN/torch environment. No automatic promotion or production code patch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

SCHEMA = "visionx.aoi_ocr_thread_trial.v1"
ARMS = ("default", "omp1", "omp1", "default")


def worker_measure(job: dict, mode: str) -> dict:
    """Invoked only in the child process with an arm-specific environment."""
    import cv2
    import numpy as np

    from src.services.aoi_ocr_benchmark import (
        VARIANTS, inspect_variant, locate_bars,
    )
    from src.services import screen_monitor
    if not screen_monitor.HAS_TESSERACT:
        raise RuntimeError("Tesseract nao instalado ou nao encontrado no PATH")
    rows = []
    for source in job["cases"]:
        path = Path(source["path"])
        raw = path.read_bytes()
        from hashlib import sha256
        if sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("Imagem mudou durante benchmark: " + str(path))
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("Falha ao decodificar: " + str(path))
        monitor = screen_monitor.ScreenMonitor()
        bars = locate_bars(monitor, frame)
        runs = []
        for _ in range(int(job["repeats"])):
            result = inspect_variant(
                monitor, frame, *bars, VARIANTS[0], screen_monitor.pytesseract,
            )
            from src.services.aoi_ocr_fields import looks_like_component_reference
            result["parts_syntax_valid"] = looks_like_component_reference(
                result["fields"]["parts"]
            )
            runs.append(result)
        rows.append({"sha256": source["sha256"], "measurements": runs})
    return {
        "arm": mode,
        "omp_thread_limit": os.environ.get("OMP_THREAD_LIMIT"),
        "cases": rows,
    }


def worker_command(job_path: Path, result_path: Path, mode: str):
    return [
        sys.executable, "-m", "src.services.aoi_ocr_thread_trial",
        "--_worker", "--_job", str(job_path),
        "--_response", str(result_path), "--_mode", mode,
    ]


def run_isolated_arms(
    job: dict, *, worker_runner=None, sequence=ARMS, timeout=900,
) -> list[dict]:
    """Executes one arm at a time; never mutates os.environ in this process."""
    runner = worker_runner or subprocess.run
    outputs = []
    with tempfile.TemporaryDirectory(prefix="odin_ocr_ab_") as directory:
        temp = Path(directory)
        job_path = temp / "job.json"
        job_path.write_text(
            json.dumps(job, ensure_ascii=False), encoding="utf-8"
        )
        for index, mode in enumerate(sequence):
            if mode not in ("default", "omp1"):
                raise ValueError("OCR mode invalido")
            response = temp / f"arm_{index}.json"
            env = dict(os.environ)
            # Replicate the standard executable environment in the control;
            # OMP=1 is only supplied to the separate candidate subprocess.
            env.pop("OMP_THREAD_LIMIT", None)
            if mode == "omp1":
                env["OMP_THREAD_LIMIT"] = "1"
            proc = runner(
                worker_command(job_path, response, mode),
                env=env, check=False, capture_output=True,
                text=True, timeout=timeout,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Braço OCR {mode} falhou (exit={proc.returncode}): "
                    + (proc.stderr or proc.stdout or "")[-1200:]
                )
            result = json.loads(response.read_text(encoding="utf-8"))
            if result.get("arm") != mode:
                raise ValueError("Resposta do worker com modo incorreto")
            if mode == "omp1" and result.get("omp_thread_limit") != "1":
                raise ValueError("OMP_THREAD_LIMIT=1 nao confirmado")
            if len(result.get("cases", ())) != len(job["cases"]):
                raise ValueError("Resposta incompleta do benchmark")
            outputs.append(result)
    return outputs


def summarize_trial(job: dict, outputs: list[dict]) -> dict:
    if tuple(run["arm"] for run in outputs) != ARMS:
        raise ValueError("Comparacao deve usar ordem ABBA")
    cases = []
    measured = {"default": [], "omp1": []}
    disagreed = 0
    baseline_unstable = 0
    mismatched_human = {"default": set(), "omp1": set()}
    invalid_parts = {"default": set(), "omp1": set()}
    for i, source in enumerate(job["cases"]):
        by_mode = {"default": [], "omp1": []}
        for result in outputs:
            row = result["cases"][i]
            if row["sha256"] != source["sha256"]:
                raise ValueError("SHA da imagem nao corresponde ao manifesto")
            by_mode[result["arm"]].extend(row["measurements"])
        base_fields = [run["fields"] for run in by_mode["default"]]
        candidate_fields = [run["fields"] for run in by_mode["omp1"]]
        stable_baseline = all(x == base_fields[0] for x in base_fields)
        baseline_unstable += not stable_baseline
        different = not stable_baseline or any(
            x != base_fields[0] for x in candidate_fields
        )
        disagreed += different
        for mode, runs in by_mode.items():
            measured[mode].extend(float(r["elapsed_ms"]) for r in runs)
            if any(not r["parts_syntax_valid"] for r in runs):
                invalid_parts[mode].add(i)
            expected = source.get("expected")
            if expected is not None and any(
                r["fields"] != expected for r in runs
            ):
                mismatched_human[mode].add(i)
        cases.append({
            "sha256": source["sha256"],
            "path": source["path"],
            "expected": source.get("expected"),
            "baseline_stable": stable_baseline,
            "different_fields": different,
            "default": by_mode["default"],
            "omp1": by_mode["omp1"],
        })
    base = statistics.median(measured["default"])
    fast = statistics.median(measured["omp1"])
    count = len(job["cases"])
    human_total = sum(r.get("expected") is not None for r in job["cases"])
    return {
        "schema": SCHEMA,
        "mode": "offline_read_only_comparison",
        "no_production_change": True,
        "auto_promotion": False,
        "repeats_per_arm": job["repeats"] * 2,
        "sequence": list(ARMS),
        "case_count": count,
        "human_labels_supplied": human_total,
        "baseline_unstable_cases": baseline_unstable,
        "different_from_baseline_cases": disagreed,
        "default_median_ms": round(base, 2),
        "omp1_median_ms": round(fast, 2),
        "speedup_pct": round(100 * (base - fast) / base, 2) if base else None,
        "invalid_parts_cases": {
            k: len(v) for k, v in invalid_parts.items()
        },
        "human_mismatches": {
            k: len(v) for k, v in mismatched_human.items()
        },
        "eligible_for_automatic_activation": False,
        "limitations": (
            "A/B isola a variavel OMP apenas nos subprocessos de teste. "
            "Igualdade com baseline nao prova acerto. Rótulos humanos devem "
            "conferir Board/Parts/Value/Category e capturas independentes. "
            "Sem ativacao automatica; o comportamento sob carga do ODIN pode diferir."
        ),
        "cases": cases,
    }


def make_job(sources, repeats: int) -> dict:
    from hashlib import sha256
    if not 1 <= repeats <= 8:
        raise ValueError("repeats deve ser 1..8")
    rows = []
    for path, expected in sources:
        rows.append({
            "path": str(path),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "expected": expected,
        })
    if not rows:
        raise ValueError("Nenhuma captura")
    return {"repeats": repeats, "cases": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_job", help=argparse.SUPPRESS)
    parser.add_argument("--_response", help=argparse.SUPPRESS)
    parser.add_argument("--_mode", choices=("default", "omp1"), help=argparse.SUPPRESS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--images", nargs="+", help="Capturas originais AOI")
    group.add_argument("--manifest", help="JSON com expected humano")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", help="Novo arquivo JSON, sem sobrescrita")
    args = parser.parse_args(argv)

    if args._worker:
        if args.images or args.manifest or args.output or not (
            args._job and args._response and args._mode
        ):
            parser.error("Argumentos de worker invalidos")
        job = json.loads(Path(args._job).read_text(encoding="utf-8"))
        result = worker_measure(job, args._mode)
        Path(args._response).write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )
        return 0

    if (args.images is None) == (args.manifest is None) or not args.output:
        parser.error("Informe --images ou --manifest e --output")
    from src.services.aoi_ocr_benchmark import _image_paths
    sources = _image_paths(args)
    out = Path(args.output).resolve()
    if out in {path for path, _ in sources}:
        raise ValueError("Arquivo de saida nao pode substituir captura")
    if out.exists():
        raise FileExistsError("Relatorio ja existe: " + str(out))
    job = make_job(sources, args.repeats)
    results = run_isolated_arms(job)
    report = summarize_trial(job, results)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("OCR ABBA concluido (somente leitura / sem ativacao).")
    print(
        f"Casos={report['case_count']} default={report['default_median_ms']} ms "
        f"OMP1={report['omp1_median_ms']} ms; "
        f"speedup={report['speedup_pct']}%"
    )
    print(
        f"Divergencias={report['different_from_baseline_cases']}; "
        f"Parts invalidos={report['invalid_parts_cases']}; "
        f"rótulos humanos={report['human_labels_supplied']}"
    )
    print("Resultado: " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
