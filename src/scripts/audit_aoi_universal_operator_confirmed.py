"""Offline replay of ALL 15 operator-verified AOI PNGs against a candidate.

Uses prepared AOI pairs when source SHA matches; extracts missing pairs from
archived screenshots with the same offline AOI extractor used by preparation.
No weights are trained, no active pointer is loaded or changed, no XP command.
Not a blind/independent test: some reviewed captures were in train/validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from uuid import uuid4

import cv2
import numpy as np
import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.scripts.train_aoi_universal_quality_offline import SCHEMA, REPORT_SCHEMA
from src.scripts.train_faltando_cnn_v2_online import _assess
from src.services.faltando_neural_dataset import AOIPairExtractor
from src.services.odin_operator_confirmed_labels import verified_operator_labels

SCHEMA_REPLAY = "visionx.aoi_universal_operator_confirmed_replay.v1"
LIGHT_RE = re.compile(r"_(SIDE|TOP|MID)(?:_\d+)?$", re.I)


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Relatório ou manifesto ausente/symlink: " + str(path))
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Relatório/manifesto inválido")
    return obj


def _original_manifest(root: Path, relative_or_absolute: Path, digest: str) -> tuple[Path, dict]:
    path = relative_or_absolute if relative_or_absolute.is_absolute() else root / relative_or_absolute
    path = path.resolve()
    if (not path.is_relative_to(root / "reports/faltando_neural")
            or path.name != "manifest.json"):
        raise ValueError("Manifesto fora da preparação AOI")
    doc = _read_json(path)
    if (doc.get("schema") != "visionx.faltando_neural_preparation.v1"
            or Path(doc.get("root", "")).resolve() != root
            or sha256(path.read_bytes()).hexdigest() != digest):
        raise ValueError("Manifesto mudou ou pertence a outra estação")
    return path, doc


def _candidate(root: Path, report_path: Path) -> tuple[dict, Path, dict]:
    report_path = report_path if report_path.is_absolute() else root / report_path
    report_path = report_path.resolve()
    experiments = (root / "reports/aoi_universal_quality/experiments").resolve()
    if (report_path.name != "training_report.json" or report_path.is_symlink()
            or report_path.parent.parent != experiments):
        raise ValueError("Relatório não pertence aos candidatos universais")
    report = _read_json(report_path)
    if (report.get("schema") != REPORT_SCHEMA
            or report.get("candidate_only") is not True
            or report.get("promoted") is not False
            or report.get("automatic_ok_authorized") is not False):
        raise ValueError("Relatório fora do contrato experimental")
    rel = report.get("source_candidate_relative_path")
    if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ValueError("Caminho de checkpoint inseguro")
    path = root / rel
    if (path.is_symlink() or not path.is_file()
            or path.resolve().parent != report_path.parent
            or path.name != "aoi_general_quality_offline_candidate.pt"):
        raise ValueError("Checkpoint fora da pasta experimental")
    if sha256(path.read_bytes()).hexdigest() != report.get("candidate_sha256"):
        raise ValueError("Hash do candidato divergente")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict)
            or checkpoint.get("schema") != SCHEMA
            or checkpoint.get("architecture_schema") != MODEL_SCHEMA_V2
            or checkpoint.get("candidate_only") is not True
            or checkpoint.get("production_approved") is not False
            or checkpoint.get("automatic_ok_authorized") is not False
            or checkpoint.get("source_manifest_sha256") !=
                report.get("provenance", {}).get("manifest_sha256")):
        raise ValueError("Checkpoint incompatível com replay seguro")
    return report, path, checkpoint


def _pair_file(run_dir: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts):
        raise ValueError("Recorte derivado inválido")
    candidate = run_dir / relative
    if (candidate.is_symlink() or not candidate.is_file()
            or not candidate.resolve().is_relative_to((run_dir / "pairs").resolve())):
        raise ValueError("Par preparado ausente/fora da pasta de pares")
    return candidate


def _archive_frame(path: Path) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR,
    )
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 12:
        raise ValueError("Imagem AOI original inválida: " + path.name)
    return image


def _save_derived(path: Path, frame: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError("Falha ao codificar recorte diagnóstico")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _lighting(source: str) -> str:
    m = LIGHT_RE.search(Path(source).stem)
    return m.group(1).upper() if m else "SIDE"


def _build_records(root: Path, manifest_path: Path, manifest: dict,
                   confirmed: dict[str, dict], destination: Path,
                   *, extractor=None) -> tuple[list[dict], dict]:
    samples = {
        row["source_path"]: row
        for row in manifest.get("samples", [])
    }
    if len(samples) != len(manifest.get("samples", [])):
        raise ValueError("Manifesto possui entradas duplicadas")
    prepared = 0
    extracted = 0
    records = []
    for source, human in sorted(confirmed.items()):
        original = root / source
        if sha256(original.read_bytes()).hexdigest() != human["source_sha256"]:
            raise ValueError("Origem mudou durante replay: " + source)
        sample = samples.get(source)
        if sample:
            if (sample.get("source_sha256") != human["source_sha256"]
                    or sample.get("status") != "EXTRACTED_PENDING_REVIEW"
                    or sample.get("lighting_mode") not in ("SIDE", "TOP", "MID")):
                raise ValueError("Par preparado incompatível: " + source)
            ref = _pair_file(manifest_path.parent, sample["reference_path"])
            test = _pair_file(manifest_path.parent, sample["test_path"])
            light = sample["lighting_mode"]
            prepared += 1
            provenance = "EXISTING_PREPARED_PAIR"
        else:
            if extractor is None:
                extractor = AOIPairExtractor()
            try:
                ref_frame, test_frame, _ = extractor(_archive_frame(original))
            except Exception as exc:
                raise ValueError("Falha de extração do par AOI para " + source) from exc
            light = _lighting(source)
            base = destination / "reextracted_pairs" / human["source_sha256"]
            ref, test = base / "reference.png", base / "test.png"
            _save_derived(ref, ref_frame)
            _save_derived(test, test_frame)
            extracted += 1
            provenance = "REEXTRACTED_FROM_VERIFIED_ARCHIVE_PNG"
        records.append({
            "id": source,
            "label": human["quality_label"],
            "views": {light: {"reference": ref, "test": test}},
            "source_sha256": human["source_sha256"],
            "defect_type_operator": human["physical_type"],
            "pair_provenance": provenance,
        })
    if len(records) != 15 or prepared != 6 or extracted != 9:
        raise ValueError(
            f"Esperado 6 pares preparados e 9 extraídos; encontrados {prepared}/{extracted}"
        )
    return records, {"pairs_prepared": prepared, "pairs_reextracted": extracted}


def replay_operator_confirmed(
    root: Path, training_report: Path, manifest: Path,
    *, extractor=None, assessment=_assess,
) -> dict:
    root = Path(root).resolve()
    confirmed = verified_operator_labels(root)  # Pins all 15 input PNG hashes.
    report, checkpoint_path, checkpoint = _candidate(root, Path(training_report))
    manifest_path, manifest_doc = _original_manifest(
        root, Path(manifest), report["provenance"]["manifest_sha256"],
    )
    pointer = root / "reports/neural_online/live_active.json"
    before = sha256(pointer.read_bytes()).hexdigest() if pointer.exists() else None
    out = root / "reports/aoi_universal_quality/operator_replay" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + "_" + uuid4().hex[:6]
    )
    out.mkdir(parents=True, exist_ok=False)
    rows, collection = _build_records(
        root, manifest_path, manifest_doc, confirmed, out, extractor=extractor,
    )
    size = checkpoint.get("image_size")
    focus = checkpoint.get("focus_fraction")
    if (type(size) is not int or size < 64 or size > 256
            or type(focus) not in (float, int) or not .1 < focus < 1.):
        raise ValueError("Pré-processamento do candidato inválido")
    model = FaltandoCNNV2()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    assessed = assessment(model, rows, size, focus)
    cases = assessed.get("cases", [])
    if len(cases) != 15:
        raise ValueError("Replay de 15 decisões não foi concluído")
    locations = {}
    for fold in ("train", "validation", "test"):
        for event in report.get("split", {}).get("folds", {}).get(fold, {}).get("source_ids", []):
            locations[event.removeprefix("universal_provisional:")] = fold
    results = []
    for row, outcome in zip(rows, cases):
        if (outcome["event_id_or_group_hint"] != row["id"]
                or outcome["label"] != row["label"]):
            raise ValueError("Predição fora da ordem de evidência humana")
        results.append({
            "source_path": row["id"],
            "source_sha256": row["source_sha256"],
            "operator_label": row["label"],
            "operator_physical_type": row["defect_type_operator"],
            "source_fold": locations.get(row["id"], "OUTSIDE_TRAINING_MANIFEST"),
            "pair_provenance": row["pair_provenance"],
            "candidate_prediction": outcome["predicted"],
            "ng_score": outcome["ng_score"],
            "correct": outcome["correct"],
        })
    after = sha256(pointer.read_bytes()).hexdigest() if pointer.exists() else None
    if before != after:
        raise ValueError("Ponteiro ativo mudou durante auditoria")
    if sha256(checkpoint_path.read_bytes()).hexdigest() != report["candidate_sha256"]:
        raise ValueError("Candidato mudou durante replay")
    wrong_ng = [x for x in results if x["operator_label"] == "NG" and not x["correct"]]
    wrong_ok = [x for x in results if x["operator_label"] == "OK" and not x["correct"]]
    audit = {
        "schema": SCHEMA_REPLAY,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": report["candidate_sha256"],
        "checkpoint_schema": checkpoint["schema"],
        "source_manifest_sha256": report["provenance"]["manifest_sha256"],
        "active_pointer_sha256_unchanged": before == after,
        "training_performed": False,
        "active_checkpoint_modified": False,
        "source_pngs_modified": False,
        "xp_commands_sent": 0,
        "production_approved": False,
        "automatic_ok_authorized": False,
        "source_images": 15,
        "operator_confirmed_ng": 12,
        "operator_confirmed_ok": 3,
        "confirmed_ng_false_ok": len(wrong_ng),
        "confirmed_ok_false_ng": len(wrong_ok),
        "all_confirmed_ng_detected": len(wrong_ng) == 0,
        "all_confirmed_ok_accepted": len(wrong_ok) == 0,
        "production_ready": False,
        "physically_independent_validation": False,
        "note": "Replay diagnóstico: parte dos casos foi vista pelo treino/validação. NÃO usar como autorização de produção.",
        "collection": collection,
        "cases": results,
    }
    output = out / "operator_confirmed_replay.json"
    output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    return {"report": str(output), "confirmed_ng_false_ok": len(wrong_ng),
            "confirmed_ok_false_ng": len(wrong_ok),
            "ready_for_production": False, "cases_checked": len(results),
            "pairs": collection}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay offline dos 15 casos humanos contra CNN universal")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(
        replay_operator_confirmed(args.root, args.training_report, args.manifest),
        indent=2, ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
