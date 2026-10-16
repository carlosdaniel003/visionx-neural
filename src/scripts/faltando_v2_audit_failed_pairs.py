"""Read-only forensic contact sheets for FALTANDO V2 mistakes.

Creates PNG panels from the ORIGINAL AOI capture, reference/test crops, and
contrast-expanded absolute difference. Reads a local experiment report and its
immutable preparation manifest. Does not load CNN weights, retrain, approve
labels, or edit any input file. The holdout is ALREADY contaminated by past
debugging, so panels are diagnostics rather than blind evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from src.services.faltando_neural_qualification import QualificationStore, _read_image

REPORT_SCHEMA = "visionx.faltando_v2_guarded_light_repair_experiment.v1"
CATEGORIES = (
    ("historico", "new_historical"),
    ("treino", "new_train"),
    ("holdout", "new_holdout"),
)
TILE_W, TILE_H, GAP = 280, 176, 12
PANEL_W = 4 * TILE_W + 5 * GAP
ROW_H = TILE_H + 106


def _read_report(root: Path, report: Path) -> dict:
    allowed = (root / "reports/faltando_neural/archive_repairs").resolve()
    given = report if report.is_absolute() else root / report
    if given.is_symlink() or not given.is_file():
        raise ValueError("Relatório ausente ou symlink")
    resolved = given.resolve()
    if not resolved.is_relative_to(allowed) or resolved.name != "repair_report.json":
        raise ValueError("Relatório fora de archive_repairs")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or (
        data.get("schema") != REPORT_SCHEMA
        or data.get("candidate_only") is not True
        or data.get("promoted") is not False
        or data.get("production_approval_eligible") is not False
    ):
        raise ValueError("Relatório de reparo experimental inválido")
    for _, key in CATEGORIES:
        result = data.get(key)
        if not isinstance(result, dict) or not isinstance(result.get("cases"), list):
            raise ValueError("Casos não encontrados no relatório: " + key)
    return data


def _collect_failures(report: dict, store: QualificationStore) -> tuple[dict, dict]:
    """Only real misclassifications, no fabricated class labels."""
    groups = {}
    omitted = Counter()
    for title, key in CATEGORIES:
        records = []
        prefix = "single:" if key == "new_historical" else "archive_provisional:"
        for case in report[key]["cases"]:
            if case.get("correct") is True:
                continue
            name = case.get("event_id_or_group_hint")
            if not isinstance(name, str) or not name.startswith(prefix):
                omitted[title] += 1
                continue
            source = name[len(prefix):]
            if source not in store.by_path:
                # Online/historical records can be outside this archive.
                omitted[title] += 1
                continue
            if case.get("label") not in ("NG", "OK") or not (
                isinstance(case.get("ng_score"), (int, float))
                and np.isfinite(case["ng_score"])
                and 0. <= case["ng_score"] <= 1.
            ):
                raise ValueError("Rótulo/score inválido do caso " + source)
            light, views = store.by_path[source]["lighting_mode"], case.get("per_light")
            if not isinstance(views, dict) or light not in views:
                raise ValueError("Iluminação do relatório não corresponde à fonte " + source)
            records.append({
                "source": source,
                "light": light,
                "label_from_report": case["label"],
                "label_from_folder": store.expected_labels[source],
                "prediction": case["predicted"],
                "ng_score": round(float(case["ng_score"]), 6),
                "category": title,
                "source_sha256": store.by_path[source]["source_sha256"],
            })
        groups[title] = sorted(records, key=lambda e: (
            0 if e["label_from_report"] == "OK" else 1,
            e["light"], e["source"],
        ))
    return groups, dict(omitted)


def _fit(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.ndim != 3 or min(frame.shape[:2]) < 1:
        raise ValueError("Imagem ilegível")
    h, w = frame.shape[:2]
    scale = min(TILE_W / w, TILE_H / h)
    width, height = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    reduced = cv2.resize(frame, (width, height), interpolation=interp)
    tile = np.full((TILE_H, TILE_W, 3), 30, dtype=np.uint8)
    y, x = (TILE_H - height) // 2, (TILE_W - width) // 2
    tile[y:y+height, x:x+width] = reduced
    return tile


def _difference(reference: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, bool]:
    """Difference is magnified visually, never a model input or ground truth."""
    resized = reference.shape != test.shape
    if resized:
        test = cv2.resize(
            test, (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    raw = cv2.absdiff(reference, test).astype(np.uint16)
    return np.minimum(raw * 3, 255).astype(np.uint8), resized


def _draw_page(rows: list[dict], store: QualificationStore, page: int,
               output: Path, *, verify_source: bool = True) -> None:
    canvas = np.full(
        (42 + ROW_H * len(rows) + 14, PANEL_W, 3), 248, dtype=np.uint8,
    )
    cv2.putText(canvas, f"FALTANDO V2 - AUDITORIA VISUAL / PAGINA {page}",
                (18, 27), cv2.FONT_HERSHEY_SIMPLEX, .64, (25, 25, 25), 2,
                cv2.LINE_AA)
    for i, item in enumerate(rows):
        source = item["source"]
        if verify_source:
            store._validate_source(source)
        origin, ref_path, test_path = store.files[source]
        images = [_read_image(p) for p in (origin, ref_path, test_path)]
        difference, scaled = _difference(images[1], images[2])
        images.append(difference)
        offset = 42 + i * ROW_H
        header = (
            f"{item['category'].upper()} | {item['light']} | "
            f"rotulo {item['label_from_report']} | CNN {item['prediction']} "
            f"| NG={item['ng_score']:.6f}"
        )
        cv2.putText(canvas, header, (18, offset+21),
                    cv2.FONT_HERSHEY_SIMPLEX, .52, (30, 30, 30), 1,
                    cv2.LINE_AA)
        filename = Path(source).name
        cv2.putText(canvas, filename[:110], (18, offset+43),
                    cv2.FONT_HERSHEY_SIMPLEX, .46, (75, 75, 75), 1,
                    cv2.LINE_AA)
        names = ("CAPTURA AOI", "GABARITO", "TESTE", "DIFERENCA x3" +
                 (" (REDIM.)" if scaled else ""))
        for index, (frame, label) in enumerate(zip(images, names)):
            x = GAP + index * (TILE_W + GAP)
            canvas[offset+54:offset+54+TILE_H, x:x+TILE_W] = _fit(frame)
            cv2.putText(canvas, label, (x+5, offset+53+TILE_H+20),
                        cv2.FONT_HERSHEY_SIMPLEX, .46, (30, 30, 30), 1,
                        cv2.LINE_AA)
        cv2.line(canvas, (12, offset + ROW_H-4),
                 (PANEL_W-12, offset + ROW_H-4), (210, 210, 210), 1)
    valid, encoded = cv2.imencode(".png", canvas)
    if not valid:
        raise ValueError("Falha ao criar prancha de falhas")
    output.write_bytes(encoded.tobytes())


def audit_failed_pairs(
    root: Path, manifest: Path, report: Path, *, per_page: int = 4,
) -> dict:
    """Create only forensic PNGs and a summary in a unique reports/ folder."""
    root = Path(root).resolve()
    if type(per_page) is not int or not 1 <= per_page <= 8:
        raise ValueError("per-page deve ser de 1 a 8")
    manifest = Path(manifest)
    manifest = manifest if manifest.is_absolute() else root / manifest
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("Manifesto ausente ou symlink")
    data = _read_report(root, Path(report))
    digest = sha256(manifest.read_bytes()).hexdigest()
    if digest != data.get("manifest_sha256"):
        raise ValueError("Manifesto mudou desde o treinamento")
    store = QualificationStore(manifest)
    if store.root != root:
        raise ValueError("Raiz do manifesto diferente da raiz informada")
    groups, omitted = _collect_failures(data, store)
    # Validate all selected images BEFORE writing a single page.
    selected = {e["source"] for items in groups.values() for e in items}
    for source in selected:
        store._validate_source(source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = root / "reports/faltando_neural/failed_pair_audits" / (
        stamp + "_" + uuid4().hex[:6]
    )
    out.mkdir(parents=True, exist_ok=False)
    files = []
    for group, rows in groups.items():
        for start in range(0, len(rows), per_page):
            part = start // per_page + 1
            png = out / f"{group}_pagina_{part:02d}.png"
            _draw_page(rows[start:start+per_page], store, part, png)
            files.append(png.relative_to(root).as_posix())
    index = {
        "schema": "visionx.faltando_v2_failed_pair_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "training_performed": False,
        "labels_changed": False,
        "champion_unchanged": True,
        "checkpoint_promoted": False,
        "manifest_sha256": digest,
        "source_candidate_sha256": data.get("candidate_sha256"),
        "same_holdout_previously_exposed": True,
        "archive_labels_individually_unverified": True,
        "images_may_contain_sensitive_AOI_information": True,
        "note": (
            "Inspecao manual dos casos falhos; diferenca x3 e apenas visual. "
            "Nao usar porcentagens deste holdout como teste independente."
        ),
        "counts": {key: len(rows) for key, rows in groups.items()},
        "omitted_outside_archive": omitted,
        "unique_source_count": len(selected),
        "pages": files,
        "cases": groups,
    }
    path = out / "audit_index.json"
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("CASOS PARA AUDITORIA:", index["counts"], flush=True)
    print("Fontes distintas:", len(selected), flush=True)
    for row in files:
        print("Prancha:", root / row, flush=True)
    print("Resumo:", path, flush=True)
    return index


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Auditoria de erros CNN FALTANDO V2 sem treinamento"
    )
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--per-page", type=int, default=4)
    args = p.parse_args(argv)
    audit_failed_pairs(
        args.root, args.manifest, args.report, per_page=args.per_page,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
