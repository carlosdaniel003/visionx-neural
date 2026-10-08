"""Preparação offline do acervo FALTANDO para futuro treinamento neural.

Extrai o par AOI completo pelo mesmo ScreenMonitor da produção. Os rótulos
humanos e vínculos multilight são provisórios até qualificação independente.
Não carrega KNN/CNN, não altera PNGs, não treina nem altera produção.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import re
from typing import Callable, Any

import cv2
import numpy as np

from src.services.startup_regression.archive_inventory import inventory_archives

SCHEMA = "visionx.faltando_neural_preparation.v1"
LIGHTS = ("SIDE", "TOP", "MID")
LIGHT_PATTERN = re.compile(r"_(SIDE|TOP|MID)(?:_([0-9]+))?$", re.IGNORECASE)


def _read_png(path: Path) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None or image.size == 0 or image.ndim != 3:
        raise ValueError("PNG não decodificado em imagem BGR válida")
    return image


class AOIPairExtractor:
    """Reusa o algoritmo de extração da captura real, sem executar IA."""

    def __init__(self, monitor=None):
        if monitor is None:
            # Lazy import: não exige iniciar QApplication ou servidor AOI.
            from src.services.screen_monitor import ScreenMonitor
            monitor = ScreenMonitor()
        self.monitor = monitor
        self.monitor._replay_no_debug = True

    def __call__(self, frame: np.ndarray):
        captured = []
        errors = []

        def on_layout(reference, test, info):
            captured.append((reference.copy(), test.copy(), dict(info or {})))

        def on_log(message):
            if "SUCESSO" not in str(message):
                errors.append(str(message))

        self.monitor.layout_detected.connect(on_layout)
        self.monitor.log_updated.connect(on_log)
        try:
            with redirect_stdout(StringIO()):
                self.monitor.process_external_image(frame)
        finally:
            self.monitor.layout_detected.disconnect(on_layout)
            self.monitor.log_updated.disconnect(on_log)
            self.monitor.last_capture_frame = None

        if len(captured) != 1:
            raise ValueError(
                "AOI não forneceu um par gabarito/teste único: "
                + "; ".join(errors[:2])
            )
        reference, test, info = captured[0]
        if not all(
            isinstance(image, np.ndarray) and image.ndim == 3
            and image.size > 0 and min(image.shape[:2]) >= 12
            for image in (reference, test)
        ):
            raise ValueError("Recortes AOI incompletos ou excessivamente pequenos")
        return reference, test, info


def _write_png(path: Path, image: np.ndarray) -> None:
    valid, encoded = cv2.imencode(".png", image)
    if not valid:
        raise ValueError("Falha ao codificar recorte PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Uma execução sempre usa pasta exclusiva, evitando sobrescrever treino antigo.
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


def _group_hint(relative: str) -> str | None:
    stem = Path(relative).stem
    match = LIGHT_PATTERN.search(stem)
    if not match:
        return None
    # Não vincular versões _2 à peça original, nem unir OK com NG.
    return stem[:match.start()] + ("_COPY_" + match.group(2) if match.group(2) else "")


def _summary_text(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "ODIN — PREPARAÇÃO OFFLINE FALTANDO",
        "Sem treinamento, KNN, mudança de rótulos ou decisão automática",
        "Resultado de arquivo: extração de pares; NÃO é dataset aprovado",
        "",
        f"Total FALTANDO: {summary['total']}",
        f"Pares extraídos: {summary['extracted']}",
        f"Falhas/invalidos: {summary['unusable']}",
        f"NG: {summary['by_label'].get('NG', 0)}",
        f"OK: {summary['by_label'].get('OK', 0)}",
        f"Grupos multilight com manifesto válido: {summary['verified_multilight_events']}",
        f"Trincas por nome NÃO confirmadas: {summary['name_only_triplets']}",
        f"Elegíveis automaticamente para treino: {summary['training_ready']}",
        "",
        "PENDÊNCIAS: validar rótulos humanos, pares, independência visual e",
        "vínculo de evento antes de preparar splits ou treinar CNN.",
        "",
    ]
    for case in report["samples"]:
        if case["status"] != "EXTRACTED_PENDING_REVIEW":
            lines.append(
                f"{case['status']}: {case['source_path']} — {case.get('error','')}"
            )
    return "\n".join(lines) + "\n"


def prepare_faltando(
    root: Path,
    output_base: Path | None = None,
    *,
    extractor: Callable[[np.ndarray], tuple] | None = None,
    inventory: dict | None = None,
) -> tuple[dict[str, Any], Path]:
    """Gera pares derivados e manifesto, nunca altera o acervo de origem.

    Mantém training_ready=False até qualificação humana explícita. Um sufixo
    SIDE/TOP/MID é apenas pista; só manifesto válido confirma event_id.
    """
    root = Path(root).expanduser().resolve()
    reports_root = (root / "reports").resolve()
    output_base = (
        Path(output_base).expanduser().resolve()
        if output_base is not None else reports_root / "faltando_neural"
    )
    # Nunca permitir recortes ou manifestos em dataset/ ou arquivos produtivos.
    if output_base == reports_root or reports_root not in output_base.parents:
        raise ValueError("Saída deve ficar sob reports/ em pasta dedicada")

    source_inventory = inventory if inventory is not None else inventory_archives(root)
    selected = sorted(
        (
            item for item in source_inventory.get("images", [])
            if item.get("category_hint") == "FALTANDO"
            and item.get("expected_label") in ("OK", "NG")
        ),
        key=lambda item: item["path"],
    )
    now = datetime.now(timezone.utc)
    run_dir = output_base / ("run_" + now.strftime("%Y%m%dT%H%M%S_%fZ"))
    run_dir.mkdir(parents=True, exist_ok=False)
    samples: list[dict[str, Any]] = []
    proposed: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    verified: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for item in selected:
        relative = item["path"]
        source = root / relative
        label = item["expected_label"]
        lighting = item["lighting_mode"]
        sample = {
            "source_path": relative,
            "source_sha256": item.get("file_sha256", ""),
            "expected_label_from_archive": label,
            "category_hint": "FALTANDO",
            "lighting_mode": lighting,
            "lighting_source": item.get("lighting_source"),
            "event_id": item.get("event_id"),
            "status": "INVALID_SOURCE",
            "reference_path": None,
            "test_path": None,
            "ocr_unverified": True,
            "training_ready": False,
            "review_required": [
                "CONFIRM_OPERATOR_LABEL",
                "VERIFY_PAIR_AND_COMPONENT",
                "VERIFY_INDEPENDENT_OBSERVATION",
            ],
        }
        samples.append(sample)
        if item.get("status") != "VALID_PNG":
            sample["error"] = "Inventário classificou o PNG como inválido"
            continue
        # Arquivos em symlinks ou fora da árvore pública são rejeitados.
        if source.is_symlink() or not source.resolve().is_relative_to(root / "public"):
            sample["error"] = "Origem fora de public/ ou via symlink"
            continue

        if item["lighting_source"] == "LEGACY_DEFAULT":
            sample["review_required"].append("ASSUMED_LEGACY_SIDE")
        hint = _group_hint(relative)
        if hint:
            proposed[(label, hint)][lighting].append(relative)
            sample["review_required"].append("CHECK_MULTILIGHT_EVENT_ID")

        links = item.get("manifest_links", []) or []
        if len(links) == 1 and item.get("event_id"):
            verified[(label, item["event_id"])][lighting].append(relative)

        try:
            frame = _read_png(source)
            if extractor is None:
                extractor = AOIPairExtractor()
            reference, test, info = extractor(frame)
            if not isinstance(info, dict):
                raise ValueError("Metadados AOI inválidos")
            if any(
                not isinstance(image, np.ndarray) or image.size == 0
                or image.ndim != 3 or min(image.shape[:2]) < 12
                for image in (reference, test)
            ):
                raise ValueError("Par incompleto ou inválido")
            # ID único por imagem de origem; evita unir arquivos com
            # nomes, classes ou datas coincidentes.
            identity = label.lower() + "_" + str(item["file_sha256"])
            relative_base = Path("pairs") / identity
            target_reference = run_dir / relative_base / "reference.png"
            target_test = run_dir / relative_base / "test.png"
            _write_png(target_reference, reference)
            _write_png(target_test, test)
            sample.update({
                "status": "EXTRACTED_PENDING_REVIEW",
                "reference_path": (relative_base / "reference.png").as_posix(),
                "test_path": (relative_base / "test.png").as_posix(),
                "reference_size": [int(reference.shape[1]), int(reference.shape[0])],
                "test_size": [int(test.shape[1]), int(test.shape[0])],
                "ocr_observed": {
                    key: str(info.get(key, "") or "")
                    for key in ("board", "parts", "value", "category")
                },
            })
            if (str(info.get("category", "") or "").strip().upper()
                    not in ("", "FALTANDO")):
                sample["review_required"].append("OCR_CATEGORY_MISMATCH")
        except (OSError, ValueError, cv2.error, RuntimeError) as exc:
            sample["status"] = "EXTRACTION_FAILED"
            sample["error"] = f"{type(exc).__name__}: {exc}"

    def groups(data, *, confirmed):
        records = []
        for (label, key), modes in sorted(data.items()):
            if set(modes) != set(LIGHTS) or any(len(modes[m]) != 1 for m in LIGHTS):
                continue
            records.append({
                "label": label,
                "id": key,
                "paths": {mode: modes[mode][0] for mode in LIGHTS},
                "status": (
                    "LINKED_BY_VERIFIED_MANIFEST" if confirmed
                    else "NAME_ONLY_NEEDS_HUMAN_QUALIFICATION"
                ),
            })
        return records

    verified_groups = groups(verified, confirmed=True)
    proposals = groups(proposed, confirmed=False)
    labels = Counter(item["expected_label_from_archive"] for item in samples)
    statuses = Counter(item["status"] for item in samples)
    report = {
        "schema": SCHEMA,
        "generated_at_utc": now.isoformat(),
        "root": str(root),
        "output_dir": str(run_dir),
        "stage": "NEURAL_FALTANDO_DATA_PREPARATION_ONLY",
        "training_performed": False,
        "model_or_production_modified": False,
        "dataset_ready": False,
        "summary": {
            "total": len(samples),
            "by_label": dict(labels),
            "by_lighting": dict(Counter(item["lighting_mode"] for item in samples)),
            "extracted": statuses["EXTRACTED_PENDING_REVIEW"],
            "unusable": len(samples) - statuses["EXTRACTED_PENDING_REVIEW"],
            "verified_multilight_events": len(verified_groups),
            "name_only_triplets": len(proposals),
            "training_ready": 0,
        },
        "verified_events": verified_groups,
        "name_only_triplet_candidates": proposals,
        "samples": samples,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.txt").write_text(_summary_text(report), encoding="utf-8")
    return report, run_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepara pares visuais FALTANDO em reports/, sem treinamento."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    print("Preparação offline FALTANDO (sem KNN/CNN e sem mudar dataset)", flush=True)
    report, run_dir = prepare_faltando(root, args.output_dir)
    print(_summary_text(report))
    print(f"Manifesto: {run_dir / 'manifest.json'}")
    print(f"Resumo:    {run_dir / 'summary.txt'}")
    return 0 if report["summary"]["unusable"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
