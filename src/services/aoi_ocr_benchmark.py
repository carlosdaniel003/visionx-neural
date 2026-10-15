"""Benchmark AOI OCR offline; nunca altera a inferência operacional.

O controle chama o ScreenMonitor original. As variantes substituem APENAS
o pré-processamento do OCR geral; a extração Board/Parts/Value, recuperação
de ambiguidades, e normalização da categoria são as mesmas da Produção.

Comparações com baseline NÃO são verdade humana. Somente rótulos conferidos
manualmente no manifesto permitem avaliar correção; este script jamais promove.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import cv2
import numpy as np

from src.utils.text_normalizer import normalize_aoi_text

FIELDS = ("board", "parts", "value", "category")
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp"})
SCHEMA = "visionx.aoi_ocr_benchmark.v1"


@dataclass(frozen=True)
class Variant:
    name: str
    scale: int | None = None
    trim_top: int = 0
    trim_sides: int = 0


VARIANTS = (
    Variant("baseline_original"),
    Variant("scale_2", scale=2),
    Variant("scale_3", scale=3),
    Variant("scale_4", scale=4),
    Variant("scale_5", scale=5),
    Variant("scale_3_trim_top_60", scale=3, trim_top=60),
    Variant("scale_3_trim_sides_20", scale=3, trim_sides=20),
)


def prepare_variant_binary(image: np.ndarray, variant: Variant) -> np.ndarray:
    """Só candidatos: controle baseline permanece código operacional original."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Região OCR inválida")
    h, w = image.shape[:2]
    top = variant.trim_top
    side = variant.trim_sides
    if h - top < 12 or w - 2 * side < 12:
        raise ValueError("Recorte candidato insuficiente")
    cropped = image[top:h, side:w-side]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    scale = variant.scale
    if scale is None:
        scale = max(3, min(5, 200 // max(gray.shape[0], 1)))
    enlarged = cv2.resize(
        gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
    )
    _, binary = cv2.threshold(
        enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary


def _candidate_reader(variant: Variant, pytesseract_module):
    def read(image, *, timing=None, phase="general_blue"):
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return ""
        try:
            binary = prepare_variant_binary(image, variant)
            callback = pytesseract_module.image_to_string
            raw = (
                callback(binary, config="--psm 6 --oem 3")
                if timing is None else timing.call(
                    phase, callback, binary, config="--psm 6 --oem 3"
                )
            )
            return raw.strip()
        except Exception:
            # Mesmo comportamento de falha da _ocr_fast operacional.
            return ""
    return read


def _image_paths(args) -> list[tuple[Path, dict | None]]:
    """Manifesto opcional com verdade HUMANA, nunca derivada de nome/pasta."""
    if args.manifest:
        manifest = Path(args.manifest).resolve(strict=True)
        rows = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("Manifesto deve ser lista não vazia")
        output = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValueError("Manifesto exige path por imagem")
            expected = row.get("expected")
            if expected is not None:
                if (
                    not isinstance(expected, dict)
                    or set(expected) != set(FIELDS)
                    or any(not isinstance(expected[k], str) for k in FIELDS)
                    or not expected["category"].strip()
                ):
                    raise ValueError(
                        "expected exige campos legado board/parts/value e "
                        "category com rótulo humano não vazio"
                    )
            source = (manifest.parent / row["path"]).resolve(strict=True)
            if manifest.parent not in source.parents:
                raise ValueError("Imagem fora da pasta do manifesto")
            output.append((source, expected))
    else:
        output = []
        for item in args.images or ():
            path = Path(item).resolve(strict=True)
            candidates = sorted(path.iterdir()) if path.is_dir() else [path]
            for candidate in candidates:
                if candidate.suffix.lower() in IMAGE_SUFFIXES:
                    output.append((candidate.resolve(strict=True), None))
    if not output:
        raise ValueError("Nenhuma imagem original AOI encontrada")
    unique = set()
    for path, _ in output:
        if path in unique or not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("Imagem duplicada ou formato inválido: " + str(path))
        unique.add(path)
    return sorted(output, key=lambda pair: str(pair[0]))


def locate_bars(monitor, frame: np.ndarray):
    """Mesmas máscaras e heurísticas de barras do fluxo externo operacional."""
    from src.config.settings import settings

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(
        hsv, settings.COLOR_BLUE_LOWER, settings.COLOR_BLUE_UPPER
    )
    red = cv2.bitwise_or(
        cv2.inRange(hsv, settings.COLOR_RED1_LOWER, settings.COLOR_RED1_UPPER),
        cv2.inRange(hsv, settings.COLOR_RED2_LOWER, settings.COLOR_RED2_UPPER),
    )
    blue_bar = monitor._find_color_bar(blue_mask)
    if blue_bar is None:
        raise ValueError("Barra azul AOI ausente")
    red_bar = (
        monitor._find_sibling_bar(red, blue_bar)
        or monitor._find_color_bar(red)
    )
    if red_bar is None:
        raise ValueError("Barra vermelha AOI ausente")
    return blue_bar, red_bar


def inspect_variant(monitor, frame, blue_bar, red_bar, variant, pytesseract_module):
    monitor._shadow_auxiliary_info = None
    started = time.perf_counter()
    if variant.name == "baseline_original":
        info = monitor._extract_text_info(frame, blue_bar, red_bar)
    else:
        with patch.object(
            monitor, "_ocr_fast",
            _candidate_reader(variant, pytesseract_module),
        ):
            info = monitor._extract_text_info(frame, blue_bar, red_bar)
    elapsed = round((time.perf_counter() - started) * 1000.0, 2)
    category, _unused = normalize_aoi_text(
        info.get("category") or info.get("value", "")
    )
    fields = {
        "board": "", "parts": "", "value": "",
        "category": category,
    }
    profile = getattr(monitor, "last_ocr_profile", {})
    return {
        "fields": fields,
        "ocr_text_detected": bool(info.get("ocr_text_detected", False)),
        "elapsed_ms": elapsed,
        "tesseract_calls": profile.get("call_count"),
        "tesseract_ms": profile.get("tesseract_calls_ms"),
        "parts_reason": profile.get("parts_reason"),
    }


def compare_variants(
    sources: list[tuple[Path, dict | None]], repeats: int = 3,
    *,
    monitor_factory=None,
    tesseract_module=None,
) -> dict:
    if not 1 <= repeats <= 20:
        raise ValueError("repeats deve estar entre 1 e 20")
    from src.services import screen_monitor
    if not screen_monitor.HAS_TESSERACT and tesseract_module is None:
        raise RuntimeError("Tesseract indisponível; comparação OCR inválida")
    module = tesseract_module or screen_monitor.pytesseract
    factory = monitor_factory or screen_monitor.ScreenMonitor
    cases = []
    by_variant = {v.name: {"ms": [], "different": set(), "wrong": set()}
                  for v in VARIANTS}
    human_count = 0
    unsupported_category_cases = set()
    for file_index, (path, expected) in enumerate(sources):
        raw = path.read_bytes()
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("Captura AOI não decodificada: " + str(path))
        monitor = factory()
        bars = locate_bars(monitor, frame)
        human_count += expected is not None
        runs = []
        baseline_fields = None
        for repetition in range(repeats):
            # Controle real SEM patch, sempre antes de comparar os candidatos.
            baseline = inspect_variant(
                monitor, frame, *bars, VARIANTS[0], module
            )
            # Campo categoria é a única saída operacional de OCR.
            # "Unknown" pode ser medido offline, sem ser elegível a promoção.
            if (
                not baseline.get("ocr_text_detected")
                or not isinstance(baseline["fields"].get("category"), str)
                or not baseline["fields"]["category"].strip()
            ):
                raise ValueError(
                    "Baseline tem campo OCR vazio/desconhecido; recusar comparação: "
                    + str(path)
                )
            if baseline["fields"]["category"] == "Unknown":
                unsupported_category_cases.add(file_index)
            if baseline_fields is None:
                baseline_fields = dict(baseline["fields"])
            elif baseline["fields"] != baseline_fields:
                raise ValueError(
                    "Baseline OCR instável entre repetições; não comparar: "
                    + str(path)
                )
            runs.append({"variant": VARIANTS[0].name, **baseline})
            by_variant[VARIANTS[0].name]["ms"].append(baseline["elapsed_ms"])
            if expected is not None and baseline["fields"] != expected:
                by_variant[VARIANTS[0].name]["wrong"].add(file_index)
            # Rotação mitiga viés da ordem de aquecimento e de uso da CPU.
            candidates = list(VARIANTS[1:])
            shift = (file_index + repetition) % len(candidates)
            candidates = candidates[shift:] + candidates[:shift]
            for variant in candidates:
                result = inspect_variant(monitor, frame, *bars, variant, module)
                changes = [
                    key for key in FIELDS
                    if result["fields"][key] != baseline["fields"][key]
                ]
                if changes:
                    by_variant[variant.name]["different"].add(file_index)
                if expected is not None and result["fields"] != expected:
                    by_variant[variant.name]["wrong"].add(file_index)
                by_variant[variant.name]["ms"].append(result["elapsed_ms"])
                runs.append({
                    "variant": variant.name, **result,
                    "different_fields": changes,
                })
        cases.append({
            "image": str(path),
            "sha256": sha256(raw).hexdigest(),
            "manifest_expected_supplied": expected is not None,
            "baseline_category_supported": (
                baseline_fields["category"] != "Unknown"
            ),
            "expected": expected,
            "measurements": runs,
        })
    summary = {}
    control_median = statistics.median(
        by_variant["baseline_original"]["ms"]
    )
    baseline_human_clean = (
        human_count == len(cases)
        and not by_variant["baseline_original"]["wrong"]
        and not unsupported_category_cases
    )
    for variant in VARIANTS:
        stat = by_variant[variant.name]
        median = statistics.median(stat["ms"])
        summary[variant.name] = {
            "median_ocr_ms": round(median, 2),
            "relative_speedup_pct": (
                round(100 * (control_median - median) / control_median, 2)
                if control_median > 0 else None
            ),
            "different_from_baseline_cases": len(stat["different"]),
            "mismatch_with_human_labels_cases": len(stat["wrong"]),
            "same_fields_as_baseline": not stat["different"],
            "human_labels_complete": human_count == len(cases),
            "unsupported_baseline_category_cases": len(unsupported_category_cases),
            "manifest_reference_agreement": (
                baseline_human_clean and not stat["wrong"]
            ),
            "variant_configuration": {
                "scale": variant.scale,
                "trim_top": variant.trim_top,
                "trim_sides": variant.trim_sides,
            },
            "eligible_for_automatic_activation": False,
        }
    return {
        "schema": SCHEMA,
        "mode": "offline_read_only_benchmark",
        "no_production_change": True,
        "auto_promotion": False,
        "reference_note": (
            "Igualdade com OCR antigo NAO prova acerto; expected no manifesto "
            "e fornecido pelo operador, sem verificação automática de autoria. "
            "Conferir manualmente I/1, O/0, intervalos e categoria. "
            "Categorias Unknown sao permitidas somente para benchmark; "
            "sao fora do escopo operacional e nao contam como aprovadas. "
            "Mesmo sem divergências, nenhuma variante é ativada."
        ),
        "case_count": len(cases),
        "manifest_expected_count": human_count,
        "unsupported_baseline_category_cases": len(unsupported_category_cases),
        "repeats": repeats,
        "summary": summary,
        "cases": cases,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", nargs="+", help="PNG/JPGs AOI integrais ou diretórios")
    group.add_argument("--manifest", help="JSON com path e expected humano opcional")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True, help="Relatório JSON novo (não sobrescreve)")
    args = parser.parse_args(argv)
    sources = _image_paths(args)
    out = Path(args.output).resolve()
    if out in {path for path, _ in sources}:
        raise ValueError("Relatório não pode substituir imagem AOI")
    data = compare_variants(sources, repeats=args.repeats)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusividade impede sobrescrever relatórios anteriores do operador.
    with out.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print("OCR benchmark finalizado (somente leitura).")
    print("Casos: {} | referencias informadas: {} | rounds: {}".format(
        data["case_count"], data["manifest_expected_count"], data["repeats"]
    ))
    if data["unsupported_baseline_category_cases"]:
        print(
            "AVISO: {} imagem(ns) com categoria fora do escopo operacional "
            "(Unknown). Benchmark diagnóstico; nao validar para Produção.".format(
                data["unsupported_baseline_category_cases"]
            )
        )
    for name, value in data["summary"].items():
        print("{}: mediana {} ms; diferencas {}; ganho {}%".format(
            name, value["median_ocr_ms"],
            value["different_from_baseline_cases"],
            value["relative_speedup_pct"],
        ))
    print("Nenhum candidato foi ativado. Relatório: " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
