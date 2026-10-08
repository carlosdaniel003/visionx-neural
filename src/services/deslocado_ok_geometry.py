"""Diagnóstico offline de geometria DESLOCADO — somente pares OK.

Não desenha máscaras, não usa memória KNN, não carrega checkpoint,
não treina rede neural e não atribui OK/NG a defeitos desconhecidos.
Cada indicador é DESCRITIVO, não uma previsão de deslocamento.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from src.scripts.train_deslocado_cnn import load_ok_events, _read, _safe_png
from src.scripts.replay_deslocado_ok_v2 import all_archived_deslocado
from src.services.deslocado_neural_dataset import latest_deslocado_manifest

SCHEMA = "visionx.deslocado_ok_geometry_diagnostic.v1"
LIGHTS = ("SIDE", "TOP", "MID")
HARD_SIDE = (
    "2026-10-02_1349_DESLOCADO.png",
    "2026-10-02_1400_DESLOCADO.png",
    "2026-10-02_1415_DESLOCADO.png",
)


def _zones(height: int, width: int) -> dict[str, np.ndarray]:
    """Janelas NORMALIZADAS DE ANÁLISE, jamais segmentação de componente.

    A janela central pode conter inscrições, mas depende do enquadramento.
    A região externa pode conter PCB ou parte do componente: não prova
    por si só o alinhamento com pads fixos.
    """
    def rectangle(x0, y0, x1, y1):
        mask = np.zeros((height, width), dtype=bool)
        mask[round(height*y0):round(height*y1),
             round(width*x0):round(width*x1)] = True
        return mask

    center = rectangle(.36, .34, .64, .71)
    body_context = rectangle(.18, .13, .82, .88) & ~center
    outer_context = ~rectangle(.16, .12, .84, .88)
    return {"center": center, "body_context": body_context,
            "outer_context": outer_context}


def _edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _scale_for_pair(reference: np.ndarray, test: np.ndarray):
    h, w = reference.shape[:2]
    ht, wt = test.shape[:2]
    fractional_difference = max(abs(ht/h - 1.0), abs(wt/w - 1.0))
    if fractional_difference <= .05 and (h, w) != (ht, wt):
        test = cv2.resize(test, (w, h), interpolation=cv2.INTER_AREA)
    return test, fractional_difference


def analyze_pair(reference: np.ndarray, test: np.ndarray) -> dict:
    """Retorna pistas físicas auditáveis, sem classificar NG ou aprovar OK."""
    if (
        reference is None or test is None
        or reference.ndim != 3 or test.ndim != 3
        or reference.shape[2] != 3 or test.shape[2] != 3
        or min(*reference.shape[:2], *test.shape[:2]) < 32
    ):
        raise ValueError("Imagem de gabarito/teste inválida para diagnóstico")

    ref_h, ref_w = reference.shape[:2]
    test_h, test_w = test.shape[:2]
    test_scaled, scale_change = _scale_for_pair(reference, test)
    output = {
        "reference_size_wh": [ref_w, ref_h],
        "test_size_wh": [test_w, test_h],
        "resized_test_for_comparison": (ref_h, ref_w) != (test_h, test_w)
                                        and scale_change <= .05,
        "relative_size_change": round(float(scale_change), 6),
        "method": "PERIPHERAL_EDGE_PHASE_CORRELATION_DIAGNOSTIC_ONLY",
        "component_segmented": False,
        "fixed_pads_verified": False,
        "detected_real_ng": None,
        "requires_review_for_operational_decision": True,
    }
    if scale_change > .05:
        output.update(
            evidence_status="EVIDENCIA_INSUFICIENTE",
            reasons=["CROP_DIMENSIONS_INCOMPATIBLE"],
            anchor_phase_response=None, estimated_global_shift_xy=None,
            central_gray_difference=None, peripheral_edge_difference=None,
        )
        return output

    h, w = reference.shape[:2]
    zones = _zones(h, w)
    gref = _edges(reference)
    gtest = _edges(test_scaled)
    anchor = zones["outer_context"]

    # Textura distribuída por quatro regiões reduz falsa confiança quando
    # somente um traço ou pad isolado aparece na borda do recorte.
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    top, left = ys < h//2, xs < w//2
    sectors = [
        anchor & top & left,
        anchor & top & ~left,
        anchor & ~top & left,
        anchor & ~top & ~left,
    ]
    strength = float(np.mean(gref[anchor])) if anchor.any() else 0.0
    textured_sectors = sum(float(np.mean(gref[s])) >= 6.0 for s in sectors if s.any())
    output["outer_reference_edge_strength"] = round(strength, 4)
    output["outer_textured_sectors"] = textured_sectors

    # O centro (potencial inscrição) NÃO participa da estimativa de registro.
    # A correlação na faixa exterior é somente hipótese de movimento da câmera.
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    context_window = window * anchor.astype(np.float32)
    first = gref * context_window
    second = gtest * context_window
    shift, response = cv2.phaseCorrelate(first, second)
    dx, dy = float(shift[0]), float(shift[1])
    finite_registration = all(np.isfinite(x) for x in (dx, dy, response))
    output["anchor_phase_response"] = (
        round(float(response), 6) if finite_registration else None
    )
    output["estimated_global_shift_xy"] = (
        [round(dx, 3), round(dy, 3)] if finite_registration else None
    )

    reasons = []
    if strength < 6.0 or textured_sectors < 2:
        reasons.append("WEAK_OR_LOCALIZED_BACKGROUND_ANCHORS")
    if not finite_registration or response < .12:
        reasons.append("UNRELIABLE_BACKGROUND_REGISTRATION")
    if abs(dx) > .08*w or abs(dy) > .08*h:
        reasons.append("LARGE_OR_AMBIGUOUS_GLOBAL_SHIFT")

    # Os valores abaixo descrevem dissimilaridade de pixels/arestas após
    # registro global; não detectam nem rotulam o corpo do componente.
    if not reasons:
        affine = np.float32([[1, 0, -dx], [0, 1, -dy]])
        aligned = cv2.warpAffine(
            test_scaled, affine, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        gray_test = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        differences = cv2.absdiff(gray_ref, gray_test)
        edges_test = _edges(aligned)
        # Escala fixa (0-255) para ser comparável entre observações.
        out_diff = np.abs(gref - edges_test)
        out_diff = np.minimum(out_diff, 255)
        output["central_gray_difference"] = round(
            float(np.mean(differences[zones["center"]])/255), 6
        )
        output["peripheral_edge_difference"] = round(
            float(np.mean(out_diff[zones["body_context"]])/255), 6
        )
        output["evidence_status"] = "METRICAS_DESCRITIVAS_DISPONIVEIS"
    else:
        output["central_gray_difference"] = None
        output["peripheral_edge_difference"] = None
        output["evidence_status"] = "EVIDENCIA_INSUFICIENTE"
    output["reasons"] = reasons
    return output


def _new_output(root: Path) -> Path:
    when = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = (
        root / "reports" / "deslocado_neural" / "diagnostics"
        / ("ok_only_geometry_" + when)
    )
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def diagnose_ok_geometry(root: Path, manifest: Path | None = None) -> tuple[dict, Path]:
    """Inspeciona integralmente o acervo OK disponível no instante do replay."""
    root = Path(root).expanduser().resolve()
    manifest_path = (
        Path(manifest).expanduser().resolve()
        if manifest is not None else latest_deslocado_manifest(root)
    )
    events, data = load_ok_events(manifest_path)
    if data["root"] != root:
        raise ValueError("Manifesto pertence a outra raiz")
    archived_ok, archived_ng = all_archived_deslocado(root)
    if archived_ng:
        raise ValueError(
            "DESLOCADO NG real encontrado: interromper diagnóstico exclusivamente OK"
        )
    if archived_ok != set(data["samples"]):
        missing = sorted(archived_ok - set(data["samples"]))
        stale = sorted(set(data["samples"]) - archived_ok)
        raise ValueError(
            f"Inventário DESLOCADO mudou: novos={len(missing)}, removidos={len(stale)}; "
            "execute novamente a preparação offline."
        )
    if data["warnings"]:
        raise ValueError("Agrupamento SIDE/TOP/MID incerto: " + repr(data["warnings"]))

    rows = []
    image_hashes = defaultdict(list)
    per_event = []
    for event in events:
        for light, source in event["observations"].items():
            info = data["samples"][source]
            reference_file = _safe_png(data["run"], info["reference_path"])
            test_file = _safe_png(data["run"], info["test_path"])
            reference, test = _read(reference_file), _read(test_file)
            result = analyze_pair(reference, test)
            pair_key = (
                sha256(reference_file.read_bytes()).hexdigest(),
                sha256(test_file.read_bytes()).hexdigest(),
            )
            image_hashes[pair_key].append(source)
            rows.append({
                "source_path": source, "event_id": event["id"],
                "event_association": event["association"],
                "split_key_candidate": event["split_key"],
                "lighting_mode": light,
                "lighting_source": info.get("lighting_source"),
                "source_sha256": info["source_sha256"],
                "reference_sha256": pair_key[0],
                "test_sha256": pair_key[1],
                "reference_path": info["reference_path"],
                "test_path": info["test_path"],
                "historical_false_ng_case": Path(source).name in HARD_SIDE,
                "archive_label": "OK",
                **result,
            })
        per_event.append({
            "event_id": event["id"],
            "association": event["association"],
            "split_key_candidate": event["split_key"],
            "lights": list(event["observations"]),
            "sources": list(event["observations"].values()),
        })

    if len(rows) != len(archived_ok) or {x["source_path"] for x in rows} != archived_ok:
        raise AssertionError("Diagnóstico incompleto: houve imagens omitidas")
    by_light = {
        light: dict(Counter(x["evidence_status"] for x in rows
                            if x["lighting_mode"] == light))
        for light in LIGHTS
    }
    duplicates = [
        sorted(paths) for paths in image_hashes.values() if len(paths) > 1
    ]
    hard_results = [x for x in rows if x["historical_false_ng_case"]]
    summary = {
        "schema": SCHEMA,
        "category": "DESLOCADO",
        "mode": "OK_ONLY_DESCRIPTIVE_GEOMETRY",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "total_ok_images": len(rows),
        "total_candidate_events": len(events),
        "source_lights": dict(Counter(x["lighting_mode"] for x in rows)),
        "evidence_statuses": dict(Counter(x["evidence_status"] for x in rows)),
        "by_light": by_light,
        "historical_false_ng_cases_present": len(hard_results),
        "historical_false_ng_case_results": [
            {"source_path": x["source_path"],
             "evidence_status": x["evidence_status"],
             "reasons": x["reasons"]} for x in hard_results
        ],
        "candidate_event_links_unverified": sum(
            e["association"] == "UNVERIFIED_NAME_OCR_CANDIDATE"
            for e in per_event
        ),
        "exact_duplicate_extracted_pair_groups": duplicates,
        "unverified_links_are_not_independent_evidence": True,
        "real_ng_count": 0, "real_ng_recall": None,
        "trained_model": False, "knn_used": False,
        "production_approved": False, "production_modified": False,
        "automated_ok_decision_enabled": False,
        "meaning": (
            "Medidas automáticas exploratórias de contraste/arestas. "
            "Não são prova de alinhamento físico ou capacidade NG."
        ),
        "events": per_event,
        "images": rows,
    }
    out = _new_output(root)
    (out / "deslocado_ok_geometry.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "DESLOCADO — DIAGNÓSTICO AUTOMÁTICO SOMENTE OK (SEM DESENHO)",
        f"Manifesto: {manifest_path}",
        f"SHA-256: {summary['source_manifest_sha256']}",
        f"Imagens: {len(rows)} | Eventos candidatos: {len(events)}",
        f"Iluminações: {summary['source_lights']}",
        f"Status de evidência: {summary['evidence_statuses']}",
        f"Trincas não verificadas por manifesto: {summary['candidate_event_links_unverified']}",
        f"Pares extraídos idênticos: {len(duplicates)} grupos",
        f"Três falsos NG SIDE históricos encontrados: {len(hard_results)}/3",
        "SEM KNN, SEM TREINAMENTO, SEM CLASSIFICAÇÃO NG, SEM PRODUÇÃO.",
        "Métricas estruturais NÃO são um veredito físico.",
        "",
        "RESULTADOS POR IMAGEM",
    ]
    for item in rows:
        lines.append(
            f"{item['lighting_mode']} | {item['evidence_status']} | "
            f"ancora={item['anchor_phase_response']} "
            f"inscricao_central={item['central_gray_difference']} "
            f"bordas_perifericas={item['peripheral_edge_difference']} | "
            f"{item['source_path']} | {','.join(item['reasons']) or '-'}"
        )
    (out / "deslocado_ok_geometry.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary, out
