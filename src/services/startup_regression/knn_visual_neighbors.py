"""Diagnóstico OFFLINE dos vizinhos visuais da KNN, sem alterar vereditos.

Compara a assinatura da divergência gabarito/teste com memórias já carregadas
pela KNN (históricas + duráveis). Não cria rótulos humanos, não calcula votos
operacionais, não promove pesos nem autoriza comandos 0/1.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from src.core.anomaly_signature import (
    build_anomaly_signature, compare_anomaly_signatures, valid_anomaly_signature,
)
from src.core.strict_category_memory import (
    canonical_memory_category, canonical_memory_lighting,
)
from src.services.image_archive_dedup import image_fingerprint


SCHEMA = "visionx.knn_visual_neighbors_diagnostic.v1"


def _valid_image(image) -> bool:
    return (
        isinstance(image, np.ndarray) and image.dtype == np.uint8
        and image.ndim == 3 and image.shape[2] == 3
        and min(image.shape[:2]) >= 12
    )


def diagnose_visual_neighbors(
    reference, test, *, category: str, lighting: str,
    records_ok: list[dict], records_ng: list[dict], top_k: int = 5,
) -> dict:
    """Lista similares OK e NG SEPARADAMENTE, sem limiar de aprovação.

    A assinatura representa diferença visual, não prova de mesmo componente.
    Fontes antigas não verificadas são rotuladas como tais no relatório.
    Nenhum resultado desta função pode ser usado como reconhecimento KNOWN_KNN.
    """
    if not _valid_image(reference) or not _valid_image(test):
        raise ValueError("Par de imagens AOI BGR uint8 inválido")
    cat = canonical_memory_category(category)
    light = canonical_memory_lighting(lighting)
    if not cat or cat in {"UNKNOWN", "UNKNOW", "DESCONHECIDO"}:
        raise ValueError("Categoria AOI inválida")
    if light not in {"SIDE", "TOP", "MID"}:
        raise ValueError("Iluminação inválida")
    if not 1 <= int(top_k) <= 30:
        raise ValueError("top_k deve estar entre 1 e 30")

    query = build_anomaly_signature(
        reference, test, {}, {"category": cat, "lighting_mode": light}
    )
    if not valid_anomaly_signature(query):
        raise ValueError("Assinatura visual da consulta inválida")

    by_label = {}
    for label, records in (("OK", records_ok), ("NG", records_ng)):
        neighbors = []
        skipped = 0
        for record in records:
            if not isinstance(record, dict):
                skipped += 1
                continue
            if (
                record.get("mode") != "anomaly"
                or canonical_memory_category(record.get("category")) != cat
                or canonical_memory_lighting(record.get("lighting_mode")) != light
            ):
                continue
            signature = record.get("anomaly_signature")
            if not valid_anomaly_signature(signature):
                skipped += 1
                continue
            similarity, breakdown = compare_anomaly_signatures(query, signature)
            neighbors.append({
                "label": label,
                "similarity": float(similarity),
                "breakdown": breakdown,
                "source": str(record.get("path") or record.get("json_path") or ""),
                "source_kind": (
                    "PERSISTENT_ARCHIVE"
                    if record.get("archive_persistent") else "LEGACY_NOT_EXACT_VERIFIED"
                ),
            })
        neighbors.sort(key=lambda row: (-row["similarity"], row["source"]))
        by_label[label] = {
            "candidates": len(neighbors),
            "invalid_signatures": skipped,
            "nearest": neighbors[:int(top_k)],
        }

    return {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "recognition_route": None,
        "decision_authorized": False,
        "note": (
            "Similaridade de assinaturas NÃO é match exato, confiança, "
            "verdade humana ou autorização para julgamento/0/1."
        ),
        "category": cat,
        "lighting_mode": light,
        "reference_fingerprint": image_fingerprint(reference),
        "test_fingerprint": image_fingerprint(test),
        "neighbors_by_label": by_label,
    }


def _read_bgr(path: str) -> np.ndarray:
    target = Path(path)
    array = cv2.imdecode(np.frombuffer(target.read_bytes(), dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if not _valid_image(array):
        raise ValueError("Imagem inválida: " + str(target))
    return array


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnóstico local e NÃO decisório de vizinhos KNN"
    )
    parser.add_argument("--reference", required=True, help="PNG do gabarito já recortado")
    parser.add_argument("--test", required=True, help="PNG de teste já recortado")
    parser.add_argument("--category", required=True)
    parser.add_argument("--lighting", required=True, choices=("SIDE", "TOP", "MID"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    from src.core.experts.knn_expert import KNNExpert

    knn = KNNExpert()
    with knn._memory_lock:
        records_ok = list(knn.signatures_ok)
        records_ng = list(knn.signatures_ng)
    result = diagnose_visual_neighbors(
        _read_bgr(args.reference), _read_bgr(args.test),
        category=args.category, lighting=args.lighting,
        records_ok=records_ok, records_ng=records_ng, top_k=args.top_k,
    )
    if args.output is not None:
        target = args.output
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = Path("reports/startup_regression") / ("knn_neighbors_" + stamp + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("Relatório KNN diagnóstico: " + str(target))
    for label in ("OK", "NG"):
        row = result["neighbors_by_label"][label]
        print(f"{label}: {row['candidates']} candidatos, "
              f"{len(row['nearest'])} vizinhos exibidos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
