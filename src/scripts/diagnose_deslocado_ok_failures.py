"""Diagnóstico visual somente-leitura para falsos NG de DESLOCADO no replay.

Confronta imagens gabarito/teste dos falsos NG SIDE/TOP/MID conhecidos.
A diferença RGB mostra mudança visual, NÃO é máscara do componente nem
predição da CNN. Nunca re-treina, altera pesos ou muda decisão operacional.

python -m src.scripts.diagnose_deslocado_ok_failures
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np

from src.scripts.replay_deslocado_ok_v2 import SCHEMA
from src.scripts.train_deslocado_cnn import load_ok_events, _read, _safe_png
from src.services.deslocado_neural_dataset import latest_deslocado_manifest


def _last_report(root: Path) -> Path:
    paths = sorted(
        (root/"reports"/"deslocado_neural"/"replays").glob(
            "all_ok_v2_*/deslocado_ok_replay_v2.json"
        ),
        key=lambda x: x.parent.name,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError("Execute replay_deslocado_ok_v2 primeiro")
    return paths[0]


def _tile(frame: np.ndarray, label: str, w=420, h=320) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise ValueError("Formato de frame inválido")
    height, width = frame.shape[:2]
    scale = min((w-12)/width, (h-36)/height)
    resized = cv2.resize(
        frame, (max(1, round(width*scale)), max(1, round(height*scale))),
        interpolation=cv2.INTER_NEAREST if scale > 2 else cv2.INTER_AREA,
    )
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y = 32 + (h-36-resized.shape[0])//2
    x = (w-resized.shape[1])//2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    cv2.putText(
        canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
        .56, (255,255,255), 1, cv2.LINE_AA,
    )
    return canvas


def diagnose_failures(
    root: Path, *, replay: Path | None = None,
    manifest: Path | None = None,
) -> tuple[dict, Path]:
    root = Path(root).expanduser().resolve()
    report_path = Path(replay).expanduser().resolve() if replay else _last_report(root)
    if report_path.is_symlink() or not report_path.is_file():
        raise ValueError("Relatório de replay ausente/inválido")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA or report.get("model") != (
        "visionx.deslocado_comparative_cnn.v2"
    ):
        raise ValueError("Replay incompatível com DESLOCADO v2")
    manifest_path = (
        Path(manifest).expanduser().resolve()
        if manifest else latest_deslocado_manifest(root)
    )
    if sha256(manifest_path.read_bytes()).hexdigest() != report.get(
        "evaluation_manifest_sha256"
    ):
        raise ValueError("O manifesto fornecido não é o do replay original")
    events, data = load_ok_events(manifest_path)
    if data["root"].resolve() != root:
        raise ValueError("Manifesto pertence a outra raiz")
    failed = report.get("failed_images")
    if not isinstance(failed, list):
        raise ValueError("Relatório não apresenta lista de falsos NG")
    all_paths = data["samples"]
    if len(failed) > len(all_paths):
        raise ValueError("Relatório apresenta mais falhas do que o acervo")
    prepared_paths = {
        path
        for event in events for path in event["observations"].values()
    }
    if prepared_paths != set(all_paths):
        raise ValueError("Inconsistência nos eventos preparados")

    when = datetime.now(timezone.utc)
    out = (root/"reports"/"deslocado_neural"/"diagnostics"/(
        "ok_false_ng_"+when.strftime("%Y%m%dT%H%M%S_%fZ")
    ))
    out.mkdir(parents=True, exist_ok=False)
    saved = []
    seen = set()
    for item in failed:
        if not isinstance(item, dict):
            raise ValueError("Entrada falha inválida")
        source = item.get("source_path")
        if (not isinstance(source, str) or source not in all_paths
                or source in seen or item.get("expected") != "OK"
                or item.get("predicted") != "NG"
                or item.get("lighting_mode") not in ("SIDE","TOP","MID")):
            raise ValueError("Relatório contém caso não verificável")
        seen.add(source)
        sample = all_paths[source]
        if item["lighting_mode"] != sample["lighting_mode"]:
            raise ValueError("Iluminação da falha difere do manifesto")

        ref = _read(_safe_png(data["run"], sample["reference_path"]))
        tst = _read(_safe_png(data["run"], sample["test_path"]))
        reference_panel = _tile(ref, "GABARITO / REFERENCE")
        test_panel = _tile(tst, "TESTE / REAL OK")
        # Mesmos tamanhos apenas para inspeção visual — não medir precisão
        # de deslocamento com diferença de RGB sem registro/alinhamento.
        ref_resize = cv2.resize(
            ref, (tst.shape[1], tst.shape[0]), interpolation=cv2.INTER_AREA,
        )
        diff = cv2.absdiff(ref_resize, tst)
        difference_panel = _tile(
            cv2.convertScaleAbs(diff, alpha=2.5), "DIFERENCA RGB (NAO E NG)"
        )
        side_by_side = cv2.hconcat([
            reference_panel, test_panel, difference_panel,
        ])
        filename = sha256(source.encode()).hexdigest()[:16]+"_"+item["lighting_mode"]+".png"
        if not cv2.imwrite(str(out/filename), side_by_side):
            raise OSError("Não foi possível salvar diagnóstico: "+filename)
        saved.append({
            "source_path": source,
            "score": item.get("ng_proxy_score_uncalibrated"),
            "diagnostic_png": filename,
            "lighting_mode": item["lighting_mode"],
            "reference_path": sample["reference_path"],
            "test_path": sample["test_path"],
        })
    summary = {
        "schema": "visionx.deslocado_false_ng_visual_diagnostic.v1",
        "source_replay": str(report_path),
        "source_manifest": str(manifest_path),
        "failed_cases": len(failed),
        "diagnostic_panels_created": len(saved),
        "cases": saved,
        "source_modified": False,
        "production_modified": False,
        "model_retrained": False,
        "limitations": [
            "Diferença RGB bruta NÃO localiza necessariamente o componente.",
            "São OK conhecidos, não prova de recall NG real.",
            "ROI/contorno correto deve ser confirmado visualmente pelo operador.",
        ],
    }
    (out/"diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary, out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Mostrar pares gabarito/teste de falsos NG DESLOCADO"
    )
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    summary, folder = diagnose_failures(
        args.root, replay=args.replay, manifest=args.manifest
    )
    print(f"Diagnósticos gerados: {summary['diagnostic_panels_created']}")
    print("Pasta:", folder)
    for case in summary["cases"]:
        print(f"{case['source_path']} -> {folder/case['diagnostic_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
