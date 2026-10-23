"""Plano supervisionado de correção dos falsos NG da CNN FALTANDO V2.

Usa o MESMO comparador offline KNN/CNN e suas hashes cruzadas. Lista as
capturas OK que a CNN recusou, assim como exemplos NG de proteção próximos
de capturas OK. Nenhum voto KNN vira rótulo e nenhum checkpoint é alterado.

Com --verify-files confere SHA-256 atual de CADA PNG do acervo, falhando
integralmente se houver alterações (evita treino contra relatório obsoleto).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

SCHEMA = "visionx.faltando_v2_hard_ok_recovery_plan.v1"
INPUT_SCHEMA = "visionx.knn_cnn_arbitration_shadow.v1"
CHECKPOINT_PREFIX = "c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c"


def _origin(root: Path, path: str) -> Path:
    if not isinstance(path, str):
        raise ValueError("Caminho de PNG inválido")
    relative = Path(path)
    if (
        relative.is_absolute() or ".." in relative.parts
        or len(relative.parts) < 3
        or tuple(relative.parts[:2]) not in (
            ("public", "ok_archive"), ("public", "ng_archive")
        )
        or relative.suffix.lower() != ".png"
    ):
        raise ValueError(f"Caminho fora do acervo protegido: {path}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError("Referência fora da raiz ou symlink")
    return target


def recovery_plan(report: dict, root: Path, *, verify_files: bool = False) -> dict:
    contract = report.get("evidence_contract") or {}
    if not (
        report.get("schema") == INPUT_SCHEMA
        and report.get("mode") == "OFFLINE_PNG_KNN_CNN_AB_ONLY"
        and report.get("complete") is True
        and report.get("production_approved") is False
        and report.get("xp_command_authorized") is False
        and report.get("production_route_reproduced") is False
        and contract.get("original_png_hashes_verified_between_reports") is True
        and contract.get("archives_same_files") is True
        and contract.get("independent_event_holdout") is False
    ):
        raise ValueError("Comparador não é íntegro/compatível para auditoria")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != report.get("archive_pngs"):
        raise ValueError("Inventário incompleto ou ausente")
    if len(rows) < 1:
        raise ValueError("Sem imagens")
    seen = set()
    candidates, guards = [], []
    counts = Counter()
    for row in rows:
        path = row.get("source_path")
        if path in seen:
            raise ValueError("PNG duplicado no relatório")
        seen.add(path)
        source = _origin(root, path)
        digest = row.get("source_file_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("SHA original ausente")
        expected = row.get("expected_label")
        folder_label = "OK" if path.startswith("public/ok_archive/") else "NG"
        if expected != folder_label:
            raise ValueError("Rótulo não corresponde à pasta original")
        if verify_files:
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"PNG ausente ou link: {path}")
            if sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Arquivo mudou desde o comparador: {path}")
        cnn = row.get("cnn") or {}
        if cnn.get("label") not in (None, "OK", "NG"):
            raise ValueError("Classe CNN inválida")
        if cnn.get("checkpoint_sha256") and (
            cnn["checkpoint_sha256"] != CHECKPOINT_PREFIX
        ):
            raise ValueError("Checkpoint diferente do modelo original em estudo")
        if row.get("category") != "FALTANDO":
            continue  # categoria apenas para COORTE de auditoria, nunca veredito
        light = row.get("lighting")
        if light not in ("SIDE", "TOP", "MID"):
            continue
        data = row.get("knn_visual") or {}
        neighbors = data.get("neighbors") or []
        if not isinstance(neighbors, list) or len(neighbors) > 5:
            raise ValueError("Vizinhos KNN inválidos")
        for neighbor in neighbors:
            if not isinstance(neighbor, dict):
                raise ValueError("Vizinho inválido")
        score = cnn.get("ng_score_uncalibrated")
        if expected == "OK" and cnn.get("label") == "NG":
            candidates.append({
                "path": path, "source_file_sha256": digest,
                "operator_review_required": True,
                "provisional_archive_label": "OK",
                "light": light,
                "cnn_ng_score_uncalibrated": score,
                "knn_selective_suggestion": data.get("suggested_label"),
                "top_neighbors": [
                    {"path": v.get("source_path"),
                     "label": v.get("label"),
                     "similarity": v.get("similarity")}
                    for v in neighbors
                ],
                "review_priority": "FIRST" if light == "SIDE" else "OTHER_LIGHT",
            })
            counts[light] += 1
        if expected == "NG" and cnn.get("label") == "NG":
            ok_neighbors = [
                v for v in neighbors if v.get("label") == "OK"
            ]
            guards.append({
                "path": path, "source_file_sha256": digest,
                "provisional_archive_label": "NG",
                "light": light,
                "cnn_ng_score_uncalibrated": score,
                "max_ok_neighbor_similarity": max(
                    (float(v.get("similarity") or 0) for v in ok_neighbors),
                    default=0.0,
                ),
                "hard_negative": bool(ok_neighbors),
                "operator_review_required": True,
            })
    candidates.sort(key=lambda r: (
        r["light"] != "SIDE",
        -float(r["cnn_ng_score_uncalibrated"] or 0),
        r["path"],
    ))
    guards.sort(key=lambda r: (
        r["light"] != "SIDE",
        -r["max_ok_neighbor_similarity"],
        r["path"],
    ))
    if not candidates or not guards:
        raise ValueError("Faltam exemplos OK difíceis ou NG de proteção")
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report_schema": INPUT_SCHEMA,
        "source_archive_png_count": len(rows),
        "source_checkpoint_sha256": CHECKPOINT_PREFIX,
        "source_files_verified": verify_files,
        "training_run": False,
        "candidate_checkpoint_generated": False,
        "model_changed": False,
        "auto_ok_authorized": False,
        "xp_commands_sent": 0,
        "independent_holdout_verified": False,
        "provisional_labels_not_human_verified": True,
        "false_ng": {
            "count": len(candidates),
            "by_light": dict(counts),
            "prioritized_cases": candidates,
        },
        "protected_ng": {
            "count": len(guards),
            "hard_negatives_with_ok_neighbors": sum(
                x["hard_negative"] for x in guards),
            "prioritized_cases": guards,
        },
        "next_step": (
            "Revisar manualmente pares AOI das duas listas, confirmar OK/NG "
            "em qualification.json e executar treinamento balanceado "
            "offline; não utilizar rótulos de pasta como prova humana."
        ),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Audita falsos NG FALTANDO da CNN e NG de proteção (sem treinamento)"
    )
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--verify-files", action="store_true")
    ns = p.parse_args(argv)
    report = json.loads(ns.report.read_text(encoding="utf-8"))
    root = ns.root.resolve()
    result = recovery_plan(report, root, verify_files=ns.verify_files)
    dest = root / "reports/startup_regression"
    dest.mkdir(parents=True, exist_ok=True)
    basename = "faltando_hard_ok_recovery_" + datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    out = dest / (basename + ".json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    txt = dest / (basename + ".txt")
    txt.write_text("\n".join([
        "FALTANDO V2 — PLANO DE RECUPERAÇÃO DOS FALSOS NG",
        "SOMENTE ESTUDO; SEM MUDANÇA DA CNN OU AUTORIZAÇÃO XP",
        f"Arquivos originais SHA verificados: {ns.verify_files}",
        f"OK recusados pela CNN: {result['false_ng']['count']}",
        f"OK recusados por iluminação: {result['false_ng']['by_light']}",
        f"NGs para proteger: {result['protected_ng']['count']}",
        f"NGs com vizinhos OK: {result['protected_ng']['hard_negatives_with_ok_neighbors']}",
        "",
        "REVISÃO PRIORITÁRIA — FALHAS FALSAS CNN NG",
        *[
            f"{item['light']} | CNN NG={item['cnn_ng_score_uncalibrated']} "
            f"| KNN={item['knn_selective_suggestion']} | {item['path']}"
            for item in result["false_ng"]["prioritized_cases"]
        ],
        "", "CONTRAPROVAS NG (MAIORES SIMILARIDADES COM OK)",
        *[
            f"{item['light']} | vizinho OK={item['max_ok_neighbor_similarity']:.5f} "
            f"| {item['path']}"
            for item in result["protected_ng"]["prioritized_cases"][:60]
        ],
        "", result["next_step"],
    ]), encoding="utf-8")
    print(json.dumps({
        "status": "AUDIT_ONLY", "false_ng": result["false_ng"]["by_light"],
        "protected_ng": result["protected_ng"]["count"],
        "json": str(out), "text": str(txt),
        "auto_ok_authorized": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
