"""Auditoria LOCAL do NG humano: CNN ativa versus vizinhos visuais KNN.

O archive guarda exemplos semelhantes sem que isso os torne pares EXATOS.
Este serviço recupera as assinaturas históricas/persistentes da KNN por luz
e mostra os vizinhos OK/NG, sem emitir voto, mudar pesos ou criar rótulos.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.core.strict_category_memory import canonical_memory_category
from src.services.startup_regression.cnn_confirmed_event_audit import audit_confirmed_event
from src.services.startup_regression.cnn_physical_veto_replay import _load_verified_image
from src.services.startup_regression.knn_visual_neighbors import diagnose_visual_neighbors

LIGHTS = ("SIDE", "TOP", "MID")
SCHEMA = "visionx.faltando_human_ng_archive_neighbors.v1"


def audit_event_neighbors(
    root: Path, aoi_event_id: str, *,
    records_ok: list[dict], records_ng: list[dict], top_k: int = 5,
) -> dict:
    """Compara somente eventos humanos do journal, com PNGs SHA-256 válidos."""
    root = Path(root).resolve()
    audit = audit_confirmed_event(root, aoi_event_id)
    report = {
        "schema": SCHEMA, "event_id": aoi_event_id,
        "journal_state": audit["state"],
        "read_only": True, "decision_authorized": False,
        "active_cnn_weights_changed": False,
        "note": (
            "NEW_CNN significa par não idêntico; vizinhos semelhantes podem "
            "estar no ng_archive. Similaridade NÃO é rótulo nem probabilidade. "
            "Este comando NÃO retreina a CNN FALTANDO V2."
        ),
        "events": [],
    }
    for matched in audit["events"]:
        if canonical_memory_category(matched["category"]) != "FALTANDO":
            raise ValueError("Categoria da confirmação humana não é FALTANDO")
        folder = root / "reports/neural_online/events"
        path = folder / (matched["journal_event_id"] + ".json")
        info = json.loads(path.read_text(encoding="utf-8"))
        results = {}
        for image in info.get("images", []):
            light = image.get("lighting_mode")
            if light not in LIGHTS or light in results:
                raise ValueError("Iluminação duplicada ou desconhecida no evento")
            try:
                reference = _load_verified_image(
                    folder, image.get("reference"), image.get("reference_sha256"),
                )
                test = _load_verified_image(
                    folder, image.get("test"), image.get("test_sha256"),
                )
                result = diagnose_visual_neighbors(
                    reference, test, category="FALTANDO", lighting=light,
                    records_ok=records_ok, records_ng=records_ng, top_k=top_k,
                )
                results[light] = {
                    "status": "ANALYZED", "neighbors": result["neighbors_by_label"],
                    "reference_fingerprint": result["reference_fingerprint"],
                    "test_fingerprint": result["test_fingerprint"],
                }
            except (OSError, ValueError, TypeError) as exc:
                results[light] = {
                    "status": "UNAVAILABLE",
                    "reason": type(exc).__name__,
                    "neighbors": None,
                }
        for light in LIGHTS:
            results.setdefault(light, {
                "status": "UNAVAILABLE",
                "reason": "MISSING_LIGHT_PAIR",
                "neighbors": None,
            })
        report["events"].append({
            "journal_event_id": matched["journal_event_id"],
            "human_label": matched["label"],
            "learning_job_status": matched["job_status"],
            "checkpoint_promoted_by_job": matched["promoted"],
            "active_checkpoint_points_to_this_job": matched["checkpoint_active_for_event"],
            "lighting": results,
        })
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Vizinhos KNN do evento humano (diagnóstico, sem classificar)"
    )
    parser.add_argument("--aoi-event-id", required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= 30:
        parser.error("--top-k deve estar entre 1 e 30")
    from src.config.settings import BASE_DIR
    if args.root.resolve() != Path(BASE_DIR).resolve():
        parser.error("--root deve apontar para a raiz da instalação atual")
    # O KNNExpert utiliza exatamente esta instalação, sem mesclar outro disco.
    from src.core.experts.knn_expert import KNNExpert
    memory = KNNExpert()
    with memory._memory_lock:
        ok = list(memory.signatures_ok)
        ng = list(memory.signatures_ng)
    report = audit_event_neighbors(
        args.root, args.aoi_event_id,
        records_ok=ok, records_ng=ng, top_k=args.top_k,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = args.output or (
        args.root / "reports/startup_regression" /
        ("cnn_knn_ng_neighbors_" + stamp + ".json")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("Relatório de vizinhos CNN/KNN:", str(target))
    print("Journal:", report["journal_state"])
    for event in report["events"]:
        print("Rótulo humano:", event["human_label"],
              "treino:", event["learning_job_status"],
              "promovido:", event["checkpoint_promoted_by_job"])
        for light in LIGHTS:
            item = event["lighting"][light]
            if item["status"] != "ANALYZED":
                print(light, item["status"], item["reason"])
                continue
            groups = item["neighbors"]
            print(light, "vizinhos OK:", groups["OK"]["candidates"],
                  "NG:", groups["NG"]["candidates"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
