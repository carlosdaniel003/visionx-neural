"""Export the nine provisional validation disagreements for human review.

Reconstructs the exact repaired-candidate validation set, checks every SHA,
evaluates the saved model, and emits ONLY a local diagnostic ZIP. No training,
qualification writes, auto-labeling, online pointer changes or XP traffic.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.scripts.audit_aoi_universal_operator_confirmed import _original_manifest, _read_json
from src.scripts.train_aoi_universal_hard_case_repair import (
    REPAIR_REPORT, REPAIR_SCHEMA, _protected_split,
)
from src.scripts.train_aoi_universal_quality_offline import _metrics
from src.scripts.train_faltando_cnn_v2_online import _assess
from src.services.aoi_universal_quality_archive import universal_archive_events
from src.services.odin_operator_confirmed_labels import verified_operator_labels

QUEUE_SCHEMA = "visionx.aoi_universal_validation_visual_queue.v1"


def _validated_candidate(root: Path, report: Path) -> tuple[dict, Path, dict]:
    path = report if report.is_absolute() else root / report
    if path.is_symlink() or not path.is_file():
        raise ValueError("Relatório ausente ou symlink")
    path = path.resolve()
    container = (root / "reports/aoi_universal_quality/hard_case_repairs").resolve()
    if path.parent.parent != container or path.name != "hard_case_repair_report.json":
        raise ValueError("Relatório fora dos reparos offline")
    info = _read_json(path)
    if (info.get("schema") != REPAIR_REPORT
            or info.get("candidate_only") is not True
            or info.get("production_approved") is not False
            or info.get("automatic_ok_authorized") is not False
            or info.get("eligible_for_production") is not False):
        raise ValueError("Candidato não é um experimento offline")
    relative = info.get("candidate_path")
    if (not isinstance(relative, str)
            or Path(relative).is_absolute() or ".." in Path(relative).parts):
        raise ValueError("Caminho do candidato inválido")
    candidate = root / relative
    if (candidate.is_symlink() or not candidate.is_file()
            or candidate.resolve().parent != path.parent
            or candidate.name != "aoi_operator_hard_case_candidate.pt"):
        raise ValueError("Candidato ausente ou fora do relatório")
    digest = sha256(candidate.read_bytes()).hexdigest()
    if digest != info.get("candidate_sha256"):
        raise ValueError("Checkpoint modificado após treinamento")
    chk = torch.load(candidate, map_location="cpu", weights_only=True)
    if (not isinstance(chk, dict)
            or chk.get("schema") != REPAIR_SCHEMA
            or chk.get("architecture_schema") != MODEL_SCHEMA_V2
            or chk.get("candidate_only") is not True
            or chk.get("production_approved") is not False
            or chk.get("automatic_ok_authorized") is not False
            or chk.get("source_candidate_sha256") != info.get("source_candidate_sha256")):
        raise ValueError("Checkpoint incompatível")
    return info, candidate, chk


def _rank_cases(rows: list[dict], result: dict, summary: dict) -> list[dict]:
    cases = result.get("cases")
    if not isinstance(cases, list) or len(cases) != len(rows):
        raise ValueError("Replay de validação incompleto")
    expected = summary.get("validation_selected", {})
    count = Counter()
    pending = []
    for event, observation in zip(rows, cases):
        if (event["id"] != observation.get("event_id_or_group_hint")
                or event["label"] != observation.get("label")
                or observation.get("predicted") not in ("NG", "OK")):
            raise ValueError("Resultado não corresponde ao fold original")
        label, predicted = event["label"], observation["predicted"]
        count[(label, predicted)] += 1
        if label != predicted:
            pending.append((event, observation))
    if (len(cases) != expected.get("total")
            or count["NG", "OK"] != expected.get("NG_false_OK")
            or count["OK", "NG"] != expected.get("OK_false_NG")
            or count["NG", "NG"] != expected.get("NG_correct")
            or count["OK", "OK"] != expected.get("OK_correct")):
        raise ValueError("Divergência entre o replay e relatório original")
    if len(pending) != 9:
        raise ValueError("Esperados exatamente nove julgamentos divergentes")
    return pending


def build_visual_queue(root: Path, repair_report: Path, manifest: Path) -> dict:
    root = Path(root).resolve()
    info, candidate, checkpoint = _validated_candidate(root, repair_report)
    human = verified_operator_labels(root)
    manifest_path, _ = _original_manifest(
        root, Path(manifest), checkpoint["source_manifest_sha256"],
    )
    records, prov = universal_archive_events(manifest_path)
    _, validation, _, split = _protected_split(
        records, human, seed=info.get("split", {}).get(
            "original_split", {}).get("seed", 42),
    )
    if (len(validation) != info["split"]["validation_records_after_fence"]
            or len(validation) != info["validation_selected"]["total"]
            or sha256(manifest_path.read_bytes()).hexdigest() !=
               checkpoint["source_manifest_sha256"]):
        raise ValueError("A base/fold mudou desde o treinamento")
    pointer = root / "reports/neural_online/live_active.json"
    if pointer.is_symlink():
        raise ValueError("Ponteiro ativo symlink não autorizado")
    pointer_before = sha256(pointer.read_bytes()).hexdigest() if pointer.exists() else None
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    model = FaltandoCNNV2()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    assessed = _assess(
        model, validation, checkpoint["image_size"], checkpoint["focus_fraction"],
    )
    pending = _rank_cases(validation, assessed, info)
    now = datetime.now(timezone.utc)
    target = root / "reports/aoi_universal_quality/validation_visual_reviews" / (
        now.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:6]
    )
    target.mkdir(parents=True, exist_ok=False)
    items = []
    assets = {}
    html_cases = []
    for ix, (event, outcome) in enumerate(pending, 1):
        if len(event["sources"]) != 1 or len(event["views"]) != 1:
            raise ValueError("Não é um evento de vista individual")
        source = event["sources"][0]
        if source in human:
            raise ValueError("Caso já confirmado pelo operador no holdout")
        origin = root / source
        if (origin.is_symlink() or not origin.is_file()
                or sha256(origin.read_bytes()).hexdigest() != event["source_sha256"]):
            raise ValueError("Captura original não corresponde ao manifesto: " + source)
        light, views = next(iter(event["views"].items()))
        filenames = {}
        hashes = {}
        for role, path in (
            ("original", origin), ("reference", Path(views["reference"])),
            ("test", Path(views["test"])),
        ):
            file = Path(path)
            if file.is_symlink() or not file.is_file():
                raise ValueError("Imagem derivada indisponível/symlink: " + str(file))
            payload = file.read_bytes()
            relative = f"images/{ix:02d}_{role}.png"
            assets[relative] = payload
            filenames[role] = relative
            hashes[role] = sha256(payload).hexdigest()
        item = {
            "case": ix, "source_path": source, "source_sha256": event["source_sha256"],
            "folder_quality_provisional": event["label"],
            "candidate_prediction": outcome["predicted"],
            "ng_score": outcome["ng_score"],
            "aoi_hint_not_physical_defect": event["category_hint"],
            "lighting": light,
            "operator_verdict": None,
            "status": "PENDING_OPERATOR_REVIEW",
            "asset_paths": filenames, "asset_sha256": hashes,
        }
        items.append(item)
        html_cases.append(
            f"<section><h2>{ix:02d} · {escape(Path(source).name)}</h2>"
            f"<p>Arquivo provisório: <b>{event['label']}</b> | "
            f"CNN: <b>{outcome['predicted']}</b> | "
            f"Score NG: {outcome['ng_score']:.6f} | "
            f"Iluminação: {escape(light)}</p>"
            f"<div class='images'>"
            + "".join(
                f"<figure><img src='{filenames[role]}' alt='{role}' />"
                f"<figcaption>{title}</figcaption></figure>"
                for role, title in (
                    ("original", "Captura AOI original"),
                    ("reference", "Gabarito"),
                    ("test", "Teste"),
                )
            )
            + "</div><p>Veredito físico do operador: OK / NG / Incerto (pendente)</p>"
            "</section>"
        )
    html = (
        "<!doctype html><html lang='pt-BR'><meta charset='utf-8'/>"
        "<title>ODIN — revisão visual da validação CNN</title>"
        "<style>body{font:15px system-ui;background:#111827;color:#eef2ff;"
        "margin:2rem auto;max-width:1200px;padding:0 1rem}"
        "section{border:1px solid #455065;border-radius:12px;padding:1rem;"
        "margin:1rem 0}h1{font-size:1.8rem}.images{display:flex;"
        "flex-wrap:wrap;gap:16px}figure{margin:0;flex:1;min-width:230px}"
        "img{display:block;width:100%;max-height:450px;object-fit:contain;"
        "background:#222;border-radius:8px}figcaption{text-align:center;"
        "margin-top:6px;color:#b6c3db}p{line-height:1.4}</style>"
        "<h1>ODIN — 9 divergências da validação</h1>"
        "<p>Rótulos das pastas NÃO são verdade física. Examine a peça e "
        "classifique apenas a qualidade OK/NG; não inferir defeito pelo nome. "
        "Não é novo teste cego.</p>" + "".join(html_cases) + "</html>"
    )
    pointer_after = sha256(pointer.read_bytes()).hexdigest() if pointer.exists() else None
    if pointer_after != pointer_before:
        raise ValueError("Ponteiro de produção mudou durante a auditoria")
    if sha256(candidate.read_bytes()).hexdigest() != info["candidate_sha256"]:
        raise ValueError("Checkpoint mudou durante a auditoria")
    record = {
        "schema": QUEUE_SCHEMA,
        "created_at_utc": now.isoformat(),
        "source_checkpoint_sha256": info["candidate_sha256"],
        "source_manifest_sha256": checkpoint["source_manifest_sha256"],
        "total_validation": len(validation),
        "disagreements": len(items),
        "provisional_ng_predicted_ok": sum(
            x["folder_quality_provisional"] == "NG" for x in items
        ),
        "provisional_ok_predicted_ng": sum(
            x["folder_quality_provisional"] == "OK" for x in items
        ),
        "no_training_or_model_changes": True,
        "active_pointer_unchanged": pointer_after == pointer_before,
        "production_approved": False,
        "xp_commands_sent": 0,
        "cases": items,
    }
    zip_path = target / "odin_validation_visual_review.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("index.html", html)
        archive.writestr("cases.json", json.dumps(record, ensure_ascii=False, indent=2))
        for name, payload in assets.items():
            archive.writestr(name, payload)
    (target / "review_summary.json").write_text(
        json.dumps({"zip_path": str(zip_path), **{k: v for k, v in record.items()
              if k != "cases"}}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return {"archive": str(zip_path), "validation_disagreements": len(items),
            "provisional_ng_to_ok": record["provisional_ng_predicted_ok"],
            "provisional_ok_to_ng": record["provisional_ok_predicted_ng"],
            "candidate_only": True, "production_approved": False}


def main(argv=None):
    p = argparse.ArgumentParser(description="Auditoria visual 9 divergências CNN, offline")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--repair-report", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    args = p.parse_args(argv)
    print(json.dumps(build_visual_queue(
        args.root, args.repair_report, args.manifest,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
