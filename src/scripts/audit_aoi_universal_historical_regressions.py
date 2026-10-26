"""Audit the two NG -> OK regressions in the historical test, offline only.

Replays identical original 106-source fold against the preceding and the
24-operator candidate. Generates an SHA-pinned visual ZIP for operator review.
The old test has already been examined: it is diagnostic, NOT a blind holdout.
No training, promotion, live checkpoint or XP modifications.
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
from src.scripts.audit_aoi_universal_operator_confirmed import _original_manifest
from src.scripts.train_aoi_universal_24_human_repair import (
    REPORT_SCHEMA, SCHEMA, _load_prior,
)
from src.scripts.train_aoi_universal_hard_case_repair import _protected_split
from src.scripts.train_faltando_cnn_v2_online import _assess
from src.services.aoi_universal_quality_archive import universal_archive_events
from src.services.faltando_neural_qualification import QualificationStore
from src.services.odin_operator_confirmed_labels import verified_operator_labels
from src.services.odin_operator_validation_labels import verified_validation_labels

REPLAY_SCHEMA = "visionx.aoi_24_human_historical_regression_review.v1"


def _load_24(root: Path, report_name: Path) -> tuple[dict, Path, dict]:
    report_file = report_name if report_name.is_absolute() else root / report_name
    if report_file.is_symlink() or not report_file.is_file():
        raise ValueError("Relatório offline dos 24 casos indisponível")
    report_file = report_file.resolve()
    parent = (root / "reports/aoi_universal_quality/validation_repairs").resolve()
    if (report_file.parent.parent != parent
            or report_file.name != "24_operator_repair_report.json"):
        raise ValueError("Relatório fora da pasta autorizada")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA
            or report.get("candidate_only") is not True
            or report.get("production_approved") is not False
            or report.get("automatic_ok_authorized") is not False
            or report.get("eligible_for_production") is not False):
        raise ValueError("Relatório incompatível com experimento offline")
    relative = report.get("candidate_path")
    if (not isinstance(relative, str) or Path(relative).is_absolute()
            or ".." in Path(relative).parts):
        raise ValueError("Caminho do candidato novo inválido")
    model_path = root / relative
    if (model_path.is_symlink() or not model_path.is_file()
            or model_path.resolve().parent != report_file.parent
            or model_path.name != "aoi_24_operator_confirmed_candidate.pt"
            or sha256(model_path.read_bytes()).hexdigest() !=
               report.get("candidate_sha256")):
        raise ValueError("Checkpoint novo diverge do relatório")
    model = torch.load(model_path, map_location="cpu", weights_only=True)
    if (not isinstance(model, dict) or model.get("schema") != SCHEMA
            or model.get("architecture_schema") != MODEL_SCHEMA_V2
            or model.get("candidate_only") is not True
            or model.get("production_approved") is not False
            or model.get("automatic_ok_authorized") is not False
            or model.get("operator_supervised_captures") != 24
            or model.get("source_candidate_sha256") !=
               report.get("previous_candidate_sha256")
            or model.get("source_manifest_sha256") !=
               report.get("source_manifest_sha256")):
        raise ValueError("Metadados do candidato novo não conferem")
    return report, model_path, model


def _identify_regressions(records: list[dict], before: dict, after: dict,
                          expected_report: dict) -> list[tuple[dict, dict, dict]]:
    prior = before.get("cases", [])
    current = after.get("cases", [])
    if len(records) != 106 or len(prior) != 106 or len(current) != 106:
        raise ValueError("Diferente das 106 capturas do teste histórico")
    changed = []
    previous_counts, current_counts = Counter(), Counter()
    for record, first, second in zip(records, prior, current):
        for row in (first, second):
            if (row.get("event_id_or_group_hint") != record["id"]
                    or row.get("label") != record["label"]
                    or row.get("predicted") not in ("OK", "NG")):
                raise ValueError("Predição não corresponde à ordem do teste histórico")
        previous_counts[record["label"], first["predicted"]] += 1
        current_counts[record["label"], second["predicted"]] += 1
        if first["predicted"] == record["label"] and second["predicted"] != record["label"]:
            changed.append((record, first, second))
    historical = expected_report.get("historical_test_not_blind", {})
    if (previous_counts["NG", "OK"] != 0
            or previous_counts["OK", "NG"] != 0
            or len(changed) != 2
            or current_counts["NG", "OK"] != 2
            or current_counts["OK", "NG"] != 0
            or historical.get("total") != 106
            or historical.get("NG_false_OK") != 2
            or historical.get("OK_false_NG") != 0
            or any(event["label"] != "NG" for event, _, _ in changed)):
        raise ValueError("Resultados não reproduzem exatamente as duas regressões NG")
    return changed


def _readonly_pointer_digest(root: Path) -> str | None:
    pointer = root / "reports/neural_online/live_active.json"
    if pointer.is_symlink():
        raise ValueError("Ponteiro ativo é symlink")
    return sha256(pointer.read_bytes()).hexdigest() if pointer.is_file() else None


def inspect_historical_regressions(root: Path, new_report: Path,
                                   previous_report: Path, manifest: Path) -> dict:
    root = Path(root).resolve()
    original, previous_path, first_chk = _load_prior(root, Path(previous_report))
    report, new_path, second_chk = _load_24(root, Path(new_report))
    if (original["candidate_sha256"] != report["previous_candidate_sha256"]
            or report["source_manifest_sha256"] != first_chk["source_manifest_sha256"]):
        raise ValueError("Os dois candidatos não possuem a mesma origem")
    manifest_path, _ = _original_manifest(
        root, Path(manifest), report["source_manifest_sha256"],
    )
    store = QualificationStore(manifest_path)
    human = verified_operator_labels(root)
    second_human = verified_validation_labels(store)
    if set(human) & set(second_human):
        raise ValueError("Revisões humanas sobrepostas")
    human.update(second_human)
    pointer_before = _readonly_pointer_digest(root)
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    events, _ = universal_archive_events(manifest_path)
    seed = report.get("split", {}).get("original_split", {}).get("seed", 42)
    if seed != 42:
        raise ValueError("Split incompatível com a experiência registrada")
    _, _, test, original_split = _protected_split(events, human, seed=seed)
    if (len(test) != 106
            or [x["id"] for x in test] !=
               original_split["original_split"]["folds"]["test"]["source_ids"]
            or report.get("split", {}).get("old_test_untouched_records") != 106):
        raise ValueError("O teste original não foi reconstruído exatamente")
    first, second = FaltandoCNNV2(), FaltandoCNNV2()
    first.load_state_dict(first_chk["state_dict"], strict=True)
    second.load_state_dict(second_chk["state_dict"], strict=True)
    if (first_chk["image_size"] != second_chk["image_size"]
            or first_chk["focus_fraction"] != second_chk["focus_fraction"]):
        raise ValueError("Pré-processamentos não comparáveis")
    size, focus = first_chk["image_size"], first_chk["focus_fraction"]
    first_eval = _assess(first, test, size, focus)
    second_eval = _assess(second, test, size, focus)
    regressions = _identify_regressions(test, first_eval, second_eval, report)

    now = datetime.now(timezone.utc)
    folder = root / "reports/aoi_universal_quality/historical_regressions" / (
        now.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:6]
    )
    folder.mkdir(parents=True, exist_ok=False)
    assets = {}
    cases, html = [], []
    for index, (record, old, new) in enumerate(regressions, 1):
        if len(record["sources"]) != 1 or len(record["views"]) != 1:
            raise ValueError("Não foi possível associar um único par de imagens")
        source = record["sources"][0]
        source_file = root / source
        if (source_file.is_symlink() or not source_file.is_file()
                or sha256(source_file.read_bytes()).hexdigest() !=
                   record["source_sha256"]):
            raise ValueError("Captura original NG mudou: " + source)
        light, pair = next(iter(record["views"].items()))
        file_map = {}
        digest_map = {}
        for kind, local in (
            ("original", source_file),
            ("reference", Path(pair["reference"])),
            ("test", Path(pair["test"])),
        ):
            local = Path(local)
            if (local.is_symlink() or not local.is_file()
                    or not local.resolve().is_relative_to(root)):
                raise ValueError("Arquivo de par fora da pasta da estação")
            data = local.read_bytes()
            relative = f"images/{index:02d}_{kind}.png"
            assets[relative] = data
            file_map[kind] = relative
            digest_map[kind] = sha256(data).hexdigest()
        cases.append({
            "case": index, "source_path": source,
            "source_sha256": record["source_sha256"],
            "archive_provisional_label": "NG",
            "operator_confirmed_quality": None,
            "aoi_hint_not_ground_truth": record.get("category_hint"),
            "lighting": light,
            "previous_prediction": old["predicted"],
            "previous_ng_score": old["ng_score"],
            "new_prediction": new["predicted"],
            "new_ng_score": new["ng_score"],
            "human_review_status": "PENDING",
            "files": file_map, "file_sha256": digest_map,
        })
        html.append(
            f"<section><h2>Caso {index} — {escape(Path(source).name)}</h2>"
            f"<p>Arquivo: NG (provisório). Antes: <b>{old['predicted']}</b>"
            f" (score NG {old['ng_score']:.6f}); agora: <b>{new['predicted']}</b>"
            f" ({new['ng_score']:.6f}). Iluminação: {escape(light)}</p>"
            "<div>" + "".join(
                f"<figure><img src='{file_map[kind]}'><figcaption>{title}</figcaption></figure>"
                for kind, title in (
                    ("original", "Original AOI"),
                    ("reference", "Gabarito"),
                    ("test", "Teste"),
                )
            ) + "</div><p>Qualidade física: OK / NG / INCERTO — a confirmar</p></section>"
        )
    visual = (
        "<!doctype html><html lang='pt-BR'><meta charset='utf-8'>"
        "<title>ODIN — NG históricos que passaram a OK</title>"
        "<style>body{font:15px system-ui;background:#111827;color:#eee;"
        "max-width:1200px;margin:auto;padding:24px}section{border:1px solid #536078;"
        "padding:20px;border-radius:12px;margin:20px 0}div{display:flex;flex-wrap:wrap;"
        "gap:12px}figure{min-width:210px;flex:1;margin:0}img{width:100%;"
        "max-height:500px;object-fit:contain;background:#242424}figcaption{"
        "text-align:center;margin:6px}</style>"
        "<h1>ODIN — duas regressões NG→OK</h1>"
        "<p>Conjunto histórico já avaliado anteriormente, NÃO é teste cego. "
        "Rótulos da AOI são provisórios; confirme fisicamente cada caso.</p>"
        + "".join(html) + "</html>"
    )
    if (_readonly_pointer_digest(root) != pointer_before
            or sha256(previous_path.read_bytes()).hexdigest() !=
               original["candidate_sha256"]
            or sha256(new_path.read_bytes()).hexdigest() !=
               report["candidate_sha256"]
            or sha256(manifest_path.read_bytes()).hexdigest() !=
               report["source_manifest_sha256"]):
        raise ValueError("Ponteiro/pesos/manifesto alterados durante auditoria")
    summary = {
        "schema": REPLAY_SCHEMA, "created_at_utc": now.isoformat(),
        "new_checkpoint_sha256": report["candidate_sha256"],
        "previous_checkpoint_sha256": original["candidate_sha256"],
        "historic_test_count": 106, "previous_errors": 0,
        "new_false_ok_ng": 2, "new_false_ng_ok": 0,
        "regressions_found": 2, "physical_label_independently_checked": False,
        "archived_test_already_seen": True, "candidate_only": True,
        "production_approved": False, "training_performed": False,
        "xp_commands_sent": 0, "live_pointer_unchanged": True,
        "cases": cases,
    }
    archive_name = folder / "odin_two_historical_ng_regressions.zip"
    with ZipFile(archive_name, "w", compression=ZIP_DEFLATED, compresslevel=6) as zip_file:
        zip_file.writestr("index.html", visual)
        zip_file.writestr("cases.json", json.dumps(summary, indent=2, ensure_ascii=False))
        for name, blob in assets.items():
            zip_file.writestr(name, blob)
    return {
        "zip_path": str(archive_name),
        "regressed_NG_archive_cases": len(regressions),
        "previous_test_correct": 106,
        "new_test_correct": 104,
        "production_approved": False,
        "cases": [{"source": x["source_path"], "prior_ng_score": x["previous_ng_score"],
                   "new_ng_score": x["new_ng_score"]} for x in cases],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Auditar 2 regressões históricas NG→OK; nunca ativa modelo")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--new-report", type=Path, required=True)
    p.add_argument("--previous-report", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args(argv)
    print(json.dumps(inspect_historical_regressions(
        args.root, args.new_report, args.previous_report, args.manifest,
    ), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
