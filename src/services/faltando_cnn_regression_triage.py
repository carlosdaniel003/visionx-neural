"""Triagem não mutante de regressões da CNN FALTANDO V2.

Lê o relatório completo do replay pareado, aplica gate de não regressão
para OK/NG/revisões, gera fila de conferência HUMANA dos casos cuja decisão
piorou. Não treina e NUNCA autoriza promoção por conta própria.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

SCHEMA = "visionx.faltando_cnn_regression_triage.v1"
SOURCE_SCHEMA = "visionx.faltando_candidate_archive_comparison.v1"
SCOPE = "READ_ONLY_ARCHIVE_REPLAY_NOT_HOLDOUT"
LABELS = {"OK", "NG"}
STATES = {"PASSOU", "ERRO_CLASSIFICACAO", "REVISAO"}
LIGHTS = {"SIDE", "TOP", "MID"}


def _counts(model: dict) -> dict:
    if not isinstance(model, dict):
        raise ValueError("Resumo de CNN não encontrado")
    keys = (
        "eligible", "passed", "incorrect", "review", "invalid",
        "no_coverage", "not_evaluated", "ng_as_ok", "ok_as_ng",
    )
    if any(type(model.get(k)) is not int or model[k] < 0 for k in keys):
        raise ValueError("Contadores ausentes ou inválidos")
    if (sum(model[k] for k in (
            "passed", "incorrect", "review", "invalid",
            "no_coverage", "not_evaluated"
        )) != model["eligible"]):
        raise ValueError("Contadores de imagens elegíveis inconsistentes")
    return {k: model[k] for k in keys}


def triage_comparison(report: dict) -> dict:
    """Retorna diagnósticos sem confiar em status de aprovação de treino."""
    if (not isinstance(report, dict)
            or report.get("schema") != SOURCE_SCHEMA
            or report.get("mode") != "READ_ONLY_FULL_ARCHIVE_REPLAY_COMPARISON"
            or report.get("candidate_only") is not True
            or report.get("active_model_changed") is not False
            or report.get("production_approved") is not False
            or report.get("holdout_independent") is not False):
        raise ValueError("Origem não é comparativo offline completo e não mutante")

    data = report.get("comparison") or {}
    baseline = _counts(data.get("baseline"))
    candidate = _counts(data.get("candidate"))
    if (baseline["eligible"] != candidate["eligible"]
            or baseline["eligible"] <= 0
            or baseline["invalid"] or candidate["invalid"]
            or baseline["no_coverage"] or candidate["no_coverage"]
            or baseline["not_evaluated"] or candidate["not_evaluated"]):
        raise ValueError("Acervo incompleto ou com linhas inválidas")

    diffs = data.get("case_changes")
    if not isinstance(diffs, list):
        raise ValueError("Comparativo sem casos individuais alterados")

    unique_sources = set()
    by_category = defaultdict(Counter)
    queue = []
    new_bad_ok = 0
    new_bad_ng = 0
    new_reviews_on_pass = 0
    ng_recovered = 0
    for entry in diffs:
        if not isinstance(entry, dict):
            raise ValueError("Caso de regressão inválido")
        source = entry.get("source_path")
        category = entry.get("category")
        light = entry.get("lighting_mode")
        expected = entry.get("expected_label")
        old = entry.get("baseline_status")
        new = entry.get("candidate_status")
        if (not isinstance(source, str) or not source.startswith("public/")
                or source in unique_sources
                or not isinstance(category, str) or not category
                or light not in LIGHTS or expected not in LABELS
                or old not in STATES or new not in STATES
                or old == new):
            raise ValueError(f"Transição/identidade inválida: {source!r}")
        if (("/ok_archive/" in source) != (expected == "OK")
                or ("/ng_archive/" in source) != (expected == "NG")):
            raise ValueError("Pasta e rótulo da imagem divergem")
        unique_sources.add(source)
        by_category[category][f"{expected}:{old}->{new}"] += 1

        if expected == "OK" and old != "ERRO_CLASSIFICACAO" and new == "ERRO_CLASSIFICACAO":
            new_bad_ok += 1
        if expected == "NG" and old != "ERRO_CLASSIFICACAO" and new == "ERRO_CLASSIFICACAO":
            new_bad_ng += 1
        if old == "PASSOU" and new == "REVISAO":
            new_reviews_on_pass += 1
        if expected == "NG" and old == "ERRO_CLASSIFICACAO" and new in {"PASSOU", "REVISAO"}:
            ng_recovered += 1

        # Rotular com "OK" apenas porque está na pasta seria vazamento:
        # só o operador pode confirmar a proveniência da peça real.
        if expected == "OK" and old in {"PASSOU", "REVISAO"} and new in {"REVISAO", "ERRO_CLASSIFICACAO"}:
            queue.append({
                "source_path": source,
                "category": category,
                "lighting_mode": light,
                "archive_label": "OK",
                "baseline_status": old,
                "candidate_status": new,
                "baseline_ng_score_uncalibrated": entry.get("baseline_ng_score"),
                "candidate_ng_score_uncalibrated": entry.get("candidate_ng_score"),
                "human_verification": "PENDING",
                "verified_physical_event_id": None,
                "training_authorized": False,
                "note": "Conferir recorte, imagem original e rótulo com o operador.",
            })

    # O detalhamento só contém TRANSIÇÕES. Não confundir sua contagem
    # com o total de imagens de cada categoria do acervo.
    if (data.get("ng_false_ok_baseline") != baseline["ng_as_ok"]
            or data.get("ng_false_ok_candidate") != candidate["ng_as_ok"]
            or data.get("ok_false_ng_baseline") != baseline["ok_as_ng"]
            or data.get("ok_false_ng_candidate") != candidate["ok_as_ng"]
            or data.get("reviews_baseline") != baseline["review"]
            or data.get("reviews_candidate") != candidate["review"]):
        raise ValueError("Contadores do comparativo divergem de seus resumos")
    if candidate["ok_as_ng"] - baseline["ok_as_ng"] != new_bad_ok:
        raise ValueError("Falsos NG novos não coincidem com mudanças por imagem")
    if candidate["ng_as_ok"] > baseline["ng_as_ok"] and new_bad_ng == 0:
        raise ValueError("Novos falsos OK não identificados por caso")
    if new_reviews_on_pass > candidate["review"]:
        raise ValueError("Revisões novas excedem o total de revisões")

    checks = {
        "no_ng_auto_ok_remaining": candidate["ng_as_ok"] == 0,
        "no_new_ok_false_ng": new_bad_ok == 0,
        "no_new_ng_false_ok": new_bad_ng == 0,
        "no_new_review_for_previously_passed": new_reviews_on_pass == 0,
        "no_loss_of_conclusive_passes": candidate["passed"] >= baseline["passed"],
        "all_eligible_completed": (
            baseline["eligible"] == candidate["eligible"]
            and baseline["invalid"] == candidate["invalid"] == 0
        ),
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "mode": SCOPE,
        "eligible_images": candidate["eligible"],
        "baseline": baseline,
        "candidate": candidate,
        "new_ok_false_ng": new_bad_ok,
        "new_ng_false_ok": new_bad_ng,
        "new_reviews_from_previously_passed": new_reviews_on_pass,
        "recovered_ng_false_ok_as_ng_or_review": ng_recovered,
        "changes_by_category": {
            cat: dict(counts) for cat, counts in sorted(by_category.items())
        },
        "required_human_review": queue,
        "required_human_review_count": len(queue),
        "regression_checks": checks,
        "regression_gate_passed": passed,
        "candidate_rejected_by_archive_gate": not passed,
        "can_consider_for_independent_testing": passed,
        "production_approved": False,
        "independent_holdout_evaluated": False,
        "training_uses_archive_labels": False,
        "notes": [
            "O acervo de replay inclui imagens de treinamento; não é holdout.",
            "Nenhuma imagem OK piorada é treinada automaticamente.",
            "Uma captura SIDE e uma TOP podem ser a mesma peça física; agrupar somente após conferir IDs reais.",
            "Mesmo gate de regressão aprovado não autoriza produção.",
        ],
    }


def triage_report(root: Path | str, report_file: Path | str) -> tuple[Path, dict]:
    """Analisa arquivo EXISTENTE e grava JSON somente em reports/."""
    root = Path(root).resolve()
    folder = (root / "reports/startup_regression/cnn_candidate_validation").resolve()
    report_file = Path(report_file)
    if not report_file.is_absolute():
        report_file = root / report_file
    if (report_file.is_symlink() or not report_file.is_file()
            or report_file.suffix.lower() != ".json"
            or report_file.resolve().parent != folder
            or not report_file.name.startswith("comparacao_acervo_cnn_peso050_")):
        raise ValueError("Relatório fora do diretório de avaliação permitida")
    source_bytes = report_file.read_bytes()
    obj = json.loads(source_bytes.decode("utf-8"))
    result = triage_comparison(obj)
    from hashlib import sha256
    result.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_file.relative_to(root)),
        "source_report_sha256": sha256(source_bytes).hexdigest(),
    })
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (
        "triagem_regressoes_cnn_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + "_" + uuid4().hex[:6] + ".json"
    )
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path, result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnosticar regressões CNN em relatório já calculado"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    path, result = triage_report(args.root, args.report)
    print(f"Relatorio: {path}", flush=True)
    print(
        f"GATE: {'REPROVADO' if result['candidate_rejected_by_archive_gate'] else 'APROVADO SOMENTE PARA TESTES'}",
        flush=True,
    )
    print(
        f"Novos falsos NG em OK: {result['new_ok_false_ng']}; "
        f"novas revisoes em acertos: {result['new_reviews_from_previously_passed']}; "
        f"casos a conferir: {result['required_human_review_count']}",
        flush=True,
    )
    print("NENHUM CHECKPOINT ATIVADO. NENHUMA IMAGEM TREINADA.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
