"""Comparação offline, somente leitura, CNN ativa vs. candidato 0,50.

NÃO é holdout independente. Reexecuta o replay histórico do arquivo OK/NG,
incluindo amostras de treinamento. Reusa o inspetor operacional, inclusive a
faixa de REVISÃO [0,10; 0,90]. Nunca ativa checkpoint, treina ou envia 0/1.
Executar somente na janela de manutenção, sem receptor XP.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from uuid import uuid4

from src.core.neural.faltando_live import (
    FaltandoCNNLive, PINNED_RELATIVE_CHECKPOINT, PINNED_CHECKPOINT_SHA256,
)
from src.services.faltando_cnn_recovery_preflight import validate_recovery_inputs
from src.services.faltando_cnn_regression_triage import triage_comparison
from src.services.startup_regression.archive_inventory import inventory_archives
from src.services.startup_regression.interactive_training_validation import (
    configure_low_priority, save_report, validate,
)
from src.services.startup_regression.archive_inventory_report import _atomic_text

SCHEMA = "visionx.faltando_candidate_archive_comparison.v1"
CORRECTION_ID = "20261013T175115Z_ebf986ef"
COMPARISON_FILE = "comparacao_pesos_iluminacao_20261013T200852_765974Z_d7b0d1.json"
CANDIDATE_SHA = "9b94d40b46f9bd1b1930e5ee405244da43e7ad688a2fb473c492947b78271107"
WEIGHT = 0.50
EPOCHS = 7
CNN = "CNN_FALTANDO_V2"
LABELS = ("OK", "NG")
LIGHTS = ("SIDE", "TOP", "MID")


def _read_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Arquivo ausente ou link simbólico: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"JSON inválido: {path}")
    return obj


def _relative_path(root: Path, relative: str, parent: Path, suffix: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("Referência de arquivo vazia")
    value = Path(relative.replace("\\", "/"))
    if (value.is_absolute() or ".." in value.parts or ":" in value.parts[0]
            or value.suffix.lower() != suffix):
        raise ValueError("Referência de arquivo fora do contrato")
    path = root / value
    if path.is_symlink() or not path.is_file() or path.resolve().parent != parent.resolve():
        raise ValueError(f"Arquivo candidato fora da pasta esperada: {path}")
    return path


def candidate_preflight(root: Path | str) -> dict:
    """Vincula EXATAMENTE o candidato 0,50 já avaliado a SHA e transação."""
    root = Path(root).resolve()
    verified = validate_recovery_inputs(root, CORRECTION_ID)
    path = (root / "reports/neural_online/recovery_comparisons" / COMPARISON_FILE)
    report = _read_json(path)
    if (report.get("schema")
            != "visionx.faltando_recovery_light_weight_comparison.v1"
            or report.get("correction_id") != CORRECTION_ID
            or report.get("epochs_per_experiment") != EPOCHS
            or report.get("independent_from_original_checkpoint") is not True
            or report.get("candidate_only") is not True
            or report.get("active_model_changed") is not False
            or report.get("production_approved") is not False
            or report.get("automatic_selection") is not False
            or report.get("original_checkpoint_sha256") != PINNED_CHECKPOINT_SHA256
            or set(report.get("verified_event_ids", [])) !=
               {p.stem for p in verified["events"]}):
        raise ValueError("Comparativo original não atende ao contrato de auditoria")
    selected = (report.get("experiments") or {}).get("0.50") or {}
    internal = selected.get("correction_only") or {}
    historic = selected.get("historical") or {}
    online = selected.get("online_all_events") or {}
    gates = selected.get("gates") or {}
    if (selected.get("epochs") != EPOCHS
            or selected.get("light_loss_weight") != WEIGHT
            or selected.get("passed_internal_gates") is not True
            or selected.get("promoted") is not False
            or selected.get("candidate_sha256") != CANDIDATE_SHA
            or internal.get("images_correct") != 9
            or internal.get("images_total") != 9
            or internal.get("correct_by_light") != {light: 3 for light in LIGHTS}
            or historic.get("correct_images") != 117
            or historic.get("total_images") != 117
            or historic.get("ng_false_ok") != 0
            or online.get("correct_images") != 9
            or online.get("total_images") != 9
            or online.get("ng_false_ok") != 0
            or gates.get("historical_unchanged") is not True
            or gates.get("all_confirmed_online_correct") is not True
            or gates.get("no_new_historical_false_ok") is not True):
        raise ValueError("Candidato 0,50 não corresponde ao experimento aprovado internamente")
    candidate = _relative_path(
        root, selected.get("candidate_path"),
        root / "reports/neural_online/recovery_candidates", ".pt",
    )
    if sha256(candidate.read_bytes()).hexdigest() != CANDIDATE_SHA:
        raise ValueError("SHA-256 do candidato não corresponde ao modelo verificado")
    training_report = _relative_path(
        root, selected.get("report_path"),
        root / "reports/neural_online/training_reports", ".json",
    )
    trial = _read_json(training_report)
    if (trial.get("candidate_sha256") != CANDIDATE_SHA
            or trial.get("candidate_only") is not True
            or trial.get("promoted") is not False
            or trial.get("recovery_correction_id") != CORRECTION_ID
            or trial.get("light_loss_weight") != WEIGHT
            or trial.get("base_checkpoint_sha256") != PINNED_CHECKPOINT_SHA256):
        raise ValueError("Relatório individual diverge do comparativo")
    pinned = root / PINNED_RELATIVE_CHECKPOINT
    if (pinned.is_symlink() or not pinned.is_file()
            or sha256(pinned.read_bytes()).hexdigest() != PINNED_CHECKPOINT_SHA256):
        raise ValueError("Checkpoint original alterado ou ausente")
    pointer = root / "reports/neural_online/live_active.json"
    if pointer.exists() or pointer.is_symlink():
        raise ValueError("Há checkpoint online ativo: bloquear para auditoria")
    return {
        "candidate_path": candidate, "candidate_sha256": CANDIDATE_SHA,
        "baseline_path": pinned, "baseline_sha256": PINNED_CHECKPOINT_SHA256,
        "comparison_path": path, "training_report_path": training_report,
        "correction_id": CORRECTION_ID, "light_loss_weight": WEIGHT,
    }


def _inventory_signature(inventory: dict) -> list[tuple]:
    images = inventory.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Acervo vazio ou inválido")
    return sorted((
        x.get("path"), x.get("file_sha256"), x.get("status"),
        x.get("expected_label"), x.get("category_hint"),
        x.get("lighting_mode"),
    ) for x in images)


def _cnn_rows(result: dict, expected_sha: str) -> dict[str, dict]:
    if (result.get("mode") != "READ_ONLY_ARCHIVE_REPLAY"
            or result.get("complete") is not True
            or result.get("canceled") is not False
            or result.get("processed_model_checks") != result.get("planned_model_checks")
            or not result.get("archive_png_count")
            or result.get("archive_png_count") * 2 != len(result.get("cases", []))):
        raise ValueError("Validação do acervo incompleta ou incompatível")
    rows = {}
    for case in result["cases"]:
        if case.get("model") != CNN:
            continue
        key = case.get("source_path")
        if not isinstance(key, str) or key in rows:
            raise ValueError("Fonte de CNN duplicada/sem caminho no acervo")
        if (case.get("status") != "FORA_ESCOPO"
                and case.get("checkpoint_sha256") not in (None, expected_sha)):
            raise ValueError("Inferência misturou checkpoints diferentes")
        rows[key] = case
    if len(rows) != result["archive_png_count"]:
        raise ValueError("CNN não abrangeu os registros do inventário")
    summary = result.get("models", {}).get(CNN) or {}
    if summary.get("fully_evaluated") is not True:
        raise ValueError("CNN possui linhas não avaliadas")
    return rows


def compare_archive_results(baseline: dict, candidate: dict,
                            baseline_sha: str, candidate_sha: str) -> dict:
    before = _cnn_rows(baseline, baseline_sha)
    after = _cnn_rows(candidate, candidate_sha)
    if set(before) != set(after):
        raise ValueError("Acervos comparados não têm os mesmos caminhos")
    transitions = Counter()
    changes = []
    by_light = {
        light: {"baseline": Counter(), "candidate": Counter()}
        for light in LIGHTS
    }
    by_label = {
        label: {"baseline": Counter(), "candidate": Counter()}
        for label in LABELS
    }
    for source in sorted(before):
        old, new = before[source], after[source]
        if any(old.get(k) != new.get(k) for k in (
            "source_path", "category", "lighting_mode", "expected_label"
        )):
            raise ValueError("Identidade do caso alterada entre checkpoints")
        state = (old["status"], new["status"])
        transitions[" -> ".join(state)] += 1
        if old.get("expected_label") in LABELS and old["status"] != "FORA_ESCOPO":
            by_label[old["expected_label"]]["baseline"][old["status"]] += 1
            by_label[old["expected_label"]]["candidate"][new["status"]] += 1
            mode = old.get("lighting_mode")
            if mode in LIGHTS:
                by_light[mode]["baseline"][old["status"]] += 1
                by_light[mode]["candidate"][new["status"]] += 1
        if (old["status"] != new["status"] or
                old.get("predicted_label") != new.get("predicted_label")):
            changes.append({
                "source_path": source,
                "category": old.get("category"),
                "lighting_mode": old.get("lighting_mode"),
                "expected_label": old.get("expected_label"),
                "baseline_status": old["status"],
                "candidate_status": new["status"],
                "baseline_predicted": old.get("predicted_label"),
                "candidate_predicted": new.get("predicted_label"),
                "baseline_ng_score": old.get("cnn_score_ng"),
                "candidate_ng_score": new.get("cnn_score_ng"),
                "candidate_error": new.get("error"),
            })
    old_model, new_model = (
        baseline["models"][CNN], candidate["models"][CNN],
    )
    return {
        "baseline": old_model,
        "candidate": new_model,
        "status_transitions": dict(sorted(transitions.items())),
        "by_expected_label": {
            k: {model: dict(counts) for model, counts in states.items()}
            for k, states in by_label.items()
        },
        "by_lighting_mode": {
            k: {model: dict(counts) for model, counts in states.items()}
            for k, states in by_light.items()
        },
        "case_changes": changes,
        "ng_false_ok_baseline": old_model["ng_as_ok"],
        "ng_false_ok_candidate": new_model["ng_as_ok"],
        "ok_false_ng_baseline": old_model["ok_as_ng"],
        "ok_false_ng_candidate": new_model["ok_as_ng"],
        "reviews_baseline": old_model["review"],
        "reviews_candidate": new_model["review"],
        "historical_replay_only": True,
    }


def validate_candidate_on_archive(root: Path | str, *,
                                  preflight=None, inventory_fn=None,
                                  validator=None, model_factory=None) -> dict:
    """Duas inferências reais, mesmo inventário; apenas grava relatórios."""
    root = Path(root).resolve()
    config = (preflight or candidate_preflight)(root)
    inventory_fn = inventory_fn or inventory_archives
    validator = validator or validate
    model_factory = model_factory or FaltandoCNNLive
    original_pointer = root / "reports/neural_online/live_active.json"
    begin = time.monotonic()
    inventory = inventory_fn(root)
    snapshot = _inventory_signature(inventory)
    reports = root / "reports/startup_regression/cnn_candidate_validation"
    outputs = {}
    results = {}
    for label in ("baseline", "candidate"):
        # Integridade e imutabilidade entre as duas leituras.
        if original_pointer.exists() or original_pointer.is_symlink():
            raise ValueError("O ponteiro online foi ativado durante a validação")
        current = (preflight or candidate_preflight)(root)
        if (current["baseline_sha256"] != config["baseline_sha256"]
                or current["candidate_sha256"] != config["candidate_sha256"]):
            raise ValueError("Hashes mudaram durante a validação")
        checkpoint = config[f"{label}_path"]
        digest = config[f"{label}_sha256"]
        cnn = model_factory(
            checkpoint_path=checkpoint, expected_sha256=digest, online_root=root,
        )
        print(f"\nVALIDAÇÃO DE ARQUIVO: {label.upper()} / {digest}", flush=True)

        def progress(event):
            if event.get("event") == "progress":
                done = event.get("done", 0)
                if done % 40 == 0 or done == event.get("total"):
                    print(
                        f"{label.upper()} {done}/{event.get('total')} checks",
                        flush=True,
                    )

        result = validator(root, inventory=inventory, cnn=cnn, progress=progress)
        _cnn_rows(result, digest)
        new_snapshot = _inventory_signature(inventory_fn(root))
        if new_snapshot != snapshot:
            raise ValueError("Imagens do acervo foram modificadas durante a validação")
        if (original_pointer.exists() or original_pointer.is_symlink()
                or sha256(checkpoint.read_bytes()).hexdigest() != digest):
            raise ValueError("Checkpoint/pointer mudou na análise")
        result["comparison_role"] = label
        output_json, output_txt = save_report(result, reports)
        results[label] = result
        outputs[label] = {
            "json_path": str(output_json.relative_to(root)),
            "txt_path": str(output_txt.relative_to(root)),
            "checkpoint_sha256": digest,
        }
    comparison = compare_archive_results(
        results["baseline"], results["candidate"],
        config["baseline_sha256"], config["candidate_sha256"],
    )
    if (original_pointer.exists() or original_pointer.is_symlink()
            or _inventory_signature(inventory_fn(root)) != snapshot):
        raise ValueError("Acervo ou ponteiro ativo mudou antes do relatório final")
    output = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - begin, 3),
        "correction_id": config["correction_id"],
        "candidate_light_loss_weight": WEIGHT,
        "candidate_train_epochs": EPOCHS,
        "archive_png_count": results["baseline"]["archive_png_count"],
        "processed_model_checks_each": results["baseline"]["processed_model_checks"],
        "mode": "READ_ONLY_FULL_ARCHIVE_REPLAY_COMPARISON",
        "candidate_only": True,
        "active_model_changed": False,
        "production_approved": False,
        "holdout_independent": False,
        "source_training_comparison": str(config["comparison_path"].relative_to(root)),
        "checkpoints": outputs,
        "comparison": comparison,
        "notes": [
            "Comparação replay de arquivo, não teste cego nem generalização.",
            "Rótulos são os das pastas OK/NG. Parte do arquivo integra o treino.",
            "Status REVISAO é distinto de PASSOU e ERRO_CLASSIFICACAO.",
            "O score 0.5 do treino não corresponde aos limites operacionais 0.10/0.90.",
            "Nenhum arquivo de KNN, dataset, checkpoint ativo ou comando XP alterado.",
            "Coletar eventos inéditos, de peças independentes, antes de promoção.",
        ],
    }
    # Gate de segurança cobre TAMBÉM os OK do acervo integral,
    # ausentes dos gates do treino incremental (117 imagens).
    # Um score correto em 9 NG de treino não justifica sete novos falsos NG.
    output["regression_triage"] = triage_comparison(output)
    print(
        "GATE ACERVO: " +
        ("REPROVADO" if output["regression_triage"]["candidate_rejected_by_archive_gate"]
         else "SEM REGRESSAO (AINDA NAO AUTORIZADO)") +
        f"; novos falsos NG={output['regression_triage']['new_ok_false_ng']}",
        flush=True,
    )
    output_dir = root / "reports/startup_regression/cnn_candidate_validation"
    name = "comparacao_acervo_cnn_peso050_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    ) + "_" + uuid4().hex[:6] + ".json"
    path = output_dir / name
    _atomic_text(path, json.dumps(output, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n")
    print(f"\nRELATÓRIO COMPARATIVO: {path}", flush=True)
    print("NENHUM CHECKPOINT PROMOVIDO. ACERVO HISTÓRICO NÃO É HOLDOUT.", flush=True)
    return {"report_path": str(path), **output}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Comparar CNN ativa e candidata 0,50 no acervo completo"
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    configure_low_priority()
    validate_candidate_on_archive(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
