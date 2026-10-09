"""Validação independente de CNNs sobre todo o arquivo OK/NG da AOI.

Sem KNN, MoE/especialistas físicos, treinamento, decisões XP ou mudanças
nas fotos. Não instalar como gate produtivo até ambas as CNNs existirem.

CNN FALTANDO v2: FALTANDO/EMBORCADO/INVERTIDO/DESLOCADO.
CNN MEMÓRIA: TODAS as categorias (adaptador ainda não presente na central).

O arquivo é avaliado por PNG/luz, sem inventar event_id a partir de nome.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.services.faltando_cross_category_audit import (
    _archive_image, _scored_result,
)
from src.services.startup_regression.archive_inventory import inventory_archives
from src.services.startup_regression.archive_inventory_report import _atomic_text

SCHEMA = "visionx.startup_cnn_archive_validation.v1"
MODEL_FALTANDO = "CNN_FALTANDO_V2"
MODEL_MEMORY = "CNN_MEMORIA"
MODEL_NAMES = (MODEL_FALTANDO, MODEL_MEMORY)
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
KNOWN_ADHESIVE = frozenset({"MUITO ADESIVO", "MUCH ADHESIVE", "ADESIVO"})
KNOWN_ABSENCE = frozenset({"FALTANDO", "EMBORCADO", "INVERTIDO", "DESLOCADO"})
TRUTH = {"OK": "FALHA FALSA", "NG": "DEFEITO REAL"}


def scope_for_model(name: str, category: str) -> str:
    """IN_SCOPE, OUT_OF_SCOPE ou UNKNOWN; não descarta casos desconhecidos."""
    canonical = str(category or "").strip().upper()
    if name == MODEL_MEMORY:
        return "IN_SCOPE"
    if name != MODEL_FALTANDO:
        raise ValueError(f"Modelo desconhecido: {name}")
    if uses_faltando_v2(canonical):
        return "IN_SCOPE"
    if canonical in KNOWN_ADHESIVE:
        return "OUT_OF_SCOPE"
    return "UNKNOWN"


def _result_row(name: str, item: dict, scope: str) -> dict:
    return {
        "model": name,
        "source_path": item.get("path"),
        "expected_label": item.get("expected_label"),
        "expected_verdict": TRUTH.get(item.get("expected_label")),
        "category_hint": item.get("category_hint"),
        "lighting_mode": item.get("lighting_mode"),
        "lighting_source": item.get("lighting_source"),
        "scope": scope,
        "status": "NOT_EVALUATED",
        "verdict": None,
        "ng_score_uncalibrated": None,
        "checkpoint_sha256": None,
        "error": None,
    }


def _check_model_response(name: str, output: dict) -> tuple[str, float | None, str | None]:
    if not isinstance(output, dict):
        raise ValueError("CNN não retornou um dicionário")
    if name == MODEL_FALTANDO:
        verdict, _, score = _scored_result(output)
        detail = output.get("detail") or {}
        if verdict != "REVISÃO OBRIGATÓRIA" and not detail.get(
            "cnn_v2_checkpoint_verified", False
        ):
            raise ValueError("Checkpoint CNN FALTANDO v2 não verificado")
        return verdict, score, detail.get("cnn_v2_checkpoint_sha256")
    if name == MODEL_MEMORY:
        detail = output.get("detail") or {}
        # Não confundir a recuperação de rótulos da KNN com inferência de CNN.
        if (
            not isinstance(detail, dict)
            or detail.get("model_kind") != "cnn_memoria"
            or detail.get("checkpoint_verified") is not True
        ):
            raise ValueError(
                "CNN MEMÓRIA sem contrato de modelo/checkpoint próprio; "
                "resultado KNN não é aceito"
            )
        verdict = str(output.get("verdict", "")).strip().upper()
        if verdict not in {"DEFEITO REAL", "FALHA FALSA", "REVISÃO OBRIGATÓRIA"}:
            raise ValueError("Veredito CNN MEMÓRIA inválido")
        score = detail.get("ng_score_uncalibrated")
        if score is not None:
            score = float(score)
            if not 0.0 <= score <= 1.0:
                raise ValueError("Score CNN MEMÓRIA inválido")
        digest = detail.get("checkpoint_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("SHA-256 CNN MEMÓRIA não informado")
        return verdict, score, digest
    raise ValueError("Nome de modelo inválido")


def validate_archive_cnns(
    root: Path, *,
    inventory: dict | None = None,
    extractor=None,
    predictors: dict | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Avalia cada PNG elegível de cada CNN separadamente; falha se faltar modelo.

    Na ausência de CNN MEMÓRIA, reporta explicitamente MODEL_UNAVAILABLE
    para todas suas imagens, sem fingir ter validado aquele modelo.
    """
    root = Path(root).expanduser().resolve()
    evidence = inventory if inventory is not None else inventory_archives(root)
    if evidence.get("schema") != "visionx.archive_inventory.v1":
        raise ValueError("Inventário incompatível com gate CNN")
    items = evidence.get("images")
    if not isinstance(items, list) or not items or (
        len(items) != evidence.get("summary", {}).get("png_count")
    ):
        raise ValueError("Inventário vazio/incompleto")
    predictor_map = dict(predictors or {})
    if MODEL_FALTANDO not in predictor_map:
        from src.core.neural.faltando_live import FaltandoCNNLive
        predictor_map[MODEL_FALTANDO] = FaltandoCNNLive()
    # CNN MEMÓRIA não pode ter fallback para KNN, Faltando ou simulação.
    predictor_map.setdefault(MODEL_MEMORY, None)

    conflicting = {
        path for group in evidence.get("cross_label_conflicts", [])
        for path in group.get("paths", [])
    }
    if extractor is None:
        from src.services.faltando_neural_dataset import AOIPairExtractor
        extractor = AOIPairExtractor()

    all_rows = []
    # Processa os modelos da MESMA imagem em sequência. Só mantém o par
    # da imagem corrente em RAM, mesmo quando o acervo crescer muito.
    planned = [
        (name, item, scope_for_model(name, item.get("category_hint")))
        for item in items for name in MODEL_NAMES
        if (name == MODEL_MEMORY or
            scope_for_model(name, item.get("category_hint")) != "OUT_OF_SCOPE")
    ]
    current_path = None
    current_pair = None
    current_extract_error = None
    # Um PNG corrompido não pode deixar seu teste desaparecido do denominador.
    for i, (name, item, scope) in enumerate(planned, start=1):
        row = _result_row(name, item, scope)
        model = predictor_map.get(name)
        try:
            if scope == "UNKNOWN":
                raise ValueError(
                    "Categoria não coberta pela CNN FALTANDO v2; qualificar imagem"
                )
            if item.get("status") != "VALID_PNG":
                raise ValueError("Imagem PNG inválida no inventário")
            if row["source_path"] in conflicting:
                raise ValueError("Mesmo PNG visual rotulado OK e NG")
            if model is None:
                row["status"] = MODEL_UNAVAILABLE
                row["error"] = "CNN MEMÓRIA não implementada/conectada na branch central"
            else:
                key = row["source_path"]
                if key != current_path:
                    current_path = key
                    current_pair = None
                    current_extract_error = None
                    try:
                        frame = _archive_image(root, item)
                        reference, test, _aoi_info = extractor(frame)
                        current_pair = (reference, test)
                    except Exception as exc:
                        current_extract_error = exc
                if current_extract_error is not None:
                    raise ValueError(
                        f"Extração AOI não disponível: {current_extract_error}"
                    ) from current_extract_error
                if current_pair is None:
                    raise ValueError("Par de imagens ausente")
                reference, test = current_pair
                verdict, score, digest = _check_model_response(
                    name,
                    model.inspect(reference, test, row["lighting_mode"]),
                )
                row["verdict"] = verdict
                row["ng_score_uncalibrated"] = score
                row["checkpoint_sha256"] = digest
                # REVISÃO nunca é OK/NG; para gate histórico é uma regressão.
                row["status"] = (
                    "PASSOU" if verdict == row["expected_verdict"] else "REGRESSAO"
                )
        except Exception as exc:
            row["status"] = "INVALIDO"
            row["error"] = f"{type(exc).__name__}: {exc}"
        all_rows.append(row)
        if progress is not None:
            progress(i, len(planned), f"{name}: {row['source_path']}")
    counts = {}
    for name in MODEL_NAMES:
        subset = [x for x in all_rows if x["model"] == name]
        statuses = Counter(x["status"] for x in subset)
        labels = Counter(x["expected_label"] for x in subset)
        counts[name] = {
            "eligible": len(subset),
            "passed": statuses["PASSOU"],
            "regressions": statuses["REGRESSAO"],
            "invalid": statuses["INVALIDO"],
            "model_unavailable": statuses[MODEL_UNAVAILABLE],
            "expected_OK": labels["OK"],
            "expected_NG": labels["NG"],
            "passed_all": bool(subset) and statuses["PASSOU"] == len(subset),
        }
    complete = all(counts[name]["passed_all"] for name in MODEL_NAMES)
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": "DIAGNOSTIC_NOT_ATTACHED_TO_MAIN",
        "production_blocking_enabled": False,
        "training_enabled": False,
        "knn_used": False,
        "specialist_moe_used": False,
        "full_archive_png_count": len(items),
        "model_order": list(MODEL_NAMES),
        "models": counts,
        "both_cnns_passed": complete,
        "can_release_operational_startup": complete,
        "memory_model_present": predictor_map[MODEL_MEMORY] is not None,
        "notes": [
            "Imagens históricas são teste de retenção, não teste independente.",
            "Registros multilight são verificados por PNG sem inferir vínculos de eventos.",
            "Somente após conectar e validar a CNN MEMÓRIA pode-se instalar gate no main.py.",
            "Mesmo 100% histórico não certifica capacidade de reconhecer NG novos.",
        ],
        "cases": all_rows,
    }


def write_cnn_report(report: dict, output: Path) -> tuple[Path, Path]:
    root = Path(report["root"]).resolve()
    out = Path(output).resolve()
    for name in ("ok_archive", "ng_archive", "dataset"):
        protected = (root / "public" / name).resolve()
        if out == protected or protected in out.parents:
            raise ValueError("Relatórios não podem ser salvos em archive/dataset")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = out / f"cnn_validation_{stamp}.json"
    text_path = out / f"cnn_validation_{stamp}.txt"
    _atomic_text(
        json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    lines = [
        "ODIN — VALIDAÇÃO DO ACERVO CNN — SOMENTE DIAGNÓSTICO",
        "KNN: DESABILITADO | TREINO: DESABILITADO",
        "Não é o gate bloqueante do main.py.",
        "",
    ]
    for name, counts in report["models"].items():
        lines.extend([
            f"{name}",
            f"  Elegíveis: {counts['eligible']}",
            f"  Aprovados: {counts['passed']}",
            f"  Regressões: {counts['regressions']}",
            f"  Inválidos: {counts['invalid']}",
            f"  Modelo ausente: {counts['model_unavailable']}",
            f"  Resultado: {'PASSOU' if counts['passed_all'] else 'FALHOU'}",
        ])
    lines.append("Ambas CNNs passaram: " + str(report["both_cnns_passed"]))
    lines.append("")
    lines.append("CASOS NÃO APROVADOS:")
    for row in report["cases"]:
        if row["status"] == "PASSOU":
            continue
        lines.append(
            f"{row['model']} | {row['source_path']} | {row['status']} "
            f"| esperado={row['expected_verdict']} "
            f"| obtido={row['verdict'] or 'N/D'} | {row['error'] or ''}"
        )
    _atomic_text(text_path, "\n".join(lines) + "\n")
    return json_path, text_path


__all__ = [
    "MODEL_FALTANDO", "MODEL_MEMORY", "scope_for_model",
    "validate_archive_cnns", "write_cnn_report",
]
