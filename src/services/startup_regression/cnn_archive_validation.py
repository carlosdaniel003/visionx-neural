"""Validação CNN FALTANDO V2 e MEMÓRIA KNN sobre o acervo OK/NG da AOI.

Sem MoE/especialistas físicos, treinamento, decisões XP ou mudanças
nas fotos. KNN só compara o par AOI com memórias humanas exatas e verificadas.

CNN FALTANDO v2: FALTANDO/EMBORCADO/INVERTIDO/DESLOCADO.
MEMÓRIA KNN: TODAS as categorias, inclusive MUITO ADESIVO.

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

SCHEMA = "visionx.startup_cnn_knn_archive_validation.v2"
MODEL_FALTANDO = "CNN_FALTANDO_V2"
MODEL_MEMORY = "MEMORIA_KNN"
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
        "memory_status": None,
        "memory_label": None,
        "memory_source_json": None,
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
        if (
            not isinstance(detail, dict)
            or detail.get("model_kind") != "knn_verified_exact"
        ):
            raise ValueError("Resposta não veio da memória KNN exata verificada")
        verdict = str(output.get("verdict", "")).strip().upper()
        if verdict not in {"DEFEITO REAL", "FALHA FALSA", "REVISÃO OBRIGATÓRIA"}:
            raise ValueError("Veredito MEMÓRIA KNN inválido")
        status = str(detail.get("memory_status", ""))
        if status == "KNOWN":
            if detail.get("verified_exact_match") is not True:
                raise ValueError("Memória KNN sem correspondência exata humana")
            label = detail.get("memory_label")
            expected_verdict = (
                "DEFEITO REAL" if label == "NG"
                else "FALHA FALSA" if label == "OK"
                else None
            )
            if verdict != expected_verdict:
                raise ValueError("Veredito inconsistente com rótulo KNN")
        elif status in {"NEW", "UNAVAILABLE", "CONFLICT"}:
            if verdict != "REVISÃO OBRIGATÓRIA" or detail.get("verified_exact_match"):
                raise ValueError("Memória sem cobertura não pode produzir OK/NG")
        else:
            raise ValueError("Estado KNN inválido")
        # KNN não possui checkpoint CNN nem score probabilístico calibrado.
        return verdict, None, None
    raise ValueError("Nome de modelo inválido")


def validate_archive_cnns(
    root: Path, *,
    inventory: dict | None = None,
    extractor=None,
    predictors: dict | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Avalia CNN e KNN separadamente, sem transformar ausência em acerto.

    A memória KNN deve recuperar cada exemplo de registro humano verificado.
    Um PNG sem registro de memória é SEM COBERTURA, não classificação correta.
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
    knn_init_error = None
    if MODEL_MEMORY not in predictor_map:
        try:
            from .knn_archive_predictor import KNNArchivePredictor
            predictor_map[MODEL_MEMORY] = KNNArchivePredictor(root=root)
        except Exception as exc:
            knn_init_error = f"{type(exc).__name__}: {exc}"
            predictor_map[MODEL_MEMORY] = None

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
                row["error"] = knn_init_error or "Memória KNN indisponível"
            else:
                key = row["source_path"]
                if key != current_path:
                    current_path = key
                    current_pair = None
                    current_extract_error = None
                    try:
                        frame = _archive_image(root, item)
                        reference, test, _aoi_info = extractor(frame)
                        current_pair = (reference, test, _aoi_info)
                    except Exception as exc:
                        current_extract_error = exc
                if current_extract_error is not None:
                    raise ValueError(
                        f"Extração AOI não disponível: {current_extract_error}"
                    ) from current_extract_error
                if current_pair is None:
                    raise ValueError("Par de imagens ausente")
                reference, test, _aoi_info = current_pair
                if name == MODEL_MEMORY:
                    # A memória exige o OCR real da imagem. Nunca emprestar
                    # category/Board/Parts da pasta para fabricar um match.
                    if not isinstance(_aoi_info, dict):
                        raise ValueError("OCR da AOI indisponível para KNN")
                    observed_info = dict(_aoi_info)
                    original = str(row["category_hint"] or "").strip().upper()
                    from src.core.strict_category_memory import canonical_memory_category
                    if (
                        original not in {"", "UNKNOWN"}
                        and canonical_memory_category(original)
                        != canonical_memory_category(observed_info.get("category", ""))
                    ):
                        raise ValueError(
                            "Categoria OCR da imagem diverge do nome arquivado"
                        )
                    output = model.inspect(
                        reference, test, row["lighting_mode"], observed_info
                    )
                else:
                    output = model.inspect(
                        reference, test, row["lighting_mode"]
                    )
                verdict, score, digest = _check_model_response(name, output)
                row["verdict"] = verdict
                row["ng_score_uncalibrated"] = score
                row["checkpoint_sha256"] = digest
                if name == MODEL_MEMORY:
                    detail = output["detail"]
                    row["memory_status"] = detail["memory_status"]
                    row["memory_label"] = detail.get("memory_label")
                    row["memory_source_json"] = detail.get("memory_source_json")
                    # Ausência de memória é lacuna de cobertura; não culpar
                    # a CNN nem classificar NEW como falso NG/OK.
                    if detail["memory_status"] != "KNOWN":
                        row["status"] = "SEM_COBERTURA"
                    else:
                        row["status"] = (
                            "PASSOU" if verdict == row["expected_verdict"]
                            else "REGRESSAO"
                        )
                else:
                    # Revisão nunca é aprovação de treino.
                    row["status"] = (
                        "PASSOU" if verdict == row["expected_verdict"]
                        else "REGRESSAO"
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
            "without_memory_coverage": statuses["SEM_COBERTURA"],
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
        "full_archive_png_count": len(items),
        "model_order": list(MODEL_NAMES),
        "models": counts,
        "cnn_and_knn_passed": complete,
        "can_release_operational_startup": complete,
        "knn_model_present": predictor_map[MODEL_MEMORY] is not None,
        "knn_used": predictor_map[MODEL_MEMORY] is not None,
        "knn_policy": "VERIFIED_EXACT_PAIR_HUMAN_MEMORY",
        "specialist_moe_used": False,
        "notes": [
            "Imagens históricas são teste de retenção, não teste independente.",
            "Registros multilight são verificados por PNG sem inferir vínculos de eventos.",
            "KNN conhecido exige par exato humano verificado; NEW/CONFLICT são lacunas de cobertura.",
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
        "ODIN — CNN FALTANDO V2 + MEMÓRIA KNN — DIAGNÓSTICO",
        "KNN: CONSULTA EXATA VERIFICADA | TREINO: DESABILITADO",
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
            f"  Sem cobertura KNN: {counts['without_memory_coverage']}",
            f"  Resultado: {'PASSOU' if counts['passed_all'] else 'FALHOU'}",
        ])
    lines.append("CNN e KNN passaram: " + str(report["cnn_and_knn_passed"]))
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
