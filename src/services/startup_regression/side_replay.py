"""Etapa 2 — Diagnóstico offline SIDE histórico, sempre SEM MEMÓRIA.

Comando: python -m src.services.startup_regression.side_replay
Não bloqueia o ODIN, não altera imagens e não treina modelos.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from .archive_inventory import inventory_archives
from .archive_inventory_report import _atomic_text
from .inspection_runner import ReplayError, SideInspectionRunner


SCHEMA = "visionx.startup_side_replay.v1"
CONTRACT = {
    "OK": "FALHA FALSA",
    "NG": "DEFEITO REAL",
}


def verdict_status(expected: str, verdict: str, review: bool = False) -> str:
    if expected not in CONTRACT:
        return "INVALIDO"
    if review or verdict == "REVISÃO OBRIGATÓRIA":
        return "REGRESSAO"
    if verdict not in {"FALHA FALSA", "DEFEITO REAL"}:
        return "INVALIDO"
    return "PASSOU" if CONTRACT[expected] == verdict else "REGRESSAO"


def run_side_replay(
    root: Path,
    *,
    runner=None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    inventory = inventory_archives(root)
    # Legacy SIDE = antigo, sem sufixo; imagens novas com SIDE explícito
    # pertencem a eventos da Etapa 3 e não devem ser julgadas isoladamente.
    legacy = [
        item for item in inventory["images"]
        if item["lighting_source"] == "LEGACY_DEFAULT"
    ]
    counts = Counter()
    result: list[dict[str, Any]] = []
    client = runner
    initialization_error = None
    if not legacy:
        initialization_error = "Nenhuma imagem histórica SIDE para analisar"
    elif client is None:
        try:
            client = SideInspectionRunner()
        except Exception as exc:
            initialization_error = f"{type(exc).__name__}: {exc}"

    for i, item in enumerate(legacy, start=1):
        expected = item["expected_label"]
        case = {
            "source_path": item["path"],
            "expected_label": expected,
            "expected_verdict": CONTRACT[expected],
            "lighting_mode": "SIDE",
            "category_hint": item["category_hint"],
            "verdict": None,
            "status": "INVALIDO",
            "error": None,
            "memory_consulted": False,
            "knn_enabled": False,
        }
        if item["status"] != "VALID_PNG":
            case["error"] = "PNG inválido: " + str(item.get("issues", []))
        elif initialization_error:
            case["error"] = initialization_error
        else:
            try:
                path = root / item["path"]
                judgement = client.inspect_png(path, item["category_hint"])
                if judgement.get("memory_consulted") is not False:
                    raise ReplayError("Motor informou consulta à memória")
                if judgement.get("knn_enabled") is not False:
                    raise ReplayError("Motor informou KNN habilitado")
                case.update(judgement)
                case["status"] = verdict_status(
                    expected,
                    judgement["verdict"],
                    bool(judgement.get("requires_review", False)),
                )
            except Exception as exc:
                case["error"] = f"{type(exc).__name__}: {exc}"

        counts[case["status"]] += 1
        result.append(case)
        if progress is not None:
            progress(i, len(legacy), item["path"])

    by_label = {}
    for label in ("OK", "NG"):
        batch = [case for case in result if case["expected_label"] == label]
        by_label[label] = {
            "total": len(batch),
            "passed": sum(case["status"] == "PASSOU" for case in batch),
            "regressions": sum(case["status"] == "REGRESSAO" for case in batch),
            "invalid": sum(case["status"] == "INVALIDO" for case in batch),
        }

    return {
        "schema": SCHEMA,
        "phase": "STAGE_2_SIDE_DIAGNOSTIC_ONLY",
        "root": str(root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_operational_gate": False,
        "active_learning_enabled": False,
        "memory_policy": "DISABLED_NO_KNN_NO_DATASET_LOOKUP",
        "scope": "LEGACY_SIDE_ONLY",
        "inventory_total_png": inventory["summary"]["png_count"],
        "multilight_explicit_deferred": inventory["summary"]["png_count"] - len(legacy),
        "summary": {
            "total": len(legacy),
            "passed": counts["PASSOU"],
            "regressions": counts["REGRESSAO"],
            "invalid": counts["INVALIDO"],
            "by_label": by_label,
            "initialization_error": initialization_error,
        },
        "cases": result,
    }


def human_summary(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "ODIN | REPLAY SIDE HISTÓRICO | ETAPA 2",
        "=" * 58,
        "SOMENTE DIAGNÓSTICO — NÃO BLOQUEIA O ODIN",
        "KNN E MEMÓRIA DE EXEMPLOS DESABILITADOS",
        "Sem treinamento, escrita de imagens ou comandos AOI",
        "",
        f"Raiz: {report['root']}",
        f"Total SIDE histórico: {summary['total']}",
        f"Aprovados: {summary['passed']}",
        f"Regressões: {summary['regressions']}",
        f"Inválidos: {summary['invalid']}",
        f"Multilight explícito reservado à Etapa 3: {report['multilight_explicit_deferred']}",
        "",
    ]
    for label, count in summary["by_label"].items():
        lines.append(
            f"{label}: {count['passed']}/{count['total']} aprovados, "
            f"{count['regressions']} regressões, {count['invalid']} inválidos"
        )
    lines.extend(["", "CASOS NÃO APROVADOS:"])
    failures = [case for case in report["cases"] if case["status"] != "PASSOU"]
    if not failures:
        lines.append("Nenhum (na amostra executada)")
    for case in failures:
        lines.append(
            f"- {case['status']} | {case['source_path']} | "
            f"esperado={case['expected_verdict']} | "
            f"obtido={case.get('verdict') or 'SEM VEREDITO'} | "
            f"{case.get('error') or ''}"
        )
    lines.extend([
        "",
        "PASSOU indica somente concordância do motor físico com o rótulo",
        "do arquivo. Não utiliza imagens idênticas da memória.",
        "A existência de erro não pode ser convertida em OK automaticamente.",
    ])
    return "\n".join(lines) + "\n"


def write_side_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    root = Path(report["root"]).resolve()
    output_dir = Path(output_dir).resolve()
    protected = [
        root / "public" / name
        for name in ("ok_archive", "ng_archive", "dataset")
    ]
    for folder in protected:
        resolved = folder.resolve()
        if output_dir == resolved or resolved in output_dir.parents:
            raise ValueError("Relatório não pode entrar no archive/dataset")

    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = output_dir / f"side_replay_{now}.json"
    txt_path = output_dir / f"side_replay_{now}.txt"
    _atomic_text(
        json_path,
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_text(txt_path, human_summary(report))
    return json_path, txt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reanálise SIDE histórico sem memória/KNN, só diagnóstico."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "reports" / "startup_regression"
    )
    print("ODIN — replay SIDE sem memória | Etapa 2", flush=True)

    def progress(i: int, total: int, path: str):
        print(f"[{i}/{total}] {path}", flush=True)

    report = run_side_replay(root, progress=progress)
    json_path, txt_path = write_side_report(report, output_dir)
    print(human_summary(report).split("CASOS NÃO APROVADOS:")[0], flush=True)
    print(f"JSON: {json_path}\nTXT: {txt_path}", flush=True)
    # Diagnóstico não é gate. Erro deve ser visível para automação,
    # mas não modifica o processo do main.py.
    return 0 if (
        report["summary"]["total"] > 0
        and report["summary"]["passed"] == report["summary"]["total"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
