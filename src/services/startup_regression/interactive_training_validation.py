"""Validação histórica NÃO MUTANTE para o botão visual "Treinamento".

Compara KNN persistente (todas as categorias) e CNN FALTANDO V2
(FALTANDO, EMBORCADO, INVERTIDO, DESLOCADO) com PNGs dos arquivos.
Nunca treina, nunca chama XP, nunca modifica arquivo ou checkpoint.

A saída mantém casos não analisados no denominador em cancelamento.
Cancelamento cooperativo antes/depois de cada imagem e de cada motor.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable

from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.services.archive_aoi_category import resolve_archive_aoi_category
from src.services.faltando_cross_category_audit import _archive_image
from src.services.startup_regression.archive_inventory import inventory_archives
from src.services.startup_regression.archive_inventory_report import _atomic_text

SCHEMA = "visionx.interactive_archive_validation.v1"
MODELS = ("KNN_MEMORIA", "CNN_FALTANDO_V2")
EXPECTED = {"OK": "FALHA FALSA", "NG": "DEFEITO REAL"}
TARGET = .98


def _row(model: str, item: dict) -> dict:
    return {
        "model": model,
        "source_path": item.get("path"),
        "category": item.get("category_hint"),
        "lighting_mode": item.get("lighting_mode"),
        "expected_label": item.get("expected_label"),
        "expected_verdict": EXPECTED.get(item.get("expected_label")),
        "predicted_label": None,
        "verdict": None, "status": "NAO_ANALISADO",
        "memory_status": None,
        "cnn_score_ng": None, "checkpoint_sha256": None,
        "error": None,
    }


def summarize(rows: list[dict], images: list[dict]) -> dict:
    """Scores apenas de imagens elegíveis; NG->OK nunca escondido."""
    out = {}
    for model in MODELS:
        subset = [r for r in rows if r["model"] == model]
        evaluated = [r for r in subset if r["status"] != "FORA_ESCOPO"]
        by_label = {}
        for label in ("OK", "NG"):
            eligible = [r for r in evaluated if r["expected_label"] == label]
            passed = sum(r["status"] == "PASSOU" for r in eligible)
            wrong = sum(r["status"] == "ERRO_CLASSIFICACAO" for r in eligible)
            by_label[label] = {
                "total": len(eligible), "passed": passed, "incorrect": wrong,
                "failed_or_unavailable": len(eligible) - passed,
                "accuracy": round(passed / len(eligible), 8) if eligible else None,
            }
        statuses = Counter(r["status"] for r in subset)
        complete = bool(evaluated) and all(r["status"] != "NAO_ANALISADO" for r in evaluated)
        passed_all = complete and all(
            (v["total"] > 0 and v["accuracy"] >= TARGET)
            for v in by_label.values()
        )
        out[model] = {
            "total_archive": len(images),
            "eligible": len(evaluated),
            "out_of_scope": statuses["FORA_ESCOPO"],
            "passed": statuses["PASSOU"],
            "incorrect": statuses["ERRO_CLASSIFICACAO"],
            "review": statuses["REVISAO"],
            "no_coverage": statuses["SEM_COBERTURA"],
            "invalid": statuses["INVALIDO"],
            "not_evaluated": statuses["NAO_ANALISADO"],
            "by_label": by_label,
            "fully_evaluated": complete,
            "target_98pct_per_label": passed_all,
            "ng_as_ok": sum(r["expected_label"] == "NG"
                            and r["predicted_label"] == "OK" for r in subset),
            "ok_as_ng": sum(r["expected_label"] == "OK"
                            and r["predicted_label"] == "NG" for r in subset),
        }
    return out


def validate(
    root: Path, *, inventory: dict | None = None,
    extractor=None, cnn=None, memory=None,
    frame_loader: Callable = _archive_image,
    progress: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    root = Path(root).expanduser().resolve()
    start = time.monotonic()
    if progress:
        progress({"event": "phase", "phase": "Inventariando PNGs OK/NG"})
    evidence = inventory if inventory is not None else inventory_archives(root)
    images = evidence.get("images")
    if (evidence.get("schema") != "visionx.archive_inventory.v1"
            or not isinstance(images, list) or not images
            or len(images) != evidence.get("summary", {}).get("png_count")):
        raise ValueError("Inventário OK/NG vazio, incompleto ou inválido")
    # Imagens em conflito não podem ser consideradas acertos.
    conflicts = {p for group in evidence.get("cross_label_conflicts", ())
                 for p in group.get("paths", ())}
    rows = []
    for item in images:
        for model in MODELS:
            row = _row(model, item)
            if model == "CNN_FALTANDO_V2" and not uses_faltando_v2(
                    item.get("category_hint", "")):
                row["status"] = "FORA_ESCOPO"
            rows.append(row)
    valid_count = sum(r["status"] != "FORA_ESCOPO" for r in rows)
    if progress:
        progress({
            "event": "started", "total": valid_count,
            "images": len(images),
            "by_label": dict(Counter(i.get("expected_label") for i in images)),
        })

    if extractor is None:
        from src.services.faltando_neural_dataset import AOIPairExtractor
        extractor = AOIPairExtractor()
    if cnn is None:
        from src.core.neural.faltando_live import FaltandoCNNLive
        cnn = FaltandoCNNLive(online_root=root)
    if memory is None:
        from src.core.archive_persistent_memory import PersistentArchiveKNN
        memory = PersistentArchiveKNN(root)
    # Falhas de integridade da memória invalidam o teste KNN, nunca
    # silenciosamente geram uma memória vazia.
    memory.records()
    processed = 0
    stopped = False
    for index, item in enumerate(images):
        if cancelled and cancelled():
            stopped = True
            break
        current_rows = rows[index * 2:index * 2 + 2]
        working = [r for r in current_rows if r["status"] != "FORA_ESCOPO"]
        frame_error = None
        reference = test = info = None
        try:
            if item.get("status") != "VALID_PNG":
                raise ValueError("PNG não passou pela validação de integridade")
            if item.get("path") in conflicts:
                raise ValueError("PNG com rótulos OK/NG contraditórios")
            if item.get("expected_label") not in EXPECTED:
                raise ValueError("Rótulo de arquivo desconhecido")
            frame = frame_loader(root, item)
            reference, test, info = extractor(frame)
            if not isinstance(info, dict):
                raise ValueError("Extrator AOI não devolveu metadados")
            actual, normalized_value = resolve_archive_aoi_category(
                info, item.get("category_hint", ""),
            )
            info = {
                **info, "category": actual,
                "value": normalized_value,
                "lighting_mode": item.get("lighting_mode", "SIDE"),
            }
        except Exception as exc:
            frame_error = f"{type(exc).__name__}: {exc}"
        for row in working:
            if cancelled and cancelled():
                stopped = True
                break
            try:
                if frame_error is not None:
                    raise ValueError(frame_error)
                if row["model"] == "KNN_MEMORIA":
                    match = memory.lookup(reference, test, info)
                    row["memory_status"] = match.get("status")
                    if match.get("status") == "KNOWN":
                        label = match.get("label")
                        if label not in EXPECTED:
                            raise ValueError("Memória sem rótulo OK/NG válido")
                        row["predicted_label"] = label
                        row["verdict"] = EXPECTED[label]
                        row["status"] = (
                            "PASSOU" if label == row["expected_label"]
                            else "ERRO_CLASSIFICACAO"
                        )
                    else:
                        row["status"] = "SEM_COBERTURA"
                        row["error"] = str(match.get("reason"))
                else:
                    out = cnn.inspect(reference, test, info["lighting_mode"])
                    detail = out.get("detail", {}) if isinstance(out, dict) else {}
                    if (not isinstance(detail, dict)
                            or detail.get("cnn_v2_active") is not True
                            or detail.get("cnn_v2_checkpoint_verified") is not True
                            or detail.get("cnn_v2_status") != "INFERENCE_OK"):
                        raise ValueError("CNN V2 não disponível/checkpoint inválido")
                    verdict = str(out.get("verdict", "")).strip().upper()
                    score = detail.get("cnn_v2_ng_score_uncalibrated")
                    if score is None:
                        raise ValueError("CNN sem score NG")
                    import math
                    if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
                        raise ValueError("CNN score não finito")
                    row["cnn_score_ng"] = float(score)
                    row["checkpoint_sha256"] = detail.get("cnn_v2_checkpoint_sha256")
                    if verdict == "REVISÃO OBRIGATÓRIA":
                        row["status"] = "REVISAO"
                    elif verdict in EXPECTED.values():
                        row["verdict"] = verdict
                        row["predicted_label"] = (
                            "OK" if verdict == EXPECTED["OK"] else "NG"
                        )
                        row["status"] = (
                            "PASSOU" if verdict == row["expected_verdict"]
                            else "ERRO_CLASSIFICACAO"
                        )
                    else:
                        raise ValueError("Veredito CNN fora do contrato")
            except Exception as exc:
                row["status"] = "INVALIDO"
                row["error"] = f"{type(exc).__name__}: {exc}"
            processed += 1
            if progress:
                progress({
                    "event": "progress", "done": processed,
                    "total": valid_count, "model": row["model"],
                    "path": row["source_path"],
                    "status": row["status"],
                    "expected": row["expected_label"],
                    "predicted": row["predicted_label"],
                })
        if stopped:
            break
    result = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": "READ_ONLY_ARCHIVE_REPLAY",
        "title_in_ui": "TREINAMENTO (VALIDACAO HISTORICA)",
        "canceled": bool(stopped),
        "complete": not stopped,
        "archive_png_count": len(images),
        "processed_model_checks": processed,
        "planned_model_checks": valid_count,
        "duration_seconds": round(time.monotonic() - start, 3),
        "models": summarize(rows, images),
        "cases": rows,
        "notes": [
            "Nenhuma rede foi retreinada; nenhuma memória foi modificada.",
            "A métrica é replay histórico com rótulos das pastas, não generalização.",
            "A CNN FALTANDO V2 não tem escopo MUITO ADESIVO; FORA_ESCOPO não é acerto.",
            "Revisão, ausência de cobertura e dados inválidos são falhas na métrica.",
            "Cancelamento impede alegação de aprovação com acervo parcial.",
            "Não há comandos 0/1 nem alteração de gate, checkpoint ou dataset.",
        ],
    }
    if progress:
        progress({"event": "summary", "canceled": bool(stopped),
                  "models": result["models"]})
    return result


def save_report(result: dict, output_dir: Path) -> tuple[Path, Path]:
    root = Path(result["root"]).resolve()
    output_dir = Path(output_dir).resolve()
    permitted = (root / "reports").resolve()
    if permitted not in output_dir.parents:
        raise ValueError("Relatórios fora de reports/ bloqueados")
    for directory in ("ok_archive", "ng_archive", "dataset"):
        protected = (root / "public" / directory).resolve()
        if output_dir == protected or protected in output_dir.parents:
            raise ValueError("Relatório não pode alterar arquivo/dataset")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = output_dir / f"training_validation_{stamp}.json"
    txt_path = output_dir / f"training_validation_{stamp}.txt"
    _atomic_text(json_path, json.dumps(result, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n")
    text = [
        "ODIN | TREINAMENTO - VALIDACAO HISTORICA CNN FALTANDO V2 + KNN",
        f"Status: {'CANCELADO - PARCIAL' if result['canceled'] else 'CONCLUIDO'}",
        "Somente leitura | nao retreina pesos | nao muda memoria/linha de producao",
        f"Total PNGs: {result['archive_png_count']}",
        f"Verificacoes: {result['processed_model_checks']}/{result['planned_model_checks']}",
        f"Duracao: {result['duration_seconds']:.2f}s",
    ]
    for name, stats in result["models"].items():
        text.extend([
            "", name,
            f"Elegiveis: {stats['eligible']} | Fora de escopo: {stats['out_of_scope']}",
            f"Corretos: {stats['passed']} | Incorretos: {stats['incorrect']}",
            f"Revisao: {stats['review']} | Sem cobertura: {stats['no_coverage']}",
            f"Invalidos: {stats['invalid']} | Nao avaliados: {stats['not_evaluated']}",
            f"NG classificados como OK: {stats['ng_as_ok']}",
            f"OK classificados como NG: {stats['ok_as_ng']}",
            f"Meta 98% OK e NG: {stats['target_98pct_per_label']}",
        ])
        for label, metrics in stats["by_label"].items():
            accuracy = metrics["accuracy"]
            pct = "N/D" if accuracy is None else f"{100*accuracy:.2f}%"
            text.append(
                f"{label}: {metrics['passed']}/{metrics['total']} = {pct}"
                f"; pendentes/incorretos={metrics['failed_or_unavailable']}"
            )
    text.extend(["", "CASOS QUE NAO PASSARAM:"])
    for row in result["cases"]:
        if row["status"] not in ("PASSOU", "FORA_ESCOPO"):
            text.append(
                f"{row['model']} | {row['source_path']} | {row['status']}"
                f" | esperado={row['expected_label']} | obtido="
                f"{row['predicted_label'] or 'N/D'} | {row['error'] or ''}"
            )
    text.extend(["", "LIMITACOES:", *result["notes"]])
    _atomic_text(txt_path, "\n".join(text) + "\n")
    return json_path, txt_path


def configure_low_priority():
    """A validação não concorre com a captura com prioridade normal."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        if os.name == "nt":
            import ctypes
            # IDLE_PRIORITY_CLASS = 0x40 (Windows)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, 0x40)
        else:
            os.nice(10)
    except (OSError, AttributeError):
        pass
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Validação visual independente de ODIN")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--cancel-file", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    cancel = args.cancel_file.expanduser().resolve()
    reports_dir = (root / "reports" / "startup_regression").resolve()
    if reports_dir not in cancel.parents or cancel.suffix != ".cancel":
        raise ValueError("Cancelamento deve ficar dentro do diretório de relatórios")
    configure_low_priority()

    def emit(payload: dict):
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)

    try:
        report = validate(
            root, cancelled=cancel.exists, progress=emit
        )
        jp, tp = save_report(report, reports_dir)
        emit({
            "event": "finished", "canceled": report["canceled"],
            "json_path": str(jp), "txt_path": str(tp),
            "models": report["models"], "complete": report["complete"],
        })
        return 0 if report["complete"] else 2
    except Exception as exc:
        emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
