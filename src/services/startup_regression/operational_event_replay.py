"""Replay diagnóstico de eventos com identidade realmente verificável.

Somente manifestos LINKED com event_id, três arquivos e hashes conferidos
podem ser avaliados como UMA peça. Nunca agrupar fotos pelo minuto, categoria
ou aparência. Não envia XP, não treina, não altera acervo, KNN ou checkpoints.

Reutiliza a decisão SIDE e a fusão multilight existentes; o ambiente offline
não reproduz o tempo real do XP, dos workers ou todo o estado de Produção.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from src.core.multilight_fusion import fuse_multilight
from src.core.side_early_decision import build_side_early_decision
from src.services.startup_regression.archive_inventory import inventory_archives
from src.services.startup_regression.archive_inventory_report import _atomic_text

SCHEMA = "visionx.operational_event_replay.v1"
LIGHTS = ("SIDE", "TOP", "MID")
LABELS = {"NG": "DEFEITO REAL", "OK": "FALHA FALSA"}


def verified_event_manifests(inventory: dict) -> tuple[list[dict], dict]:
    """Recusa evento incompleto, duplicado ou inferido pelo nome do PNG."""
    if inventory.get("schema") != "visionx.archive_inventory.v1":
        raise ValueError("Inventário sem schema verificável")
    images = inventory.get("images")
    manifests = inventory.get("manifests")
    if not isinstance(images, list) or not isinstance(manifests, list):
        raise ValueError("Inventário incompleto")
    by_path = {item["path"]: item for item in images}
    collisions = Counter(
        str(item.get("event_id") or "")
        for item in manifests if item.get("event_id")
    )
    accepted, excluded = [], Counter()
    for item in manifests:
        event_id = str(item.get("event_id") or "")
        paths = item.get("frame_paths")
        label = item.get("expected_label")
        if item.get("status") != "LINKED":
            excluded["MANIFEST_INVALID"] += 1
            continue
        if collisions[event_id] > 1:
            excluded["DUPLICATE_EVENT_ID"] += 1
            continue
        if (not event_id or label not in LABELS or not isinstance(paths, dict)
                or set(paths) != set(LIGHTS)):
            excluded["INCOMPLETE_EVENT"] += 1
            continue
        if any(
            by_path.get(paths[light], {}).get("status") != "VALID_PNG"
            or by_path[paths[light]].get("expected_label") != label
            or not by_path[paths[light]].get("file_sha256")
            or not any(
                link.get("event_id") == event_id
                and link.get("lighting_mode") == light
                for link in by_path[paths[light]].get("manifest_links", ())
            )
            for light in LIGHTS
        ):
            excluded["UNVERIFIED_FRAME"] += 1
            continue
        accepted.append({
            "event_id": event_id, "expected_label": label,
            "manifest_path": item["path"],
            "frame_paths": {k: paths[k] for k in LIGHTS},
            "sha256": {k: by_path[paths[k]]["file_sha256"] for k in LIGHTS},
        })
    linked = {path for event in accepted for path in event["frame_paths"].values()}
    coverage = {
        "archive_png_count": len(images),
        "qualified_event_count": len(accepted),
        "linked_png_count": len(linked),
        "unlinked_png_count": len(images) - len(linked),
        "unusable_manifests": dict(sorted(excluded.items())),
        "coverage_warning": (
            "Arquivos sem manifesto event_id+SHA NÃO são eventos "
            "reconstruíveis; não usar nome/minuto para agrupá-los."
        ),
    }
    return sorted(accepted, key=lambda r: r["event_id"]), coverage


def _verified_frame(root: Path, filename: str, expected_sha: str):
    """A própria execução reconfere SHA e evita caminhos fora dos arquivos."""
    target = (root / filename).resolve()
    archives = (
        (root / "public" / "ok_archive").resolve(),
        (root / "public" / "ng_archive").resolve(),
    )
    if not any(parent in target.parents for parent in archives):
        raise ValueError("Caminho de imagem fora do acervo")
    data = target.read_bytes()
    if sha256(data).hexdigest() != expected_sha:
        raise ValueError("Arquivo alterado após o inventário")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("Imagem PNG não decodificada")
    return frame


class ReplayRuntime:
    """Mesmos três wrappers decisórios finais, sem controller XP ou persistência.

    O roteador legado completo da aplicação tem outras extensões; por isso o
    relatório distingue replay do NÚCLEO de uma inspeção física certificada.
    """

    def __init__(self, root: Path):
        from src.config.settings import BASE_DIR
        if Path(BASE_DIR).resolve() != root.resolve():
            raise ValueError("O replay precisa rodar no root da instalação ODIN")
        from src.core.moe_orchestrator import MoEOrchestrator
        from src.core.neural.faltando_live import (
            FaltandoCNNLive, install_faltando_cnn_live,
        )
        from src.core.archive_persistent_memory import PersistentArchiveKNN
        from src.core.verified_memory_router import (
            VerifiedKNNMemory, install_memory_first_router,
        )
        from src.core.knn_c_live import LiveKNNStrategyC, install_live_knn_strategy_c
        from src.core.strict_category_memory import install_strict_category_memory
        from src.core.experts.knn_expert import KNNExpert

        class OfflineMoE(MoEOrchestrator):
            pass

        install_strict_category_memory(KNNExpert)
        install_faltando_cnn_live(
            OfflineMoE, predictor=FaltandoCNNLive(online_root=root)
        )
        install_memory_first_router(
            OfflineMoE,
            memory=VerifiedKNNMemory(
                archive_store=PersistentArchiveKNN(root)
            ),
        )
        self.strategy_c = LiveKNNStrategyC(root=root)
        self.strategy_c_warmup = self.strategy_c.warmup()
        install_live_knn_strategy_c(OfflineMoE, strategy=self.strategy_c)
        self.orchestrator = OfflineMoE()
        from src.services.faltando_neural_dataset import AOIPairExtractor
        self.extractor = AOIPairExtractor()

    def inspect(self, frame: np.ndarray, light: str, expected_category: str | None):
        from src.core.adhesive_multilight_analysis import (
            analyze_lighting, build_lighting_context,
        )
        from src.services.archive_aoi_category import resolve_archive_aoi_category
        reference, test, info = self.extractor(frame)
        actual, value = resolve_archive_aoi_category(
            info, expected_category or "",
        )
        if actual == "Unknown":
            raise ValueError("Categoria AOI indisponível")
        info = {**info, "category": actual, "value": value}
        context = build_lighting_context(reference, test)
        result = analyze_lighting(
            self.orchestrator, reference, test, info, light, context=context,
        )
        if not isinstance(result, dict):
            raise ValueError("Roteador operacional não retornou análise")
        return result, actual, {
            "lighting_mode": light,
            "sample_image": reference,
            "test_image": test,
            "analysis": result,
        }

    def finalize(self, analyses: dict, samples: dict, category: str, *, side_only: bool):
        if side_only:
            return build_side_early_decision(analyses["SIDE"], category)
        final = fuse_multilight(analyses, category)
        if final is None:
            return None
        from src.core.physical_missing_veto import apply_physical_missing_veto
        from src.core.knn_visual_corroboration import apply_knn_visual_corroboration
        final = apply_physical_missing_veto(final, samples, category)
        knn = self.orchestrator.experts.get("knn")
        return apply_knn_visual_corroboration(final, samples, category, knn)


def _classify(final: dict | None) -> str | None:
    if not isinstance(final, dict):
        return None
    verdict = str(final.get("verdict") or "").upper()
    if verdict == "DEFEITO REAL" and final.get("is_defect") is True:
        return "NG"
    if verdict == "FALHA FALSA" and final.get("is_defect") is False:
        return "OK"
    return None


def replay(
    root: Path, *, inventory: dict | None = None,
    runtime_factory: Callable = ReplayRuntime,
    frame_loader: Callable = _verified_frame,
    progress: Callable[[dict], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> dict:
    """NÃO simula julgamentos ausentes: sem manifesto é SEM COBERTURA."""
    root = Path(root).expanduser().resolve()
    begin = time.monotonic()
    if progress:
        progress({"event": "phase", "phase": "Verificando manifests/event_id"})
    inv = inventory if inventory is not None else inventory_archives(root)
    events, coverage = verified_event_manifests(inv)
    if progress:
        progress({"event": "started", "total": len(events),
                  "qualified_events": len(events),
                  "unlinked_pngs": coverage["unlinked_png_count"]})
    rows = []
    runtime = runtime_factory(root) if events else None
    complete = True
    for n, item in enumerate(events, 1):
        if cancelled and cancelled():
            complete = False
            break
        row = {
            "event_id": item["event_id"],
            "expected_label": item["expected_label"],
            "manifest_path": item["manifest_path"],
            "frame_paths": item["frame_paths"],
            "predicted_label": None,
            "verdict": None, "status": "INVALIDO",
            "route": None, "analyzed_lights": [], "skipped_lights": [],
            "scores_by_light": {}, "checkpoints_by_light": {},
            "recognition_by_light": {},
            "production_auto_eligible_diagnostic": None,
            "error": None,
        }
        try:
            analyses, samples, category = {}, {}, None
            for light in LIGHTS:
                if light != "SIDE" and row["route"] == "SIDE_EARLY":
                    break
                source = frame_loader(
                    root, item["frame_paths"][light], item["sha256"][light]
                )
                if cancelled and cancelled():
                    complete = False
                    row["status"] = "NAO_ANALISADO"
                    break
                result, local_category, sample = runtime.inspect(
                    source, light, category
                )
                if category is None:
                    category = local_category
                elif category != local_category:
                    raise ValueError("Categoria AOI divergente entre iluminações")
                analyses[light] = result
                samples[light] = sample
                row["analyzed_lights"].append(light)
                detail = result.get("detail", {})
                detail = detail if isinstance(detail, dict) else {}
                row["scores_by_light"][light] = detail.get(
                    "cnn_v2_ng_score_uncalibrated"
                )
                row["checkpoints_by_light"][light] = detail.get(
                    "cnn_v2_checkpoint_sha256"
                )
                row["recognition_by_light"][light] = detail.get(
                    "recognition_route", "UNAVAILABLE"
                )
                if light == "SIDE":
                    early = build_side_early_decision(result, category)
                    if early is not None:
                        row["route"] = "SIDE_EARLY"
            if row["status"] == "NAO_ANALISADO":
                rows.append(row)
                break
            side_only = row["route"] == "SIDE_EARLY"
            if not side_only:
                row["route"] = "MULTILIGHT"
            final = runtime.finalize(
                analyses, samples, category, side_only=side_only
            )
            row["skipped_lights"] = [
                light for light in LIGHTS if light not in row["analyzed_lights"]
            ]
            row["predicted_label"] = _classify(final)
            if isinstance(final, dict):
                row["verdict"] = final.get("verdict")
                row["fusion_rule"] = (
                    final.get("detail", {}).get("fusion_rule")
                    if isinstance(final.get("detail"), dict) else None
                )
                # Somente simulação da política (sem nenhum comando XP).
                from src.ui.production_confidence_gate import production_decision_policy
                policy = production_decision_policy(final)
                row["production_auto_eligible_diagnostic"] = bool(
                    policy.get("auto_allowed", False)
                )
            row["status"] = (
                "PASSOU" if row["predicted_label"] == row["expected_label"]
                else "ERRO_CLASSIFICACAO" if row["predicted_label"]
                else "REVISAO"
            )
        except Exception as exc:
            row["status"] = "INVALIDO"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if progress:
            progress({"event": "progress", "done": n, "total": len(events),
                      "expected": item["expected_label"],
                      "predicted": row["predicted_label"],
                      "status": row["status"], "event_id": item["event_id"]})
    counts = Counter(r["status"] for r in rows)
    errors = {
        "ng_as_ok": sum(r["expected_label"] == "NG" and
                        r["predicted_label"] == "OK" for r in rows),
        "ok_as_ng": sum(r["expected_label"] == "OK" and
                        r["predicted_label"] == "NG" for r in rows),
    }
    finished = bool(complete and len(rows) == len(events))
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": "READ_ONLY_MANIFEST_EVENT_REPLAY",
        "complete": finished,
        "canceled": not complete,
        "is_end_to_end_production_certification": False,
        "is_unseen_generalization_test": False,
        "coverage": coverage,
        "counts": {
            "qualified_events": len(events),
            "replayed": len(rows),
            "passed": counts["PASSOU"],
            "incorrect": counts["ERRO_CLASSIFICACAO"],
            "review": counts["REVISAO"],
            "invalid": counts["INVALIDO"],
            "not_evaluated": len(events) - len(rows) + counts["NAO_ANALISADO"],
            **errors,
        },
        "duration_seconds": round(time.monotonic() - begin, 3),
        "strategy_c_warmup": getattr(runtime, "strategy_c_warmup", None),
        "events": rows,
        "warnings": [
            "Não agrupa eventos por horário: somente manifestos vinculados por event_id e SHA.",
            "PNG isolado sem manifesto não é considerado evento nem acerto.",
            "Usa núcleo decisório KNN/CNN/SIDE/fusão; não replica captura temporal, todas as extensões de legado e estado físico XP.",
            "Não há interação com XP, mudança de checkpoint, treinamento nem escrita no acervo.",
            "Memória e PNGs históricos podem conter a própria consulta: não equivale a generalização independente.",
            "Rótulos provêm das pastas OK/NG, não de auditoria humana de cada evento.",
            "Sem cobertura suficiente, as contagens não podem ser comparadas aos 420/21 eventos informados pelo operador.",
        ],
    }


def save_report(result: dict, directory: Path) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = directory / f"operational_event_replay_{stamp}.json"
    txt_path = directory / f"operational_event_replay_{stamp}.txt"
    _atomic_text(json_path, json.dumps(result, ensure_ascii=False, indent=2))
    coverage, count = result["coverage"], result["counts"]
    report = [
        "ODIN | REPLAY OPERACIONAL (SOMENTE EVENT_ID VALIDADO)",
        f"Completo: {result['complete']}",
        f"PNGs: {coverage['archive_png_count']}",
        f"Eventos qualificados: {coverage['qualified_event_count']}",
        f"PNGs sem vínculo de evento: {coverage['unlinked_png_count']}",
        f"Eventos avaliados: {count['replayed']} | Corretos: {count['passed']}",
        f"NG classificados como OK: {count['ng_as_ok']}",
        f"OK classificados como NG: {count['ok_as_ng']}",
        f"Revisão: {count['review']} | Inválidos: {count['invalid']}",
        "",
        *result["warnings"], "",
    ]
    for row in result["events"]:
        report.append(
            f"{row['event_id']} | {row['route']} | {row['status']} "
            f"| {row['expected_label']} -> {row['predicted_label']} "
            f"| {','.join(row['analyzed_lights'])} | {row['error'] or ''}"
        )
    _atomic_text(txt_path, "\n".join(report) + "\n")
    return json_path, txt_path


__all__ = [
    "SCHEMA", "ReplayRuntime", "replay", "save_report",
    "verified_event_manifests",
]
