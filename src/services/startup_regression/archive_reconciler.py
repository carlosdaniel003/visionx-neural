"""Reconciliação offline e somente leitura do acervo AOI com MEMÓRIA KNN.

Diagnostica a elegibilidade de cada JSON de public/dataset segundo as
MESMAS exigências de VerifiedKNNMemory._entry, e compara pares AOI dos
screenshots históricos sem usar o rótulo da pasta para criar um match.

Não modifica registros, rótulos, pesos, limiares ou startup.
Nunca converte NEW em KNOWN e não aprova lacunas de proveniência.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.core.anomaly_signature import valid_anomaly_signature
from src.core.strict_category_memory import (
    canonical_memory_category,
    canonical_memory_lighting,
)
from src.core.verified_memory_router import VerifiedKNNMemory, _clean, _file_within
from src.services.image_archive_dedup import image_fingerprint
from src.services.startup_regression.archive_inventory import inventory_archives
from src.services.startup_regression.archive_inventory_report import _atomic_text
from src.utils.text_normalizer import normalize_aoi_text

from .legacy_memory_compat import (
    COMPATIBLE as LEGACY_COMPATIBLE,
    inspect_legacy_record,
    summarize_legacy_profiles,
)

SCHEMA = "visionx.archive_memory_reconciliation.v1"
HUMAN_SOURCES = frozenset({
    "button", "xp_keyboard", "keyboard", "manual",
    "operator", "physical_keyboard", "human",
})
DATA_FOLDERS = (("nao_anomalia", "OK"), ("anomalia", "NG"))
MAX_CANDIDATES = 4


def _load_bgr(path: Path) -> np.ndarray:
    frame = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("PNG ilegível para OpenCV")
    return frame


def _safe_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _anomaly_signature(data: dict) -> bool:
    analysis = data.get("analysis")
    analysis = analysis if isinstance(analysis, dict) else {}
    return any(valid_anomaly_signature(value) for value in (
        analysis.get("anomaly_memory"),
        analysis.get("anomaly_signature"),
        data.get("anomaly_memory"),
    ))


def _legacy_embedding(data: dict) -> bool:
    analysis = data.get("analysis")
    if not isinstance(analysis, dict):
        return False
    try:
        embedding = np.asarray(analysis.get("embedding") or [], dtype=np.float32)
        return bool(embedding.size and np.all(np.isfinite(embedding)))
    except (TypeError, ValueError):
        return False


def _audit_record(root: Path, json_path: Path, folder_label: str) -> dict:
    relative = _safe_relative(root, json_path)
    entry = {
        "path": relative, "folder_label": folder_label,
        "label": None, "reason": None, "category": None,
        "lighting_mode": None, "board": None, "parts": None,
        "value": None, "key": None, "source_pixel_sha256": None,
        "source_png_present": False, "audit_pair_present": False,
    }
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON não contém objeto")
        info = data.get("aoi_info")
        info = info if isinstance(info, dict) else {}
        storage = data.get("storage")
        storage = storage if isinstance(storage, dict) else {}
        decision = data.get("decision")
        decision = decision if isinstance(decision, dict) else {}
        label = str(data.get("label", "")).strip().upper()
        operator = str(decision.get("operator_label", "")).strip().upper()
        source = str(decision.get("source", "")).strip().lower()
        category = canonical_memory_category(info.get("category"))
        lighting = canonical_memory_lighting(info.get("lighting_mode"))
        entry.update({
            "label": label, "category": category, "lighting_mode": lighting,
            "board": _clean(info.get("board")),
            "parts": _clean(info.get("parts")),
            "value": _clean(info.get("value")),
        })
        # Os PNGs fonte servem apenas para investigação, NÃO atestam
        # par AOI, nem dispensam registro de operador.
        source_path = _file_within(
            json_path.parent, storage.get("source_image_file", "")
        )
        if source_path is not None:
            try:
                entry["source_pixel_sha256"] = image_fingerprint(
                    _load_bgr(source_path)
                )
                entry["source_png_present"] = True
            except (OSError, ValueError, cv2.error):
                pass
        reference_path = _file_within(
            json_path.parent, storage.get("reference_image_file", "")
        )
        test_path = _file_within(
            json_path.parent, storage.get("test_image_file", "")
        )
        entry["audit_pair_present"] = bool(reference_path and test_path)

        if data.get("schema") != "visionx.memory.v3":
            entry["reason"] = "SCHEMA_NAO_SUPORTADO"
            # Auditoria complementar: não modificar schema nem considerar
            # o registro antigo KNOWN. O adaptador só simula identidade
            # exata caso os metadados e os dois PNGs sejam demonstráveis.
            entry["legacy_profile"] = inspect_legacy_record(
                json_path, folder_label, data
            )
        elif label not in {"OK", "NG"} or label != folder_label or operator != label:
            entry["reason"] = "ROTULO_HUMANO_INCONSISTENTE"
        elif source not in HUMAN_SOURCES and not source.startswith("operator_"):
            entry["reason"] = "SEM_CONFIRMacao_HUMANA".upper()
        elif not _anomaly_signature(data):
            entry["reason"] = (
                "SOMENTE_EMBEDDING_LEGADO" if _legacy_embedding(data)
                else "SEM_ASSINATURA_ANOMALIA"
            )
        elif not reference_path or not test_path:
            entry["reason"] = "PAR_PNG_AUDITORIA_AUSENTE"
        elif not all((category, entry["board"], entry["parts"])):
            entry["reason"] = "METADADOS_INCOMPLETOS"
        else:
            # Usa exatamente a validação de produção, inclusive o SHA
            # declarado, categoria/peça, iluminação e hashes visuais.
            record = {
                "mode": "anomaly",
                "json_path": str(json_path),
                "label": label, "folder_label": folder_label,
                "category": category, "lighting_mode": lighting,
                "part": entry["parts"],
            }
            approved = VerifiedKNNMemory._entry(record)
            if approved is None:
                entry["reason"] = "PAR_OU_METADATA_INCOMPATIVEL"
            else:
                entry["key"] = approved[0]
                entry["reason"] = "VERIFICADO_KNN"

    except (OSError, ValueError, TypeError, cv2.error, UnicodeError) as exc:
        entry["reason"] = "JSON_OU_PNG_INVALIDO"
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def scan_memory_dataset(root: Path) -> dict:
    """Varre JSONs mesmo quando o carregador KNN os ignoraria."""
    root = Path(root).expanduser().resolve()
    rows = []
    for subdir, label in DATA_FOLDERS:
        parent = root / "public" / "dataset" / subdir
        if not parent.is_dir():
            continue
        for path in sorted(parent.rglob("*.json")):
            # Nunca seguir symlinks externos ao dataset.
            if path.is_symlink():
                rows.append({
                    "path": str(path), "folder_label": label,
                    "reason": "LINK_SIMBOLICO_IGNORADO", "key": None,
                })
                continue
            try:
                path.resolve().relative_to(parent.resolve())
            except ValueError:
                continue
            rows.append(_audit_record(root, path, label))
    return {
        "rows": rows,
        "counts": dict(sorted(Counter(
            item["reason"] for item in rows
        ).items())),
    }


def _index_records(rows: list[dict]) -> dict:
    verified, contexts, visual_pairs, tests, screenshots, legacy = (
        defaultdict(list) for _ in range(6)
    )
    categories = Counter()
    for entry in rows:
        category = entry.get("category")
        lighting = entry.get("lighting_mode")
        if category:
            categories[(category, lighting)] += 1
        ctx = (
            category, lighting,
            entry.get("board"), entry.get("parts"),
        )
        if all(ctx):
            contexts[ctx].append(entry)
        key = entry.get("key")
        if key is not None:
            verified[key].append(entry)
            visual_pairs[key[-2:]].append(entry)
            tests[key[-1]].append(entry)
        profile = entry.get("legacy_profile")
        if (
            isinstance(profile, dict)
            and profile.get("status") == LEGACY_COMPATIBLE
            and profile.get("would_be_key") is not None
        ):
            legacy[tuple(profile["would_be_key"])].append(entry)
        if entry.get("source_pixel_sha256"):
            screenshots[entry["source_pixel_sha256"]].append(entry)
    return {
        "verified": verified, "contexts": contexts,
        "visual_pairs": visual_pairs, "tests": tests,
        "screenshots": screenshots, "categories": categories,
        "legacy_simulated_exact": legacy,
    }


def _summarize_candidate(entry: dict, root: Path) -> dict:
    return {
        "record": entry["path"],
        "record_label": entry.get("label"),
        "eligibility": entry.get("reason"),
    }


def _classify(key: tuple, source_hash: str, label: str, indexes: dict):
    context = key[:4]
    matched = indexes["verified"].get(key, [])
    if matched:
        labels = {item["label"] for item in matched}
        if len(labels) > 1:
            return "CONFLITO_DE_ROTULOS", matched
        return (
            "PAR_VERIFICADO" if label in labels
            else "ROTULO_DIVERGENTE",
            matched,
        )
    same_pair = indexes["visual_pairs"].get(key[-2:], [])
    if same_pair:
        return "PAR_EXATO_METADADOS_DIFERENTES", same_pair
    same_test = indexes["tests"].get(key[-1], [])
    if same_test:
        return "TESTE_EXATO_GABARITO_DIFERENTE", same_test
    same_source = indexes["screenshots"].get(source_hash, [])
    if same_source:
        return "SCREENSHOT_FONTE_ENCONTRADO", same_source
    same_context = indexes["contexts"].get(context, [])
    if same_context:
        bad = [item for item in same_context if item["reason"] != "VERIFICADO_KNN"]
        if bad:
            return "REGISTRO_COMPATIVEL_INELEGIVEL", bad
        return "CONTEXTO_IGUAL_PAR_VISUAL_DIFERENTE", same_context
    if indexes["categories"][(key[0], key[1])] > 0:
        return "CATEGORIA_LUZ_COM_OCR_DIFERENTE", []
    return "SEM_REGISTROS_NA_CATEGORIA_LUZ", []


def reconcile_archive(
    root: Path,
    *,
    inventory: dict | None = None,
    extractor: Callable | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Diagnóstico somente leitura. Nunca grava ou aprova memória."""
    root = Path(root).expanduser().resolve()
    inventory = inventory if inventory is not None else inventory_archives(root)
    if inventory.get("schema") != "visionx.archive_inventory.v1":
        raise ValueError("Inventário incompatível")
    items = inventory.get("images", [])
    if not items or len(items) != inventory.get("summary", {}).get("png_count"):
        raise ValueError("Inventário vazio ou incompleto")
    if extractor is None:
        from src.services.faltando_neural_dataset import AOIPairExtractor
        extractor = AOIPairExtractor()
    dataset = scan_memory_dataset(root)
    indexes = _index_records(dataset["rows"])

    visual_conflicts = set()
    for group in inventory.get("cross_label_conflicts", []):
        visual_conflicts.update(group.get("paths", []))
    cases = []
    for index, item in enumerate(items, 1):
        row = {
            "source_path": item["path"],
            "expected_label": item["expected_label"],
            "category_hint": item["category_hint"],
            "lighting_mode": item["lighting_mode"],
            "status": "NAO_ANALISADO",
            "reason": None,
            "candidates": [], "candidate_count": 0,
            "ocr": None,
            "legacy_compatibility": {
                "status": "NAO_CONSULTADO",
                "candidate_count": 0,
                "candidates": [],
                "verified_in_production": False,
            },
        }
        try:
            if item["status"] != "VALID_PNG":
                row["status"] = "PNG_INVALIDO"
            elif item["path"] in visual_conflicts:
                row["status"] = "ROTULOS_ARQUIVO_CONFLITANTES"
            else:
                path = (root / item["path"]).resolve()
                path.relative_to(root)
                frame = _load_bgr(path)
                reference, test, info = extractor(frame)
                if not isinstance(info, dict):
                    raise ValueError("OCR não retornou metadados")
                category, value = normalize_aoi_text(info.get("value", ""))
                row["ocr"] = {
                    "board": str(info.get("board", "")),
                    "parts": str(info.get("parts", "")),
                    "category": category,
                    "value": str(value),
                }
                if category == "Unknown":
                    row["status"] = "OCR_INVALIDO"
                elif (
                    item.get("category_hint") not in ("UNKNOWN", category)
                ):
                    row["status"] = "OCR_CATEGORIA_DIVERGENTE"
                elif not all((row["ocr"]["board"].strip(),
                              row["ocr"]["parts"].strip(), str(value).strip())):
                    row["status"] = "OCR_INVALIDO"
                else:
                    info = {
                        **info, "category": category, "value": value,
                        "lighting_mode": item["lighting_mode"],
                    }
                    key = VerifiedKNNMemory._key(
                        info, image_fingerprint(reference),
                        image_fingerprint(test),
                    )
                    if key is None:
                        row["status"] = "PAR_OU_OCR_INVALIDO"
                    else:
                        status, candidates = _classify(
                            key, image_fingerprint(frame),
                            item["expected_label"], indexes,
                        )
                        row["status"] = status
                        row["candidate_count"] = len(candidates)
                        row["candidates"] = [
                            _summarize_candidate(entry, root)
                            for entry in candidates[:MAX_CANDIDATES]
                        ]
                        legacy = indexes["legacy_simulated_exact"].get(key, [])
                        legacy_labels = {entry["label"] for entry in legacy}
                        verified_labels = {
                            entry["label"]
                            for entry in indexes["verified"].get(key, [])
                        }
                        if len(legacy_labels | verified_labels) > 1:
                            legacy_status = "LEGADO_CONFLITO_EXATO"
                        elif legacy and row["expected_label"] in legacy_labels:
                            legacy_status = "LEGADO_PAR_SIMULADO_CONCORDA"
                        elif legacy:
                            legacy_status = "LEGADO_PAR_SIMULADO_DIVERGE"
                        else:
                            legacy_status = "SEM_PAR_LEGADO_AUDITAVEL"
                        row["legacy_compatibility"] = {
                            "status": legacy_status,
                            "candidate_count": len(legacy),
                            "candidates": [
                                _summarize_candidate(entry, root)
                                for entry in legacy[:MAX_CANDIDATES]
                            ],
                            # O adapter não instalou o registro no índice
                            # do ODIN, então não é cobertura real da KNN.
                            "verified_in_production": False,
                        }
                        if legacy_status == "LEGADO_CONFLITO_EXATO":
                            row["status"] = "CONFLITO_COM_MEMORIA_LEGADA"
        except (OSError, ValueError, TypeError, cv2.error) as exc:
            row["status"] = "EXTRACAO_OU_LEITURA_INVALIDA"
            row["reason"] = f"{type(exc).__name__}: {exc}"
        cases.append(row)
        if progress is not None:
            progress(index, len(items), row["source_path"])
    legacy_profile = summarize_legacy_profiles(dataset["rows"])
    legacy_case_statuses = dict(sorted(Counter(
        row["legacy_compatibility"]["status"] for row in cases
    ).items()))
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": "READ_ONLY_DIAGNOSTIC",
        "writes_to_dataset": False,
        "trains_models": False,
        "changes_verdicts": False,
        "startup_gate_enabled": False,
        "archive_png_count": len(items),
        "memory_json_count": len(dataset["rows"]),
        "memory_record_reasons": dataset["counts"],
        "legacy_memory_profiles": legacy_profile,
        "legacy_record_diagnostics": [
            {
                "path": item["path"],
                "original_rejection": item.get("reason"),
                "schema": item["legacy_profile"]["schema"],
                "status": item["legacy_profile"]["status"],
                "human_confirmed": item["legacy_profile"]["human_confirmed"],
                "signature_present": item["legacy_profile"]["signature_present"],
                "reference_png_present": item["legacy_profile"]["reference_png_present"],
                "test_png_present": item["legacy_profile"]["test_png_present"],
                "has_dedup_pointer": item["legacy_profile"]["has_dedup_pointer"],
                "declared_test_fingerprint_matches": item["legacy_profile"][
                    "declared_test_fingerprint_matches"
                ],
            }
            for item in dataset["rows"] if isinstance(
                item.get("legacy_profile"), dict
            )
        ],
        "legacy_case_status_counts": legacy_case_statuses,
        "case_status_counts": dict(sorted(Counter(
            row["status"] for row in cases
        ).items())),
        "by_label": {
            label: dict(sorted(Counter(
                row["status"] for row in cases
                if row["expected_label"] == label
            ).items()))
            for label in ("OK", "NG")
        },
        "cases": cases,
        "important": (
            "Somente PAR_VERIFICADO confirma memória KNN atualmente "
            "recuperável. LEGADO_PAR_SIMULADO_CONCORDA não é KNOWN: "
            "descreve compatibilidade potencial em memória, sem modificar "
            "o índice de produção. Conflitos e provas incompletas "
            "não autorizam migração automática ou aprovação do gate."
        ),
    }


def write_reconciliation_report(report: dict, output_dir: Path):
    root = Path(report["root"]).resolve()
    output = Path(output_dir).resolve()
    for name in ("ok_archive", "ng_archive", "dataset"):
        protected = (root / "public" / name).resolve()
        if output == protected or protected in output.parents:
            raise ValueError("Não escrever relatório dentro de archive/dataset")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = output / f"reconciliation_{stamp}.json"
    txt_path = output / f"reconciliation_{stamp}.txt"
    _atomic_text(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    text = [
        "ODIN — RECONCILIAÇÃO DO ACERVO E MEMÓRIA KNN",
        "APENAS LEITURA | SEM MIGRAÇÃO | SEM TREINAMENTO | SEM GATE",
        "",
        f"PNG no acervo: {report['archive_png_count']}",
        f"JSONs inspecionados na memória: {report['memory_json_count']}",
        "",
        "MOTIVOS DE EXCLUSÃO DOS JSONs DA MEMÓRIA:",
    ]
    for key, count in report["memory_record_reasons"].items():
        text.append(f"  {key}: {count}")
    text.extend(["", "SCHEMAS LEGADOS IDENTIFICADOS:"])
    for key, count in report.get("legacy_memory_profiles", {}).get(
        "schema_distribution", {}
    ).items():
        text.append(f"  {key}: {count}")
    text.append("")
    text.append("ELEGIBILIDADE DOS FORMATOS LEGADOS (SIMULAÇÃO):")
    for key, count in report.get("legacy_memory_profiles", {}).get(
        "eligibility_reasons", {}
    ).items():
        text.append(f"  {key}: {count}")
    text.append("")
    text.append("RESULTADO SIMULADO POR PNG, SEM EFEITO NA PRODUÇÃO:")
    for key, count in report.get("legacy_case_status_counts", {}).items():
        text.append(f"  {key}: {count}")
    text.append("")
    text.append("DIAGNÓSTICO DOS PNGs DO ACERVO:")
    for key, count in report["case_status_counts"].items():
        text.append(f"  {key}: {count}")
    text.extend(["", "CASOS SEM CORRESPONDÊNCIA VERIFICADA:"])
    for item in report["cases"]:
        if item["status"] == "PAR_VERIFICADO":
            continue
        candidates = ", ".join(
            f"{c['record']} [{c['eligibility']}]"
            for c in item["candidates"]
        )
        legacy = item.get("legacy_compatibility") or {}
        legacy_candidates = ", ".join(
            str(c["record"]) for c in legacy.get("candidates", [])
        )
        text.append(
            f"- {item['expected_label']} | {item['source_path']} | "
            f"{item['status']} | candidatos={item['candidate_count']} | "
            f"{candidates or item['reason'] or '-'} | "
            f"legado={legacy.get('status', 'N/D')}, "
            f"pares={legacy.get('candidate_count', 0)} "
            f"{legacy_candidates}"
        )
    text.extend([
        "", report["important"],
        "Os arquivos originais permanecem intactos.",
    ])
    _atomic_text(txt_path, "\n".join(text) + "\n")
    return json_path, txt_path


__all__ = [
    "SCHEMA", "scan_memory_dataset", "reconcile_archive",
    "write_reconciliation_report",
]
