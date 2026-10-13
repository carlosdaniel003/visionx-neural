"""Importação incremental durável dos PNGs OK e NG + replay de memorização.

    python -m src.services.startup_regression.knn_archive_memory_cli

Diferente do replay leave_day_out: primeiro memoriza rótulos de todas as
capturas do arquivo; depois consulta cada uma pela identidade visual.
A resposta mede RETENÇÃO do acervo, não generalização nem precisão futura.
Memória fica em data/knn_archive_memory/memory.json mesmo após apagar PNGs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

from src.core.anomaly_signature import build_anomaly_signature
from src.core.archive_persistent_memory import (
    PersistentArchiveKNN, make_record,
)
from src.core.strict_category_memory import canonical_memory_category
from src.services.faltando_cross_category_audit import _archive_image
from src.services.startup_regression.archive_inventory import inventory_archives
from src.services.startup_regression.archive_inventory_report import _atomic_text
from src.utils.text_normalizer import normalize_aoi_text

SCHEMA = "visionx.knn_persistent_archive_learning.v1"


def synchronize_archive_knn(
    root: Path, *, inventory: dict | None = None, extractor: Callable | None = None,
    frame_loader: Callable | None = None,
    store: PersistentArchiveKNN | None = None, progress: Callable | None = None,
) -> dict:
    root = Path(root).expanduser().resolve()
    inv = inventory if inventory is not None else inventory_archives(root)
    items = inv.get("images")
    if (inv.get("schema") != "visionx.archive_inventory.v1"
            or not isinstance(items, list)
            or len(items) != inv.get("summary", {}).get("png_count")):
        raise ValueError("Inventário inválido/incompleto")
    if extractor is None:
        from src.services.faltando_neural_dataset import AOIPairExtractor
        extractor = AOIPairExtractor()
    if frame_loader is None:
        frame_loader = _archive_image
    if store is None:
        store = PersistentArchiveKNN(root)
    conflicts = {
        path for group in inv.get("cross_label_conflicts", [])
        for path in group.get("paths", [])
    }

    cases = []
    pending = []
    for i, item in enumerate(items, 1):
        row = {
            "source_path": item.get("path"),
            "source_sha256": item.get("file_sha256"),
            "expected_label": item.get("expected_label"),
            "category_hint": item.get("category_hint"),
            "lighting_mode": item.get("lighting_mode", "SIDE"),
            "status": "NAO_ANALISADO", "recall": None,
            "error": None, "identity": None,
        }
        cases.append(row)
        try:
            if item.get("status") != "VALID_PNG":
                raise ValueError("PNG não validado")
            if row["source_path"] in conflicts:
                raise ValueError("Mesmo PNG tem rótulos OK e NG")
            if row["expected_label"] not in ("OK", "NG"):
                raise ValueError("Rótulo do arquivo inválido")
            frame = frame_loader(root, item)
            ref, test, info = extractor(frame)
            if not isinstance(info, dict):
                raise ValueError("Metadados AOI inválidos")
            observed, value = normalize_aoi_text(info.get("value", ""))
            category = canonical_memory_category(observed)
            hint = canonical_memory_category(row["category_hint"])
            if not category or category in {"UNKNOWN", "UNKNOW"} or category != hint:
                raise ValueError(
                    f"Categoria do OCR {category!r} difere do arquivo {hint!r}"
                )
            # Não depende de OCR de board/parts para reter par visual exato,
            # mas guarda esses campos para auditoria. Sem categoria, não aprende.
            signature = build_anomaly_signature(
                ref, test, {}, {**info, "category": category, "value": value}
            )
            entry = make_record(
                label=row["expected_label"], source_path=row["source_path"],
                source_sha256=row["source_sha256"],
                category=category, lighting=row["lighting_mode"],
                reference=ref, test=test, anomaly_signature=signature,
                aoi_info=info,
            )
            row["identity"] = entry["identity"]
            pending.append(entry)
            row["status"] = "PRONTO"
            row["ocr_missing_fields"] = [
                field for field in ("board", "parts", "value")
                if not str(info.get(field, "") or "").strip()
            ]
        except Exception as exc:
            row["status"] = "INVALIDO"
            row["error"] = f"{type(exc).__name__}: {exc}"
        if progress is not None:
            progress(i, len(items), row["source_path"])

    # NUNCA substituir snapshot por lote vazio devido a erro de OCR/leitura.
    saved = store.add(pending)
    # Nova instância força reabrir snapshot, prova que os dados estão em disco.
    persisted = PersistentArchiveKNN(root, path=store.path)
    for case in cases:
        if case["status"] != "PRONTO":
            continue
        remembered = persisted.lookup_identity(case["identity"])
        case["recall"] = remembered.get("status")
        case["recalled_label"] = remembered.get("label")
        case["status"] = (
            "PASSOU" if remembered.get("status") == "KNOWN"
            and remembered.get("label") == case["expected_label"]
            else "CONFLITO" if remembered.get("status") == "CONFLICT"
            else "REGRESSAO"
        )

    by_label = {}
    for label in ("OK", "NG"):
        subset = [c for c in cases if c["expected_label"] == label]
        correct = sum(c["status"] == "PASSOU" for c in subset)
        by_label[label] = {
            "total": len(subset), "memorized_and_recalled": correct,
            "match_rate": correct / len(subset) if subset else 0.0,
            "failed": len(subset) - correct,
        }
    passed = sum(c["status"] == "PASSOU" for c in cases)
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": "LEARN_ALL_ARCHIVE_THEN_RECALL_IN_SAMPLE",
        "uses_legacy_json": False,
        "uses_cnn": False,
        "training_truth": "OK_AND_NG_ARCHIVE_FOLDER",
        "memory_path": str(store.path),
        "retained_records_survive_archive_file_deletion": True,
        "snapshot": saved,
        "summary": {
            "total": len(cases), "recalled_correct": passed,
            "failed": len(cases) - passed,
            "all_current_archive_memorized": len(cases) > 0 and passed == len(cases),
            "status_counts": dict(sorted(Counter(
                c["status"] for c in cases).items())),
            "by_label": by_label,
            "archive_98pct_target_met": (
                all(v["total"] and v["match_rate"] >= .98
                    for v in by_label.values())
            ),
        },
        "cases": cases,
        "limitations": (
            "O teste confirma recuperação exata dos exemplos aprendidos, "
            "não acerto com imagens novas; rótulos vêm exclusivamente da "
            "pasta OK/NG. Registros antigos não são descartados quando "
            "os PNGs desaparecem. A memória armazena hashes de par AOI e "
            "assinatura visual, não a fotografia. Backup externo do "
            "snapshot é necessário para proteger contra falha de disco. "
            "Casos visualmente idênticos com rótulos opostos ficam "
            "CONFLITO, nunca OK por padrão."
        ),
    }


def write_report(report: dict, output_dir: Path):
    root = Path(report["root"]).resolve()
    out = Path(output_dir).resolve()
    if (root / "reports").resolve() not in out.parents:
        raise ValueError("Relatório deve ser salvo dentro de reports/")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = out / f"knn_archive_memory_{stamp}.json"
    txt_path = out / f"knn_archive_memory_{stamp}.txt"
    _atomic_text(json_path, json.dumps(
        report, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n")
    s = report["summary"]
    lines = [
        "ODIN | IMPORTAR OK/NG PARA MEMÓRIA PERMANENTE KNN",
        "APRENDER TODO ACERVO -> GRAVAR NO DISCO -> REABRIR -> CONSULTAR",
        f"Memória durável: {report['memory_path']}",
        f"Total PNG: {s['total']}",
        f"Memorizados e recuperados: {s['recalled_correct']}/{s['total']}",
        f"Status: {s['status_counts']}",
        f"Meta recuperação por classe >=98%: {s['archive_98pct_target_met']}",
        f"TODO acervo atual persistido: {s['all_current_archive_memorized']}",
        f"Snapshot: {report['snapshot']}",
        "",
        "POR CLASSE:",
    ]
    for label, v in s["by_label"].items():
        lines.append(
            f"{label}: {v['memorized_and_recalled']}/{v['total']} "
            f"({v['match_rate']:.2%}), pendentes={v['failed']}"
        )
    lines.extend(["", "CASOS PENDENTES:"])
    for r in report["cases"]:
        if r["status"] != "PASSOU":
            lines.append(
                f"- {r['status']} | {r['source_path']} "
                f"| esperado={r['expected_label']} "
                f"| obtido={r.get('recalled_label')} | {r['error'] or ''}"
            )
    lines.extend(["", "LIMITAÇÃO:", report["limitations"]])
    _atomic_text(txt_path, "\n".join(lines) + "\n")
    return json_path, txt_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aprende e persiste TODO OK/NG, depois verifica recuperação exata"
    )
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--status", action="store_true",
        help="Verificar memória salva, sem ler PNGs ou executar OCR",
    )
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.status:
        store = PersistentArchiveKNN(root)
        entries = store.records()
        print(
            f"Snapshot KNN: {store.path}\n"
            f"Registros persistidos: {len(entries)}\n"
            f"OK: {sum(r['label'] == 'OK' for r in entries)}\n"
            f"NG: {sum(r['label'] == 'NG' for r in entries)}\n"
            "Integridade: validada (schema, checksum, identidades)",
            flush=True,
        )
        return 0
    print("Sincronizando KNN permanente: OK/NG -> snapshot...", flush=True)

    def progress(i, n, path):
        if i == 1 or i % 25 == 0 or i == n:
            print(f"[{i}/{n}] {path}", flush=True)

    report = synchronize_archive_knn(root, progress=progress)
    j, t = write_report(report, root / "reports" / "startup_regression")
    for label, v in report["summary"]["by_label"].items():
        print(f"{label}: {v['memorized_and_recalled']}/{v['total']}", flush=True)
    print(
        f"Memória salva: {report['memory_path']}\n"
        f"Meta >=98% por classe: {report['summary']['archive_98pct_target_met']}\n"
        f"Todos os PNGs atuais memorizados: {report['summary']['all_current_archive_memorized']}\n"
        f"JSON: {j}\nTXT: {t}",
        flush=True,
    )
    return 0 if report["summary"]["all_current_archive_memorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
