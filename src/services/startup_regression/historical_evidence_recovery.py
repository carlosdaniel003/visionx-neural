"""Recuperação experimental, SOMENTE LEITURA, de evidências visuais legadas.

Procura PNGs dentro de public/dataset e dos arquivos OK/NG. Os nomes
encontrados são pistas, não identidade. Associações comprováveis exigem
hash visual declarado ou um vínculo explícito no JSON, com verificação
separada de origem humana, metadados OCR e de ambos os recortes.

Sem alteração de dataset, sem treinamento, sem criação de memória KNN,
sem alteração do main.py e SEM usar o rótulo da pasta para vincular PNGs.
O resultado READY_FOR_REVIEW não é KNOWN nem autoriza migração automática.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Callable

import cv2
import numpy as np

from src.core.strict_category_memory import (
    canonical_memory_category, canonical_memory_lighting,
)
from src.core.verified_memory_router import VerifiedKNNMemory, _clean
from src.services.image_archive_dedup import image_fingerprint
from src.services.startup_regression.archive_inventory_report import _atomic_text
from src.utils.text_normalizer import normalize_aoi_text

from .archive_reconciler import DATA_FOLDERS
from .legacy_memory_compat import inspect_legacy_record

SCHEMA = "visionx.historical_evidence_recovery.v1"
SCAN_DIRS = ("public/dataset", "public/ok_archive", "public/ng_archive")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MAX_PATH_EXAMPLES = 4


def _png_basename(value) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    # Recusar arquivo fora da árvore, UNC/absoluto, separadores e traversal.
    if any(x in value for x in ("/", "\\", "\x00")):
        return None
    p = Path(value)
    if value != p.name or p.suffix.lower() != ".png" or value in (".", ".."):
        return None
    return value


def _sha(value) -> str | None:
    val = str(value or "").strip()
    return val.lower() if SHA256_RE.fullmatch(val) else None


def _bgr(path: Path):
    img = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8),
                       cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("PNG ilegível")
    return img


def _safe_under(path: Path, parent: Path) -> bool:
    if path.is_symlink():
        return False
    try:
        path.resolve(strict=True).relative_to(parent.resolve(strict=True))
        return True
    except (OSError, ValueError, RuntimeError):
        return False


class EvidenceIndex:
    """Indexa hashes de pixels somente nos três locais pré-aprovados."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.files = []
        self.by_name = defaultdict(list)
        self.by_hash = defaultdict(list)
        self.scan_errors = []
        self.scan()

    def scan(self):
        seen = set()
        public = self.root / "public"
        for name in SCAN_DIRS:
            base = self.root / name
            if not base.is_dir() or base.is_symlink() or not _safe_under(base, public):
                continue
            for folder, dirs, files in os.walk(base, followlinks=False):
                here = Path(folder)
                dirs[:] = sorted(
                    d for d in dirs
                    if not (here / d).is_symlink()
                    and _safe_under(here / d, public)
                )
                for filename in sorted(files):
                    if not filename.lower().endswith(".png"):
                        continue
                    path = here / filename
                    if not _safe_under(path, public) or path in seen:
                        continue
                    seen.add(path)
                    try:
                        digest = image_fingerprint(_bgr(path))
                        if not digest:
                            raise ValueError("Hash visual indisponível")
                        relative = path.relative_to(self.root).as_posix()
                        item = {"path": relative, "fingerprint": digest}
                        self.files.append(item)
                        self.by_name[filename].append(item)
                        self.by_hash[digest].append(item)
                    except (OSError, ValueError, cv2.error) as exc:
                        self.scan_errors.append({
                            "path": path.relative_to(self.root).as_posix(),
                            "error": f"{type(exc).__name__}: {exc}",
                        })

    def find(self, *, declared_name=None, declared_sha=None, local_parent=None):
        """Retorna pistas por nome e SHA de pixels, com proveniência separada."""
        name = _png_basename(declared_name)
        digest = _sha(declared_sha)
        by_name = list(self.by_name.get(name, [])) if name else []
        by_hash = list(self.by_hash.get(digest, [])) if digest else []
        matched = {item["path"]: item for item in by_name + by_hash}
        strong = [
            item for item in matched.values()
            if digest and item["fingerprint"] == digest
        ]
        local = [
            item for item in by_name
            if local_parent is not None and
            (self.root / item["path"]).parent == local_parent
        ]
        return {
            "declared_filename": name,
            "declared_hash_present": bool(digest),
            "declared_hash_invalid": bool(declared_sha) and not digest,
            "name_candidates": len(by_name),
            "hash_candidates": len(by_hash),
            "same_folder_filename_found": bool(local),
            "hash_verified": bool(strong),
            "unique_hashes_for_name": len({
                item["fingerprint"] for item in by_name
            }),
            "located_paths": sorted(matched)[:MAX_PATH_EXAMPLES],
            "hash_match_paths": sorted(
                item["path"] for item in strong
            )[:MAX_PATH_EXAMPLES],
            # Interno, nunca exposto diretamente no relatório.
            "_hash_matches": strong,
            "_name_matches": by_name,
            "_local_matches": local,
        }


def _external_view(loc: dict) -> dict:
    return {k: v for k, v in loc.items() if not k.startswith("_")}


def _record_paths(root: Path):
    public = root / "public"
    for dirname, label in DATA_FOLDERS:
        base = root / "public" / "dataset" / dirname
        if not base.exists() or base.is_symlink() or not _safe_under(base, public):
            continue
        for folder, dirs, files in os.walk(base, followlinks=False):
            here = Path(folder)
            dirs[:] = sorted(d for d in dirs if not (here / d).is_symlink()
                             and _safe_under(here / d, public))
            for filename in sorted(files):
                path = here / filename
                if path.suffix.lower() == ".json" and _safe_under(path, public):
                    yield path, label


def _strong_candidates(location: dict):
    return sorted(location["_hash_matches"], key=lambda x: x["path"])


def _unique_fingerprint(loc: dict):
    matches = _strong_candidates(loc)
    keys = {m["fingerprint"] for m in matches}
    return (next(iter(keys)), matches[0]["path"]) if len(keys) == 1 else (None, None)


def _check_extraction(
    root: Path, data: dict, source_path: str,
    extractor: Callable,
    reference_loc: dict, test_loc: dict,
):
    """Reextrai do frame COM VÍNCULO DE HASH, sem ler o rótulo do arquivo."""
    output = {
        "attempted": True, "source_path": source_path,
        "status": "EXTRACTION_NOT_VERIFIED",
        "key_preview": None,
        "same_stored_ocr": False,
        "declared_reference_hash_matches": None,
        "declared_test_hash_matches": None,
    }
    info_old = data.get("aoi_info") or {}
    storage = data.get("storage") or {}
    try:
        frame = _bgr(root / source_path)
        reference, test, observed = extractor(frame)
        if not isinstance(observed, dict):
            raise ValueError("OCR não retornou objeto")
        category, value = normalize_aoi_text(observed.get("value", ""))
        old_category = canonical_memory_category(info_old.get("category"))
        observed_category = canonical_memory_category(category)
        info = {
            **observed, "category": category, "value": value,
            "lighting_mode": info_old.get("lighting_mode", "SIDE"),
        }
        # Nunca completar OCR faltante usando o conteúdo do registro antigo.
        same_ocr = (
            bool(old_category) and old_category == observed_category
            and bool(_clean(info_old.get("board")))
            and _clean(info_old.get("board")) == _clean(observed.get("board"))
            and bool(_clean(info_old.get("parts")))
            and _clean(info_old.get("parts")) == _clean(observed.get("parts"))
            and bool(_clean(info_old.get("value")))
            and _clean(info_old.get("value")) == _clean(value)
        )
        output["same_stored_ocr"] = same_ocr
        ref_hash = image_fingerprint(reference)
        test_hash = image_fingerprint(test)
        declared_ref = _sha(storage.get("reference_image_fingerprint"))
        declared_test = _sha(storage.get("test_image_fingerprint"))
        output["declared_reference_hash_matches"] = (
            ref_hash == declared_ref if declared_ref else None
        )
        output["declared_test_hash_matches"] = (
            test_hash == declared_test if declared_test else None
        )
        if not same_ocr:
            output["status"] = "RECONSTRUCAO_OCR_DIVERGENTE"
        elif declared_ref and ref_hash != declared_ref:
            output["status"] = "RECONSTRUCAO_GABARITO_HASH_DIVERGENTE"
        elif declared_test and test_hash != declared_test:
            output["status"] = "RECONSTRUCAO_TESTE_HASH_DIVERGENTE"
        elif not (declared_ref and declared_test):
            output["status"] = "RECONSTRUCAO_SEM_HASH_DOS_DOIS_RECORTE"
        else:
            key = VerifiedKNNMemory._key(info, ref_hash, test_hash)
            if key is None:
                output["status"] = "RECONSTRUCAO_CHAVE_INVALIDA"
            else:
                output["status"] = "RECONSTRUCAO_COM_HASHES_CONFERIDOS"
                # A chave é um teste simulado, sem injetar memória.
                output["key_preview"] = list(key)
    except (OSError, ValueError, TypeError, cv2.error) as exc:
        output["status"] = "RECONSTRUCAO_FALHOU"
        output["error"] = f"{type(exc).__name__}: {exc}"
    return output


def inspect_record(
    root: Path, path: Path, folder_label: str,
    index: EvidenceIndex, *, extractor: Callable | None = None,
) -> dict:
    row = {
        "path": path.relative_to(root).as_posix(),
        "status": "NAO_AVALIADO", "schema": None,
        "human_confirmed": False, "signature_present": False,
        "record_sha256": None,
        "original_audit_pair_present": False,
        "source": None, "test": None, "reference": None,
        "reconstruction": None, "migration_ready": False,
    }
    try:
        json_bytes = path.read_bytes()
        row["record_sha256"] = hashlib.sha256(json_bytes).hexdigest()
        data = json.loads(json_bytes.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON deve conter objeto")
        row["schema"] = str(data.get("schema") or "SEM_SCHEMA")
        if row["schema"] == "visionx.memory.v3":
            row["status"] = "ATUAL_V3_FORA_ESCOPO"
            return row
        profile = inspect_legacy_record(path, folder_label, data)
        row["human_confirmed"] = profile["human_confirmed"]
        row["signature_present"] = profile["signature_present"]
        row["original_audit_pair_present"] = (
            profile["reference_png_present"] and profile["test_png_present"]
        )
        storage = data.get("storage") or {}
        if not isinstance(storage, dict):
            storage = {}
        declared_source = storage.get("source_image_fingerprint")
        # source_frame_fingerprint só é aceito se for explicitamente um
        # hash de pixels gerado pelo produtor; não misturar com SHA do PNG.
        source_loc = index.find(
            declared_name=storage.get("source_image_file"),
            declared_sha=declared_source, local_parent=path.parent,
        )
        test_loc = index.find(
            declared_name=storage.get("test_image_file") or data.get("image_file"),
            declared_sha=storage.get("test_image_fingerprint"),
            local_parent=path.parent,
        )
        reference_loc = index.find(
            declared_name=storage.get("reference_image_file"),
            declared_sha=storage.get("reference_image_fingerprint"),
            local_parent=path.parent,
        )
        row["source"] = _external_view(source_loc)
        row["test"] = _external_view(test_loc)
        row["reference"] = _external_view(reference_loc)

        if not row["human_confirmed"]:
            row["status"] = "ORIGEM_HUMANA_NAO_COMPROVADA"
        elif not row["signature_present"]:
            row["status"] = "ASSINATURA_DE_ANOMALIA_AUSENTE"
        elif profile["status"] == "LEGADO_PAR_AUDITAVEL_SIMULADO":
            row["status"] = "PAR_LEGADO_JA_AUDITAVEL"
        elif source_loc["declared_hash_invalid"] or test_loc[
            "declared_hash_invalid"
        ] or reference_loc["declared_hash_invalid"]:
            row["status"] = "HASH_DECLARADO_INVALIDO"
        elif source_loc["hash_verified"] and extractor is not None:
            _, source_path = _unique_fingerprint(source_loc)
            if source_path is None:
                row["status"] = "FONTE_HASH_AMBIGUO"
            else:
                row["reconstruction"] = _check_extraction(
                    root, data, source_path, extractor, reference_loc, test_loc,
                )
                if row["reconstruction"]["status"] == (
                    "RECONSTRUCAO_COM_HASHES_CONFERIDOS"
                ):
                    row["status"] = "PAR_RECONSTRUIDO_PARA_REVISAO"
                else:
                    row["status"] = row["reconstruction"]["status"]
        elif source_loc["hash_verified"] and extractor is None:
            row["status"] = "FONTE_VERIFICADA_AGUARDA_EXTRATOR"
        elif (
            source_loc["_local_matches"]
            and not source_loc["declared_hash_present"]
            and extractor is not None
        ):
            # Fonte explicitamente nomeada pelo JSON e existente no
            # MESMO diretório, mas sem prova criptográfica da origem.
            # Extrair é útil para diagnóstico; nunca promover para KNOWN.
            local_sources = source_loc["_local_matches"]
            if len({x["fingerprint"] for x in local_sources}) == 1:
                row["reconstruction"] = _check_extraction(
                    root, data, local_sources[0]["path"], extractor,
                    reference_loc, test_loc,
                )
                row["status"] = "ORIGEM_JSON_LOCAL_SEM_HASH_PARA_REVISAO"
            else:
                row["status"] = "ORIGEM_JSON_LOCAL_AMBIGUA"
        elif reference_loc["hash_verified"] and test_loc["hash_verified"]:
            row["status"] = "DOIS_HASHES_DE_PARES_LOCALIZADOS_PARA_REVISAO"
        elif test_loc["hash_verified"]:
            row["status"] = "TESTE_POR_HASH_SEM_GABARITO"
        elif source_loc["name_candidates"] or test_loc["name_candidates"] or (
            reference_loc["name_candidates"]
        ):
            row["status"] = "ARQUIVOS_POR_NOME_SEM_VINCULO_DE_HASH"
        elif any((
            source_loc["declared_hash_present"],
            reference_loc["declared_hash_present"],
            test_loc["declared_hash_present"],
        )):
            row["status"] = "HASH_DECLARADO_SEM_PNG_COMPATIVEL"
        else:
            row["status"] = "SEM_EVIDENCIA_VISUAL_LOCALIZAVEL"
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        row["status"] = "JSON_INVALIDO"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def inspect_historical_evidence(
    root: Path, *, extractor: Callable | None = None,
    progress: Callable | None = None,
) -> dict:
    """Analisa todos JSONs com schema legado, sem efetuar alterações."""
    root = Path(root).expanduser().resolve()
    index = EvidenceIndex(root)
    rows = []
    records = list(_record_paths(root))
    # Sem carregar CNN ou KNN; ScreenMonitor só no caso de fonte ligada
    # com hash suficiente. Um extrator injetado facilita testes isolados.
    if extractor is None:
        from src.services.faltando_neural_dataset import AOIPairExtractor
        extractor = AOIPairExtractor()
    for i, (path, label) in enumerate(records, 1):
        row = inspect_record(root, path, label, index, extractor=extractor)
        rows.append(row)
        if progress is not None:
            progress(i, len(records), row["path"])
    statuses = Counter(row["status"] for row in rows)
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "mode": "READ_ONLY_NO_MIGRATION",
        "scan_dirs": list(SCAN_DIRS),
        "scanned_pngs": len(index.files),
        "png_scan_errors": index.scan_errors[:50],
        "png_scan_error_count": len(index.scan_errors),
        "scanned_jsons": len(rows),
        "legacy_jsons": sum(
            row["schema"] != "visionx.memory.v3" for row in rows
        ),
        "v3_jsons": sum(
            row["schema"] == "visionx.memory.v3" for row in rows
        ),
        "status_counts": dict(sorted(statuses.items())),
        "by_label": {
            label: dict(sorted(Counter(
                row["status"] for row in rows
                if ("/nao_anomalia/" in row["path"]) == (label == "OK")
            ).items()))
            for label in ("OK", "NG")
        },
        "migration_performed": False,
        "dataset_modified": False,
        "production_knn_modified": False,
        "cnn_modified": False,
        "startup_gate_enabled": False,
        "ready_for_migration": 0,
        "preview_for_manual_review": sum(
            statuses[x] for x in (
                "PAR_RECONSTRUIDO_PARA_REVISAO",
                "DOIS_HASHES_DE_PARES_LOCALIZADOS_PARA_REVISAO",
            )
        ),
        "cases": rows,
        "important": (
            "Hash declarado + PNG existente é pista de identidade. "
            "Mesmo par reconstruído requer revisão e NÃO é KNOWN. "
            "Nenhum rótulo veio do nome da captura ou da pasta OK/NG."
        ),
    }


def write_evidence_report(report: dict, output_dir: Path):
    root = Path(report["root"]).resolve()
    reports_root = (root / "reports").resolve()
    out = Path(output_dir).resolve()
    if out == reports_root or reports_root not in out.parents:
        raise ValueError("Saída permitida somente numa subpasta de reports/")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = out / f"evidence_recovery_{stamp}.json"
    txt_path = out / f"evidence_recovery_{stamp}.txt"
    _atomic_text(
        json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "ODIN — RECUPERAÇÃO DE EVIDÊNCIAS HISTÓRICAS",
        "SOMENTE LEITURA | SEM TREINO | SEM MIGRAÇÃO | SEM STARTUP GATE",
        f"PNG indexados: {report['scanned_pngs']}",
        f"JSONs examinados: {report['scanned_jsons']}",
        f"Legados: {report['legacy_jsons']}",
        "",
        "SITUAÇÃO DOS REGISTROS:",
    ]
    for name, count in report["status_counts"].items():
        lines.append(f"  {name}: {count}")
    lines.append("")
    lines.append("EVIDÊNCIAS RECONSTRUÍDAS OU SUSPEITAS:")
    for item in report["cases"]:
        if item["schema"] == "visionx.memory.v3":
            continue
        if item["status"] not in {
            "SEM_EVIDENCIA_VISUAL_LOCALIZAVEL",
            "PAR_LEGADO_JA_AUDITAVEL",
        }:
            lines.append(
                f"- {item['path']} | {item['status']} | "
                f"source={item['source']['hash_match_paths'] if item['source'] else []} "
                f"| test={item['test']['hash_match_paths'] if item['test'] else []}"
            )
    lines.extend(["", report["important"]])
    _atomic_text(txt_path, "\n".join(lines) + "\n")
    return json_path, txt_path


__all__ = [
    "EvidenceIndex", "inspect_record", "inspect_historical_evidence",
    "write_evidence_report", "SCHEMA",
]
