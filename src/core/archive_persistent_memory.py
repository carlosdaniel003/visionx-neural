"""Memória durável de imagens OK/NG previamente curadas no arquivo AOI.

Fonte de rótulos: pastas OK/NG explicitamente designadas pelo operador.
Não exige reter PNGs após importação. Somente um match exato do par
gabarito/teste e da categoria/iluminação pode dispensar a CNN; assinaturas
aproximadas são armazenadas, mas NUNCA autorizam OK automático.

Snapshot imutável por ausência de fonte: só acrescenta registros. Armazenar
fora do arquivo/dataset, fazer cópia .bak em cada atualização e backup externo.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from threading import RLock
import tempfile

from src.config.settings import BASE_DIR
from src.core.anomaly_signature import valid_anomaly_signature
from src.core.strict_category_memory import (
    canonical_memory_category, canonical_memory_lighting,
)
from src.services.image_archive_dedup import image_fingerprint

SCHEMA = "visionx.archive_persistent_knn.v1"
DEFAULT_RELATIVE_PATH = Path("data") / "knn_archive_memory" / "memory.json"
_HEX = set("0123456789abcdef")


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _checksum(entries: list[dict]) -> str:
    return sha256(_canonical(entries).encode("utf-8")).hexdigest()


def _is_hex_hash(h) -> bool:
    return isinstance(h, str) and len(h) == 64 and set(h.lower()) <= _HEX


def _clean(value) -> str:
    return "".join(c for c in str(value or "").upper() if c.isalnum())


def identity(category: str, lighting: str, ref_hash: str, test_hash: str) -> list[str]:
    cat = canonical_memory_category(category)
    mode = canonical_memory_lighting(lighting)
    if not cat or cat in {"UNKNOWN", "UNKNOW", "DESCONHECIDO"}:
        raise ValueError("Categoria desconhecida na memória")
    if mode not in {"SIDE", "TOP", "MID"}:
        raise ValueError("Iluminação inválida")
    if not _is_hex_hash(ref_hash) or not _is_hex_hash(test_hash):
        raise ValueError("Hash de imagem inválido")
    return [cat, mode, ref_hash, test_hash]


def _check_record(item: dict):
    if not isinstance(item, dict) or item.get("label") not in ("OK", "NG"):
        raise ValueError("Rótulo persistente inválido")
    key = item.get("identity")
    if (not isinstance(key, list) or len(key) != 4
            or identity(*key) != key):
        raise ValueError("Identidade KNN persistente inválida")
    if not valid_anomaly_signature(item.get("anomaly_signature")):
        raise ValueError("Assinatura KNN persistente inválida")
    record_id = sha256(_canonical([key, item["label"]]).encode("utf-8")).hexdigest()
    if item.get("id") != record_id:
        raise ValueError("ID da memória diverge de conteúdo")
    if not isinstance(item.get("sources"), list) or not item["sources"]:
        raise ValueError("Registro sem proveniência do arquivo")
    for source in item["sources"]:
        if (not isinstance(source, dict)
                or not _is_hex_hash(source.get("source_sha256"))
                or not isinstance(source.get("source_path"), str)
                or not (source["source_path"].startswith("public/ok_archive/")
                        if item["label"] == "OK" else
                        source["source_path"].startswith("public/ng_archive/"))):
            raise ValueError("Proveniência do arquivo inválida")


class PersistentArchiveKNN:
    """Indexa em memória RAM cópia versionada em disco, independente das fotos."""

    def __init__(self, root: Path | None = None, *, path: Path | None = None):
        self.root = Path(root or BASE_DIR).expanduser().resolve()
        self.path = Path(path or self.root / DEFAULT_RELATIVE_PATH).expanduser().resolve()
        for protected in (self.root / "public" / "ok_archive",
                          self.root / "public" / "ng_archive",
                          self.root / "public" / "dataset"):
            dest = protected.resolve()
            if self.path == dest or dest in self.path.parents:
                raise ValueError("Memória não pode ficar em arquivo/dataset")
        self._lock = RLock()
        self._cached = None
        self._stat = None

    def _read(self) -> list[dict]:
        if self.path.is_symlink():
            raise ValueError("Snapshot KNN não pode ser symlink")
        if not self.path.exists():
            return []
        if not self.path.is_file():
            raise ValueError("Snapshot KNN não é um arquivo")
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if (raw.get("schema") != SCHEMA
                or not isinstance(raw.get("records"), list)
                or raw.get("checksum") != _checksum(raw["records"])):
            raise ValueError("Snapshot KNN inválido ou danificado: importação bloqueada")
        entries = raw["records"]
        ids = set()
        for item in entries:
            _check_record(item)
            if item["id"] in ids:
                raise ValueError("Snapshot KNN contém ID duplicado")
            ids.add(item["id"])
        return entries

    def records(self) -> list[dict]:
        with self._lock:
            stat = self.path.stat() if self.path.exists() else None
            stamp = ((stat.st_mtime_ns, stat.st_size) if stat else None)
            if self._cached is None or stamp != self._stat:
                entries = self._read()
                self._cached = entries
                self._stat = stamp
            return list(self._cached)

    def add(self, proposed: list[dict]) -> dict:
        """Append-only, nunca remove um exemplo antigo se o PNG foi excluído."""
        with self._lock:
            old = self._read()  # não converter snapshot corrompido em vazio
            by_id = {r["id"]: r for r in old}
            new_count = 0
            for record in proposed:
                _check_record(record)
                identifier = record["id"]
                if identifier not in by_id:
                    by_id[identifier] = record
                    new_count += 1
                else:
                    previous = by_id[identifier]
                    known = {
                        (s["source_path"], s["source_sha256"])
                        for s in previous["sources"]
                    }
                    for source in record["sources"]:
                        if (source["source_path"], source["source_sha256"]) not in known:
                            previous["sources"].append(source)
                            known.add((source["source_path"], source["source_sha256"]))
            rows = sorted(by_id.values(), key=lambda r: r["id"])
            changed = _canonical(rows) != _canonical(old)
            if changed:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.is_symlink():
                    raise ValueError("Snapshot KNN symlink: bloqueado")
                temp_name = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", encoding="utf-8", newline="\n",
                        dir=self.path.parent, prefix="._knn_",
                        suffix=".tmp", delete=False
                    ) as handle:
                        temp_name = handle.name
                        handle.write(_canonical({
                            "schema": SCHEMA, "checksum": _checksum(rows),
                            "records": rows,
                        }) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    if self.path.exists():
                        shutil.copy2(self.path, self.path.with_suffix(".json.bak"))
                    os.replace(temp_name, self.path)
                finally:
                    if temp_name and os.path.exists(temp_name):
                        os.unlink(temp_name)
            self._cached = None
            self._stat = None
            return {
                "total": len(rows), "new": new_count,
                "changed": changed, "path": str(self.path),
                "by_label": {
                    label: sum(r["label"] == label for r in rows)
                    for label in ("OK", "NG")
                },
                "conflicting_identities": sum(
                    len(labels) > 1 for labels in
                    _labels_by_key(rows).values()
                ),
            }

    def lookup(self, reference, test, info: dict) -> dict:
        if (reference is None or test is None
                or getattr(reference, "size", 0) == 0
                or getattr(test, "size", 0) == 0):
            return {"status": "UNAVAILABLE", "reason": "Par visual indisponível"}
        try:
            key = identity(
                info.get("category"), info.get("lighting_mode"),
                image_fingerprint(reference), image_fingerprint(test)
            )
        except ValueError as exc:
            return {"status": "NEW", "reason": str(exc)}
        # Falha de integridade dispara exceção; não converter em memória vazia.
        matches = [r for r in self.records() if r["identity"] == key]
        if not matches:
            return {"status": "NEW", "reason": "Par visual não consta da memória arquivada"}
        labels = {r["label"] for r in matches}
        if len(labels) != 1:
            return {
                "status": "CONFLICT", "matches": len(matches),
                "reason": "Par idêntico com rótulos OK e NG",
            }
        return {
            "status": "KNOWN", "label": labels.pop(), "matches": len(matches),
            "source_json": str(self.path),
            "reason": "Par visual exato da memória durável dos arquivos OK/NG",
            "memory_source": "ARCHIVE_PERSISTENT",
        }


def _labels_by_key(rows):
    out = defaultdict(set)
    for r in rows:
        out[tuple(r["identity"])].add(r["label"])
    return out


def make_record(*, label: str, source_path: str, source_sha256: str,
                category: str, lighting: str, reference, test,
                anomaly_signature: dict, aoi_info: dict | None = None) -> dict:
    if label not in {"OK", "NG"} or not _is_hex_hash(source_sha256):
        raise ValueError("Fonte de arquivo/rótulo inválidos")
    required_dir = "ok_archive" if label == "OK" else "ng_archive"
    if (not source_path.startswith(f"public/{required_dir}/")
            or ".." in Path(source_path).parts):
        raise ValueError("Caminho de arquivo não autorizado")
    key = identity(category, lighting,
                   image_fingerprint(reference), image_fingerprint(test))
    record_id = sha256(_canonical([key, label]).encode("utf-8")).hexdigest()
    result = {
        "id": record_id, "identity": key, "label": label,
        "anomaly_signature": anomaly_signature,
        "aoi_info": {
            k: str((aoi_info or {}).get(k, "") or "")
            for k in ("board", "parts", "value")
        },
        "sources": [{
            "source_path": source_path, "source_sha256": source_sha256
        }],
    }
    _check_record(result)
    return result


__all__ = ["PersistentArchiveKNN", "make_record", "SCHEMA"]
