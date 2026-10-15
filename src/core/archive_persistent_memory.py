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
from copy import deepcopy
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from threading import RLock
import tempfile
import time

from src.config.settings import BASE_DIR
from src.core.anomaly_signature import valid_anomaly_signature
from src.core.strict_category_memory import (
    canonical_memory_category, canonical_memory_lighting,
)
from src.services.image_archive_dedup import image_fingerprint

SCHEMA = "visionx.archive_persistent_knn.v1"
DEFAULT_RELATIVE_PATH = Path("data") / "knn_archive_memory" / "memory.json"
_HEX = set("0123456789abcdef")


@contextmanager
def _writer_lock(lock_path: Path):
    """Lock entre processos: CLI manual e observador não perdem entradas.

    Evita read-modify-write concorrente em Windows (msvcrt) e Linux (flock).
    Não remove o .lock, pois excluir o inode enquanto um processo o possui
    pode criar dois cadeados diferentes no mesmo diretório.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise ValueError("Arquivo de lock KNN não pode ser symlink")
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            # A posição inicial do lock será inicializada APÓS adquirir
            # exclusividade. Ler/escrever o byte antes do lock cria corrida no
            # Windows e pode lançar PermissionError no segundo escritor.
        else:
            import fcntl
        acquired = False
        deadline = time.monotonic() + 30.0
        while not acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Outra sincronização KNN mantém o lock de escrita")
                time.sleep(0.1)
        try:
            if os.name == "nt":
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_replace_windows_retry(temp_name: str, target: Path) -> None:
    """Publica o snapshot sem perder atomicidade em compartilhamento Win32.

    No Windows, leitores externos/antivírus podem temporariamente abrir
    memory.json sem FILE_SHARE_DELETE. Isso pode gerar WinError 5/32 durante
    os.replace(), mesmo com o lock exclusivo entre *escritores* KNN.
    Nunca remove/trunca o destino para contornar essa falha. Se o bloqueio
    persistir, propaga a exceção e o chamador conserva o snapshot anterior.
    """
    for attempt in range(5):
        try:
            os.replace(temp_name, target)
            return
        except PermissionError as exc:
            if (
                os.name != "nt"
                or getattr(exc, "winerror", None) not in (5, 32)
                or attempt == 4
            ):
                raise
            time.sleep(0.05 * (2 ** attempt))


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
        if not isinstance(source, dict):
            raise ValueError("Proveniência da memória inválida")
        kind = source.get("source_kind", "ARCHIVE_PNG")
        if kind == "OPERATOR_CONFIRMED":
            # A confirmação humana usa o par AOI capturado na própria
            # inspeção e não depende do arquivo visual de arquivo opcional.
            if (source.get("source_path", "").startswith("operator_confirmed/")
                    and source.get("source_label") == item["label"]
                    and source.get("source") in
                    {"button", "xp_keyboard", "keyboard", "manual",
                     "operator", "physical_keyboard", "human"}
                    and _is_hex_hash(source.get("source_sha256"))):
                continue
        elif kind == "ARCHIVE_PNG":
            if (isinstance(source.get("source_path"), str)
                    and _is_hex_hash(source.get("source_sha256"))
                    and (source["source_path"].startswith("public/ok_archive/")
                         if item["label"] == "OK" else
                         source["source_path"].startswith("public/ng_archive/"))):
                continue
        raise ValueError("Proveniência de memória OK/NG não confiável")


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
        with self._lock, _writer_lock(self.path.with_suffix(".json.lock")):
            old = self._read()  # não converter snapshot corrompido em vazio
            by_id = {r["id"]: deepcopy(r) for r in old}
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
                    _atomic_replace_windows_retry(temp_name, self.path)
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

    def replace_verified_labels(self, corrected_ng_records: list[dict]) -> dict:
        """Corrige identidades EXATAS: elimina rótulos antigos apenas dessas chaves.

        Exclusivo de manutenção offline com vínculo visual confirmado. Não é
        chamado no julgamento normal nem pelo observador de arquivos.
        """
        if not corrected_ng_records:
            raise ValueError("Nenhuma correção validada")
        for record in corrected_ng_records:
            _check_record(record)
            if record["label"] != "NG":
                raise ValueError("Correção autorizada somente para NG")
        keys = {tuple(r["identity"]) for r in corrected_ng_records}
        with self._lock, _writer_lock(self.path.with_suffix(".json.lock")):
            old = self._read()
            preserved = [
                deepcopy(r) for r in old
                if tuple(r["identity"]) not in keys
            ]
            corrected = {
                r["id"]: deepcopy(r) for r in old
                if tuple(r["identity"]) in keys and r["label"] == "NG"
            }
            removed_ok = sum(
                r["label"] == "OK" and tuple(r["identity"]) in keys
                for r in old
            )
            for record in corrected_ng_records:
                record_id = record["id"]
                if record_id not in corrected:
                    corrected[record_id] = deepcopy(record)
                    continue
                existing = corrected[record_id]
                seen = {
                    (s["source_path"], s["source_sha256"])
                    for s in existing["sources"]
                }
                for source in record["sources"]:
                    marker = (source["source_path"], source["source_sha256"])
                    if marker not in seen:
                        existing["sources"].append(deepcopy(source))
                        seen.add(marker)
            rows = sorted(preserved + list(corrected.values()), key=lambda r: r["id"])
            for item in rows:
                _check_record(item)
            labels_by_key = _labels_by_key(rows)
            if any(len(labels_by_key.get(key, set())) > 1 for key in keys):
                raise ValueError("Conflito de rótulo após retificação")

            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_symlink():
                raise ValueError("Snapshot KNN não pode ser symlink")
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
                _atomic_replace_windows_retry(temp_name, self.path)
            finally:
                if temp_name and os.path.exists(temp_name):
                    os.unlink(temp_name)
            self._cached = None
            self._stat = None
            result = {tuple(record["identity"]): record["label"]
                      for record in self.records()}
            if any(result.get(key) != "NG" for key in keys):
                raise ValueError("Verificação do snapshot corrigido falhou")
            return {
                "identities_corrected": len(keys),
                "old_ok_removed": removed_ok,
                "total": len(rows),
                "conflicting_identities": sum(
                    len(labels) > 1 for labels in _labels_by_key(rows).values()
                ),
            }

    def lookup_identity(self, key: list[str]) -> dict:
        """Lookup exato em snapshot persistido, para teste após sincronização."""
        key = identity(*key)
        matches = [r for r in self.records() if r["identity"] == key]
        if not matches:
            return {"status": "NEW", "reason": "Par não consta da memória durável"}
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
        return self.lookup_identity(key)


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


def make_operator_record(*, label: str, source: str,
                         event_id: str, category: str, lighting: str,
                         reference, test, anomaly_signature: dict,
                         aoi_info: dict | None = None) -> dict:
    """Registra um julgamento humano real SEM fabricar caminho PNG.

    O caso pode ser reapresentado no mesmo segundo, mesmo quando o
    arquivamento visual estiver desativado ou atrasado.
    """
    normalized = str(source or "").strip().lower()
    if normalized not in {
        "button", "xp_keyboard", "keyboard", "manual", "operator",
        "physical_keyboard", "human",
    }:
        raise ValueError("Predição automática não é rótulo para aprendizado")
    if label not in ("OK", "NG"):
        raise ValueError("Rótulo humano inválido")
    key = identity(category, lighting,
                   image_fingerprint(reference), image_fingerprint(test))
    digest = sha256(_canonical([key, label]).encode("utf-8")).hexdigest()
    # A identidade do registro é estável e colapsa confirmações iguais;
    # evento só identifica a proveniência de cada confirmação.
    import uuid
    source_id = uuid.uuid4().hex
    entry = {
        "id": digest, "identity": key, "label": label,
        "anomaly_signature": anomaly_signature,
        "aoi_info": {
            k: str((aoi_info or {}).get(k, "") or "")
            for k in ("board", "parts", "value")
        },
        "sources": [{
            "source_path": "operator_confirmed/" + source_id,
            "source_sha256": sha256(
                (source_id + str(event_id)).encode("utf-8")
            ).hexdigest(),
            "source_kind": "OPERATOR_CONFIRMED",
            "source_label": label, "source": normalized,
            "event_id": str(event_id or ""),
        }],
    }
    _check_record(entry)
    return entry


__all__ = [
    "PersistentArchiveKNN", "make_record", "make_operator_record", "SCHEMA",
]
