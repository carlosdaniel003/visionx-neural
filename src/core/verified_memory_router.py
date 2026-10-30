"""Roteamento: memória KNN *verificada* primeiro, especialidade para casos novos.

Um match aproximado de embedding/anomaly_signature NÃO significa caso visto.
O atalho exige par gabarito/teste RGB pixel-idêntico, categoria, luz e
rótulo humano verificável. Campos antigos Board/Parts/Value são opcionais
somente na captura atual sem OCR: nunca são inventados ou inferidos.
Não modifica a memória nem retreina a CNN. Instalar após faltando_live.
"""
from __future__ import annotations

from collections import defaultdict
import json
import time
from pathlib import Path
from threading import RLock

import cv2
import numpy as np

from src.core.archive_persistent_memory import PersistentArchiveKNN
from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.strict_category_memory import (
    canonical_memory_category, canonical_memory_lighting,
)
from src.services.image_archive_dedup import image_fingerprint

RECOGNITION_SCHEMA = "visionx.memory_first_routing.v1"


def _clean(value) -> str:
    return "".join(c for c in str(value or "").upper() if c.isalnum())


def _image_ok(image) -> bool:
    return (
        isinstance(image, np.ndarray) and image.ndim == 3
        and image.shape[-1] == 3 and min(image.shape[:2]) >= 12
        and image.dtype == np.uint8 and image.size > 0
    )


def _file_within(parent: Path, name: str) -> Path | None:
    """Somente o PNG em seu próprio diretório, sem traversal ou links."""
    if not isinstance(name, str) or not name or Path(name).name != name:
        return None
    target = parent / name
    if target.suffix.lower() != ".png" or target.is_symlink() or not target.is_file():
        return None
    return target


def _fingerprint_png(path: Path) -> str:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if not _image_ok(image):
        return ""
    return image_fingerprint(image)


class VerifiedKNNMemory:
    """Índice da memória KNN, apenas entradas humanas com dois PNGs auditáveis."""

    def __init__(self, *, archive_store=None):
        self.archive_store = archive_store if archive_store is not None else PersistentArchiveKNN()
        self._lock = RLock()
        self._index: dict[tuple, list[dict]] | None = None
        # Índice alternativo SOMENTE para captura atual sem Board/Parts/Value.
        # Contém exclusivamente entradas legadas auditadas por _entry().
        self._visual_index: dict[tuple, list[dict]] | None = None
        self._record_identity: tuple | None = None
        # Auditorias positivas sobrevivem a reloads; cada uso reconfere os
        # metadados dos três arquivos (JSON e dois PNGs) antes de reutilizar.
        self._audited_legacy: dict[tuple, tuple[tuple, tuple]] = {}
        # Rejeições comprovadas também são cacheadas com os stamps das
        # fontes: JSON/PNG novo ou alterado volta a passar por _entry().
        self._rejected_legacy: dict[tuple, tuple[tuple, tuple]] = {}
        self._all_scopes_indexed = False
        self._indexed_scopes: set[tuple] = set()
        self.error = ""
        self._warmup_status = {"state": "not_started", "duration_ms": None}
        self._confirmed_visual_recall = None
        self._knn_c_archive_alignment = None

    def invalidate(self, *, reason: str = "explicit_reload") -> None:
        with self._lock:
            self._index = None
            self._visual_index = None
            self._record_identity = None
            self._all_scopes_indexed = False
            self._indexed_scopes = set()
            self.error = ""
            self._warmup_status = {
                "state": "invalidated", "duration_ms": None,
                "reason": str(reason),
            }

    @staticmethod
    def _key(info: dict, reference_hash: str, test_hash: str) -> tuple | None:
        category = canonical_memory_category(info.get("category", ""))
        board = _clean(info.get("board"))
        parts = _clean(info.get("parts"))
        light = canonical_memory_lighting(info.get("lighting_mode", ""))
        value = _clean(info.get("value", ""))
        if not all((category, board, parts, reference_hash, test_hash)):
            return None
        return category, light, board, parts, value, reference_hash, test_hash

    @staticmethod
    def _visual_key(key: tuple) -> tuple:
        """Categoria + luz + hashes de AMBAS as imagens; nunca similaridade."""
        return key[0], key[1], key[5], key[6]

    @staticmethod
    def _category_only_query(info: dict) -> bool:
        """O OCR atual entrega exclusivamente a categoria (PR #33)."""
        return not any(_clean(info.get(name)) for name in ("board", "parts", "value"))

    @staticmethod
    def _entry(record: dict) -> tuple[tuple, dict] | None:
        if record.get("archive_persistent"):
            # Já consultada no snapshot indexado; não reabrir memory.json
            # como se fosse um arquivo visionx.memory.v3 para cada exemplo.
            return None
        if record.get("mode") != "anomaly":
            return None
        path = Path(str(record.get("json_path", "")))
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "visionx.memory.v3":
            return None
        label = str(data.get("label", "")).strip().upper()
        decision = data.get("decision") or {}
        if not isinstance(decision, dict):
            return None
        operator = str(decision.get("operator_label", "")).strip().upper()
        source = str(decision.get("source", "")).strip().lower()
        # Um OK automático jamais pode se transformar em memória humana
        # capaz de dispensar a CNN/motores no próximo ciclo.
        manual = (
            source in {"button", "xp_keyboard", "keyboard", "manual",
                       "operator", "physical_keyboard", "human"}
            or source.startswith("operator_")
        )
        if not manual:
            return None
        if (label not in ("OK", "NG")
                or operator != label
                or str(record.get("label", "")).strip().upper() != label
                or str(record.get("folder_label", "")).strip().upper() != label):
            return None
        info = data.get("aoi_info", {})
        storage = data.get("storage", {})
        if not isinstance(info, dict) or not isinstance(storage, dict):
            return None
        test_path = _file_within(path.parent, storage.get("test_image_file", ""))
        ref_path = _file_within(path.parent, storage.get("reference_image_file", ""))
        if test_path is None or ref_path is None:
            # JSON-only / dedup visual legado: não assumir mesmo gabarito.
            return None
        reference_hash = _fingerprint_png(ref_path)
        test_hash = _fingerprint_png(test_path)
        if not reference_hash or not test_hash:
            return None
        declared = str(storage.get("test_image_fingerprint", "")).strip()
        if declared and declared != test_hash:
            return None
        key = VerifiedKNNMemory._key(info, reference_hash, test_hash)
        if key is None:
            return None
        if (
            canonical_memory_category(record.get("category")) != key[0]
            or canonical_memory_lighting(record.get("lighting_mode")) != key[1]
            or _clean(record.get("part")) != key[3]
        ):
            return None
        return key, {
            "label": label, "source_json": str(path),
            "_verified_files": (str(path), str(ref_path), str(test_path)),
        }

    @staticmethod
    def _legacy_record_token(record: dict) -> tuple:
        """Identidade estável dos campos que autorizam _entry()."""
        if not isinstance(record, dict):
            return ("invalid", type(record).__name__, repr(record))
        fields = ("json_path", "mode", "category", "lighting_mode",
                  "part", "label", "folder_label", "path")
        return tuple(str(record.get(field, "")) for field in fields)

    @staticmethod
    def _source_stamp(filename: str) -> tuple | None:
        """Metadados físicos; symlinks, ausências e erros invalidam o cache."""
        try:
            path = Path(filename)
            if path.is_symlink():
                return None
            stat = path.stat()
            if not path.is_file():
                return None
            return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                    stat.st_ino, stat.st_mode)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _candidate_source_files(record: dict) -> tuple[str, ...]:
        """Inclui arquivos ausentes para detectar se aparecerem mais tarde."""
        json_path = str(record.get("json_path", "") or "")
        if not json_path:
            return ()
        files = [json_path]
        try:
            data = json.loads(Path(json_path).read_text(encoding="utf-8"))
            storage = data.get("storage", {})
            if isinstance(storage, dict):
                for field in ("reference_image_file", "test_image_file"):
                    name = storage.get(field)
                    if (isinstance(name, str) and name
                            and Path(name).name == name
                            and Path(name).suffix.lower() == ".png"):
                        files.append(str(Path(json_path).parent / name))
        except (OSError, ValueError, TypeError):
            pass
        return tuple(files)

    def _cached_legacy_entry(self, record: dict, profile: dict):
        """Reutiliza auditoria anterior apenas se JSON e PNGs não mudaram.

        Um novo caminho, mudança de rótulo ou de bytes/metadados dispara
        novamente _entry(), que valida proveniência humana e hashes RGB.
        A indexação permanece estritamente exata e fail-closed.
        """
        token = self._legacy_record_token(record)
        previous = self._audited_legacy.get(token)
        if previous is not None:
            entry, stamps = previous
            filenames = entry[1]["_verified_files"]
            current = tuple(self._source_stamp(name) for name in filenames)
            if all(stamp is not None for stamp in current) and current == stamps:
                profile["legacy_cache_hits"] = profile.get("legacy_cache_hits", 0) + 1
                return entry
            self._audited_legacy.pop(token, None)

        negative = self._rejected_legacy.get(token)
        if negative is not None:
            files, stamps = negative
            if tuple(self._source_stamp(name) for name in files) == stamps:
                profile["legacy_rejected_cache_hits"] = (
                    profile.get("legacy_rejected_cache_hits", 0) + 1
                )
                return None
            self._rejected_legacy.pop(token, None)

        profile["legacy_cache_misses"] = profile.get("legacy_cache_misses", 0) + 1
        entry = self._entry(record)
        if entry is not None:
            entry[1]["_verified_token"] = token
            stamps = tuple(
                self._source_stamp(name)
                for name in entry[1]["_verified_files"]
            )
            if all(stamp is not None for stamp in stamps):
                self._audited_legacy[token] = (entry, stamps)
            self._rejected_legacy.pop(token, None)
        else:
            files = self._candidate_source_files(record)
            if files:
                self._rejected_legacy[token] = (
                    files, tuple(self._source_stamp(name) for name in files),
                )
        return entry

    @staticmethod
    def _snapshot_records(knn, profile: dict) -> tuple[list[dict], tuple]:
        """Congela somente o LEGADO que o índice verifica por JSON + PNG.

        O snapshot persistente é sempre consultado separadamente por
        archive_store.lookup(), com validação de integridade. O watcher
        substitui as listas completas a cada importação durável: usar o id()
        das listas invalida inutilmente o índice legado e custa dezenas
        de segundos de I/O sob carga. Identificar por objetos LEGADOS
        preserva o índice até mudança real desses registros ou invalidate().
        """
        started_at = time.perf_counter()
        lock = getattr(knn, "_memory_lock", None)

        def collect():
            all_records = (
                list(getattr(knn, "signatures_ok", ()) or ())
                + list(getattr(knn, "signatures_ng", ()) or ())
            )
            legacy = [
                record for record in all_records
                if not (isinstance(record, dict)
                        and record.get("archive_persistent"))
            ]
            # Listas recriadas pelo reload/refresh não invalidam o legado
            # se os campos que definem os registros continuam equivalentes.
            # reload_memory invalida explicitamente o índice: nesse caso as
            # auditorias positivas só sobrevivem com stamps físicos iguais.
            identity = (
                id(knn),
                tuple(VerifiedKNNMemory._legacy_record_token(r) for r in legacy),
            )
            return all_records, legacy, identity

        if lock is not None:
            with lock:
                all_records, records, identity = collect()
        else:
            all_records, records, identity = collect()
        profile["records_snapshot_ms"] = round(
            (time.perf_counter() - started_at) * 1000.0, 2
        )
        profile["record_count"] = len(all_records)
        profile["legacy_index_record_count"] = len(records)
        profile["archive_persistent_record_count"] = (
            len(all_records) - len(records)
        )
        return records, identity

    def _ensure_index(
        self, records: list[dict], identity: tuple, profile: dict,
    ) -> None:
        """Publica um índice somente após construí-lo integralmente."""
        started_at = time.perf_counter()
        with self._lock:
            reused = (
                self._index is not None
                and identity == self._record_identity
                and self._all_scopes_indexed
            )
            profile["index_reused"] = reused
            if not reused:
                profile["index_rebuild_reason"] = (
                    "not_initialized_or_explicitly_invalidated"
                    if self._index is None else "legacy_records_changed"
                )
                build_started_at = time.perf_counter()
                built = defaultdict(list)
                visual_built = defaultdict(list)
                for record in records:
                    try:
                        entry = self._cached_legacy_entry(record, profile)
                        if entry is not None:
                            built[entry[0]].append(entry[1])
                            visual_built[self._visual_key(entry[0])].append(entry[1])
                    except (OSError, ValueError, TypeError, cv2.error):
                        # Mesma política histórica: registros não auditáveis
                        # não tornam um par conhecido.
                        continue
                self._index = dict(built)
                self._visual_index = dict(visual_built)
                self._prune_audited_legacy(records)
                self._record_identity = identity
                self._all_scopes_indexed = True
                self._indexed_scopes = set()
                profile["index_scope"] = "full_warmup"
                profile["index_build_ms"] = round(
                    (time.perf_counter() - build_started_at) * 1000.0, 2
                )
            else:
                profile["index_build_ms"] = 0.0
        profile["index_lookup_ms"] = round(
            (time.perf_counter() - started_at) * 1000.0, 2
        )

    def _ensure_lookup_scope(
        self, records: list[dict], identity: tuple,
        key: tuple, profile: dict,
    ) -> None:
        """Reconstrói escopo categoria/luz e Parts quando ainda disponível.

        _entry() só autoriza registro com metadados históricos auditados;
        na captura sem Parts, todos os componentes da categoria/luz podem
        concorrer, mas apenas hashes RGB dos DOIS PNGs permitem um match.
        O filtro é estritamente necessário, nunca suficiente: JSON, fonte
        humana e hashes dos dois PNGs ainda passam por _entry().
        O warmup, por sua vez, continua montando o índice integral.
        """
        scope = (key[0], key[1], key[3])
        started_at = time.perf_counter()
        with self._lock:
            if self._record_identity != identity or self._index is None:
                reason = (
                    "not_initialized_or_explicitly_invalidated"
                    if self._index is None else "legacy_records_changed"
                )
                self._index = {}
                self._visual_index = {}
                self._record_identity = identity
                self._all_scopes_indexed = False
                self._indexed_scopes = set()
                profile["index_rebuild_reason"] = reason
            reused = (
                self._all_scopes_indexed or scope in self._indexed_scopes
            )
            profile["index_reused"] = reused
            if reused:
                profile["index_build_ms"] = 0.0
                profile["index_scope"] = (
                    "full_warmup" if self._all_scopes_indexed else "cached_scope"
                )
            else:
                began_build = time.perf_counter()
                eligible = 0
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    if (record.get("mode") != "anomaly"
                            or canonical_memory_category(record.get("category")) != scope[0]
                            or canonical_memory_lighting(record.get("lighting_mode")) != scope[1]
                            or (scope[2] and _clean(record.get("part")) != scope[2])):
                        continue
                    eligible += 1
                    try:
                        entry = self._cached_legacy_entry(record, profile)
                        if entry is not None:
                            self._index.setdefault(entry[0], []).append(entry[1])
                            self._visual_index.setdefault(
                                self._visual_key(entry[0]), []
                            ).append(entry[1])
                    except (OSError, ValueError, TypeError, cv2.error):
                        # Nenhuma falha permite promoção de par não auditado.
                        continue
                self._indexed_scopes.add(scope)
                self._prune_audited_legacy(records)
                profile["index_scope"] = "targeted_legacy_lookup"
                profile["scope_candidate_record_count"] = eligible
                profile["index_build_ms"] = round(
                    (time.perf_counter() - began_build) * 1000.0, 2
                )
        profile["index_lookup_ms"] = round(
            (time.perf_counter() - started_at) * 1000.0, 2
        )

    def _matched_legacy_still_verified(self, detail: dict) -> bool:
        """Nenhum KNOWN legado pode usar PNG/JSON alterado após o warmup."""
        token = detail.get("_verified_token")
        cached = self._audited_legacy.get(token)
        if cached is None or cached[0][1] is not detail:
            return False
        files = detail["_verified_files"]
        current = tuple(self._source_stamp(name) for name in files)
        return all(stamp is not None for stamp in current) and current == cached[1]

    def _prune_audited_legacy(self, records: list[dict]) -> None:
        """Mantém somente entradas presentes no snapshot legado atual."""
        live = {self._legacy_record_token(record) for record in records}
        for cache in (self._audited_legacy, self._rejected_legacy):
            for token in tuple(cache):
                if token not in live:
                    cache.pop(token, None)

    def apply_saved_legacy(self, knn, json_paths) -> None:
        """Aplica deltas no índice já pronto sem revalidar todo o legado.

        A atualização ocorre sob o mesmo lock da consulta e warmup.
        Se outra alteração não relatada for observada, invalida para uma
        reconstrução segura em vez de publicar um índice incompleto.
        """
        from collections import Counter

        changed = {str(Path(p).resolve()) for p in json_paths}
        if not changed:
            return
        with self._lock:
            if self._index is None or self._record_identity is None:
                return
            records, identity = self._snapshot_records(knn, {})
            previous = [
                token for token in self._record_identity[1]
                if str(Path(token[0]).resolve()) not in changed
            ]
            current = [
                token for token in identity[1]
                if str(Path(token[0]).resolve()) not in changed
            ]
            if (self._record_identity[0] != identity[0]
                    or Counter(previous) != Counter(current)):
                self.invalidate(reason="untracked_legacy_change")
                return

            # Retira as versões antigas apenas dos caminhos efetivamente
            # persistidos; índices de demais peças permanecem íntegros.
            for index in (self._index, self._visual_index):
                for key in tuple(index):
                    kept = [row for row in index[key]
                            if str(Path(row["source_json"]).resolve()) not in changed]
                    if kept:
                        index[key] = kept
                    else:
                        del index[key]
            for token in tuple(self._audited_legacy):
                if str(Path(token[0]).resolve()) in changed:
                    self._audited_legacy.pop(token, None)
            for token in tuple(self._rejected_legacy):
                if str(Path(token[0]).resolve()) in changed:
                    self._rejected_legacy.pop(token, None)

            profile = {}
            for record in records:
                if str(Path(record.get("json_path", "")).resolve()) not in changed:
                    continue
                entry = self._cached_legacy_entry(record, profile)
                if entry is None:
                    continue
                key, detail = entry
                scope = (key[0], key[1], key[3])
                if (self._all_scopes_indexed
                        or scope in self._indexed_scopes
                        or (key[0], key[1], "") in self._indexed_scopes):
                    self._index.setdefault(key, []).append(detail)
                    self._visual_index.setdefault(
                        self._visual_key(key), []
                    ).append(detail)
            self._record_identity = identity
            self._prune_audited_legacy(records)

    def warmup(self, knn) -> dict:
        """Constrói o índice no worker, sem aprovar ou classificar imagens."""
        started_at = time.perf_counter()
        timing: dict = {"schema": "visionx.verified_knn_warmup.v1"}
        with self._lock:
            self._warmup_status = {"state": "running", "duration_ms": None}
        try:
            if knn is None:
                raise ValueError("Especialista KNN indisponível")
            records, identity = self._snapshot_records(knn, timing)
            self._ensure_index(records, identity, timing)
            # Se a memória foi alterada entre o snapshot e a publicação,
            # o próximo lookup detectará a identidade diferente e reconstruirá.
            with self._lock:
                # Um reload que terminou entre a construção e o relato
                # invalida a preparação; nunca anunciar índice antigo pronto.
                timing["state"] = (
                    "ready" if self._index is not None
                    and self._record_identity == identity else "invalidated"
                )
                timing["duration_ms"] = round(
                    (time.perf_counter() - started_at) * 1000.0, 2
                )
                timing["record_count"] = timing.get(
                    "record_count", len(records)
                )
                self._warmup_status = dict(timing)
        except Exception as exc:
            with self._lock:
                timing["state"] = "failed"
                timing["error"] = type(exc).__name__
                timing["duration_ms"] = round(
                    (time.perf_counter() - started_at) * 1000.0, 2
                )
                self._warmup_status = dict(timing)
        return dict(timing)

    def recall_confirmed_visual(self, reference, test, info: dict) -> dict:
        """Recupera hipótese visual dos julgamentos HUMANOS persistentes.

        Não substitui a busca exata, não autoriza 0/OK nem aprende
        rótulos automaticamente. O snapshot é reutilizado por geração.
        """
        from src.core.confirmed_visual_recall import ConfirmedVisualRecall
        with self._lock:
            if self._confirmed_visual_recall is None:
                self._confirmed_visual_recall = ConfirmedVisualRecall(
                    self.archive_store
                )
            recall = self._confirmed_visual_recall
        return recall.lookup(reference, test, info)

    def recall_knn_c_archive_alignment(
        self, reference, test, info: dict, human: dict,
    ) -> dict:
        """Relação humano/PNG na MESMA assinatura; A/B originais intocados."""
        from src.core.knn_c_archive_alignment import KNNArchiveAlignment
        with self._lock:
            if self._knn_c_archive_alignment is None:
                self._knn_c_archive_alignment = KNNArchiveAlignment(self.archive_store)
            engine = self._knn_c_archive_alignment
        return engine.diagnose(reference, test, info, human)

    def lookup(
        self, knn, reference, test, info: dict,
        *, timing: dict | None = None,
    ) -> dict:
        """Consulta exata; publica tempos apenas no perfil da chamada."""
        started_at = time.perf_counter()
        profile = {"schema": "visionx.verified_knn_lookup_timing.v1"}
        with self._lock:
            profile["warmup"] = dict(self._warmup_status)
        try:
            return self._lookup_profiled(knn, reference, test, info, profile)
        finally:
            profile["total_ms"] = round(
                (time.perf_counter() - started_at) * 1000.0, 2
            )
            if isinstance(timing, dict):
                timing.update(profile)

    def _lookup_profiled(
        self, knn, reference, test, info: dict, profile: dict,
        *, retry_if_changed: bool = True,
    ) -> dict:
        """Mesmo algoritmo de busca, com medições monotônicas passivas."""
        if not _image_ok(reference) or not _image_ok(test):
            return {"status": "UNAVAILABLE", "reason": "Par AOI inválido"}
        # A correção humana CRONOLÓGICA do mesmo par RGB prevalece sobre
        # registros antigos, inclusive NG de arquivo e legado. Não libera
        # decisões por similaridade: exige categoria/luz e hashes dos dois
        # recortes exatamente iguais. O histórico permanece auditável.
        recent_lookup = getattr(
            self.archive_store, "latest_confirmed_human_for_pair", None
        )
        if callable(recent_lookup):
            recent = recent_lookup(reference, test, info)
            if recent.get("status") == "KNOWN":
                profile["latest_operator_override"] = True
                profile["latest_operator_record_id"] = recent.get("record_id")
                return recent
            if recent.get("status") == "CONFLICT":
                return recent

        # Consulta independente dos PNGs (o snapshot continua após exclusão).
        # Erro de integridade é elevado e bloqueia a aprovação por segurança.
        archive_started_at = time.perf_counter()
        try:
            durable = self.archive_store.lookup(reference, test, info)
        finally:
            profile["archive_lookup_ms"] = round(
                (time.perf_counter() - archive_started_at) * 1000.0, 2
            )
        if durable["status"] in {"CONFLICT", "UNAVAILABLE"}:
            return durable
        # A estação pode receber uma imagem TOP marcada SIDE. A segunda
        # consulta só reconhece o MESMO par RGB exato em outra iluminação,
        # incluindo todas as classes para bloquear rótulos contraditórios.
        cross_lookup = getattr(self.archive_store, "lookup_exact_any_light", None)
        if callable(cross_lookup):
            cross = cross_lookup(reference, test, info)
            if cross["status"] in {"CONFLICT", "UNAVAILABLE"}:
                return cross
            if cross["status"] == "KNOWN":
                if durable["status"] == "KNOWN" and durable["label"] != cross["label"]:
                    return {
                        "status": "CONFLICT",
                        "reason": "Memória persistente contraditória entre iluminações",
                    }
                if durable["status"] != "KNOWN":
                    durable = cross
                profile["cross_light_exact_matches"] = cross["matches"]
                profile["cross_light_exact_source"] = cross["source_lighting_mode"]
        hashes_started_at = time.perf_counter()
        try:
            reference_hash = image_fingerprint(reference)
            test_hash = image_fingerprint(test)
            key = self._key(info, reference_hash, test_hash)
            category_only = self._category_only_query(info)
            if key is None and category_only:
                category = canonical_memory_category(info.get("category", ""))
                light = canonical_memory_lighting(info.get("lighting_mode", ""))
                if (category and category not in {"UNKNOWN", "UNKNOW", "DESCONHECIDO"}
                        and light in {"SIDE", "TOP", "MID"}):
                    key = (category, light, "", "", "", reference_hash, test_hash)
        finally:
            profile["fingerprints_ms"] = round(
                (time.perf_counter() - hashes_started_at) * 1000.0, 2
            )
        profile["legacy_lookup_mode"] = (
            "exact_visual_category_light" if category_only and key is not None
            else "legacy_metadata" if key is not None else "unavailable"
        )
        if key is None:
            # Uma chave incompleta nunca pode pular a auditoria de rótulos
            # humanos de todas as luzes só por existir um registro TOP.
            if durable.get("cross_light_verified"):
                return {
                    "status": "UNAVAILABLE",
                    "reason": "Metadados incompletos impedem auditoria cross-light",
                }
            return durable if durable["status"] == "KNOWN" else {
                "status": "NEW", "reason": "Categoria ou par AOI inválido"
            }
        if knn is None:
            if durable.get("cross_light_verified"):
                return {
                    "status": "UNAVAILABLE",
                    "reason": "Reconhecimento entre iluminações exige auditoria KNN completa",
                }
            return durable if durable["status"] == "KNOWN" else {
                "status": "NEW", "reason": "Nenhuma memória KNN carregada"
            }

        with self._lock:
            # Snapshot e índice da MESMA geração. Sem isto, a thread
            # de persistência pode publicar atualização entre as duas etapas.
            records, identity = self._snapshot_records(knn, profile)
            self._ensure_lookup_scope(records, identity, key, profile)
            matched = list(
                self._visual_index.get(self._visual_key(key), ())
                if category_only else self._index.get(key, ())
            )
            profile["warmup"] = dict(self._warmup_status)
            cross_legacy = []
            if durable.get("cross_light_verified"):
                # Nunca aprovar o arquivo TOP se outro registro humano
                # legado classificar o MESMO par RGB como NG em SIDE/MID.
                # Somente o índice integral auditado cobre as três luzes.
                if not self._all_scopes_indexed:
                    return {
                        "status": "UNAVAILABLE",
                        "reason": "Índice humano multilight ainda não está auditado",
                    }
                cross_legacy = [
                    item
                    for light in ("SIDE", "TOP", "MID")
                    for item in self._visual_index.get(
                        (key[0], light, key[5], key[6]), ()
                    )
                ]
                profile["cross_light_legacy_checked"] = True
            stale_match = any(
                not self._matched_legacy_still_verified(item)
                for item in (*matched, *cross_legacy)
            )
            if stale_match:
                self.invalidate(reason="legacy_source_changed_during_session")
        if stale_match:
            if retry_if_changed:
                return self._lookup_profiled(
                    knn, reference, test, info, profile,
                    retry_if_changed=False,
                )
            return {
                "status": "UNAVAILABLE",
                "reason": "Origem KNN mudou durante a verificação exata",
            }
        if (durable.get("cross_light_verified")
                and any(row["label"] != durable["label"] for row in cross_legacy)):
            return {
                "status": "CONFLICT",
                "reason": "Memória humana contradiz o arquivo entre iluminações",
            }
        profile["matched_count"] = len(matched)
        if not matched:
            return durable if durable["status"] == "KNOWN" else {
                "status": "NEW",
                "reason": "Par exato não encontrado na memória humana nem no arquivo KNN",
            }
        labels = {item["label"] for item in matched}
        if durable["status"] == "KNOWN":
            labels.add(durable["label"])
        if len(labels) != 1:
            return {
                "status": "CONFLICT",
                "reason": "Memória humana e/ou arquivo histórico divergem em OK/NG",
                "matches": len(matched) + (durable.get("matches", 0)
                                            if durable["status"] == "KNOWN" else 0),
            }
        return {
            "status": "KNOWN", "label": labels.pop(),
            "source_json": matched[0]["source_json"],
            "matches": len(matched) + (durable.get("matches", 0)
                                       if durable["status"] == "KNOWN" else 0),
            "reason": (
                "Par RGB exato auditado por categoria e iluminação, sem OCR legado"
                if category_only else "Par visual exato com rótulo consistente"
            ),
            "memory_source": "HUMAN_VERIFIED_PAIR",
        }


def _recognized_result(match: dict, mode: str) -> dict:
    label = match["label"]
    ng = label == "NG"
    score = 1.0 if ng else 0.0
    cross_light = match.get("cross_light_verified") is True
    source_light = str(match.get("source_lighting_mode", "") or "")
    latest_operator = match.get("memory_source") == "LATEST_OPERATOR_EXACT_PAIR"
    trace = {
        "schema": RECOGNITION_SCHEMA,
        "dominant_engine": "knn_verified_exact",
        "fusion_rule": "knn_verified_pair_exact",
        "final_score": score,
        "physical_score": 0.0,
        "cutoff": .5,
        "operator_review_required": False,
        "weights": {"physical": 0.0, "knn": 1.0, "cnn": 0.0},
        "recognition_route": "KNOWN_KNN",
        "latest_operator_override": latest_operator,
        "memory": {
            "has_memory": True, "best_similarity": 1.0,
            "best_match_label": label, "best_ok_similarity": 0.0 if ng else 1.0,
            "best_ng_similarity": 1.0 if ng else 0.0,
            "role": "Correspondência exata verificada, sem votação aproximada",
        },
    }
    return {
        "is_defect": ng, "confidence": .99,
        "verdict": "DEFEITO REAL" if ng else "FALHA FALSA",
        "reason": (
            f"MEMÓRIA KNN: último julgamento humano {label} prevaleceu. "
            if latest_operator else
            f"MEMÓRIA KNN: caso já analisado e confirmado como {label}. "
        ) + (
            + (f"Gabarito/teste RGB e categoria idênticos; origem {source_light}, "
               f"evento {mode}. Luz do XP divergente, sem usar similaridade."
               if cross_light else
               "Gabarito/teste, categoria e iluminação idênticos.")
        ),
        "production_review_required": False,
        "active_engines": ["knn_expert.py"],
        "bounding_box": None, "all_boxes": {},
        "lighting_mode": mode,
        "detail": {
            "recognition_route": "KNOWN_KNN",
            "recognition_schema": RECOGNITION_SCHEMA,
            "recognition_match": "EXACT_PAIR",
            "recognition_known_label": label,
            "recognition_memory_path": match["source_json"],
            "recognition_latest_operator_override": latest_operator,
            "recognition_latest_operator_record_id": match.get("record_id", ""),
            "recognition_latest_operator_event_id": match.get("source_event_id", ""),
            "recognition_latest_operator_confirmed_at_ns": match.get("confirmed_at_ns"),
            "recognition_superseded_labels": match.get("superseded_labels", []),
            "recognition_memory_matches": match["matches"],
            "recognition_memory_verified": True,
            "recognition_memory_source": match.get("memory_source", "HUMAN_VERIFIED_PAIR"),
            "recognition_memory_cross_light_exact": cross_light,
            "recognition_memory_source_lighting": source_light,
            "recognition_memory_source_png": match.get("source_png", ""),
            "recognition_specialists_skipped": True,
            "has_memory": True, "memory_available": True,
            "best_similarity": 1.0, "best_match_label": label,
            "best_ok_similarity": 0.0 if ng else 1.0,
            "best_ng_similarity": 1.0 if ng else 0.0,
            "memory_scope": (
                "categoria_par_rgb_exato_entre_iluminacoes" if cross_light
                else "categoria_e_luz_exata"
            ),
            "memory_mode": "verified_pair",
            "vote_defect": score,
            "final_score": score, "physical_score": 0.0,
            "fusion_rule": "knn_verified_pair_exact",
            "dominant_engine": "knn_verified_exact",
            "operator_review_required": False,
            "decision_trace": trace,
        },
    }


def _review_result(reason: str, mode: str) -> dict:
    return {
        "is_defect": False, "confidence": .5,
        "verdict": "REVISÃO OBRIGATÓRIA",
        "reason": "MEMÓRIA KNN: " + reason,
        "production_review_required": True,
        "active_engines": ["knn_expert.py"],
        "bounding_box": None, "all_boxes": {},
        "lighting_mode": mode,
        "detail": {
            "recognition_route": "MEMORY_CONFLICT",
            "recognition_schema": RECOGNITION_SCHEMA,
            "recognition_specialists_skipped": True,
            "final_score": .5, "physical_score": 0.0,
            "dominant_engine": "knn_verified_exact",
            "fusion_rule": "knn_label_conflict_review",
            "operator_review_required": True,
            "decision_trace": {
                "operator_review_required": True,
                "fusion_rule": "knn_label_conflict_review",
                "final_score": .5,
                "physical_score": 0.0,
            },
        },
    }


def install_memory_first_router(orchestrator_cls, *, memory=None) -> None:
    """Último wrapper: KNOWN usa rótulo humano KNN; NEW especialista sem KNN."""
    if getattr(orchestrator_cls, "_memory_first_router_installed", False):
        return
    original = orchestrator_cls.inspect
    original_reload = orchestrator_cls.reload_memory

    def inspect(
        self, full_gab, full_test, raw_anomalies,
        aoi_info, global_box_info, aoi_epicenters,
    ):
        info = aoi_info if isinstance(aoi_info, dict) else {}
        if (
            info.get("_replay_without_memory", False)
            or type(self).__name__ == "PhysicalOnlyOrchestrator"
        ):
            return original(
                self, full_gab, full_test, raw_anomalies,
                aoi_info, global_box_info, aoi_epicenters
            )

        index = self._get_verified_memory_index()

        # Medir a PRIMEIRA rota da memória, antes da CNN.
        # Essa etapa não é coberta pelo perfil interno FaltandoCNNLive.
        memory_timing: dict = {}
        lookup_started_at = time.perf_counter()
        try:
            if type(index) is VerifiedKNNMemory:
                match = index.lookup(
                    getattr(self, "experts", {}).get("knn"),
                    full_gab, full_test, info, timing=memory_timing,
                )
            else:
                match = index.lookup(
                    getattr(self, "experts", {}).get("knn"),
                    full_gab, full_test, info,
                )
        except Exception as exc:
            # Índice indisponível: nunca permitir aprovação silenciosa.
            match = {"status": "UNAVAILABLE", "reason": str(exc)}
        memory_timing.setdefault(
            "total_ms",
            round((time.perf_counter() - lookup_started_at) * 1000.0, 2),
        )
        memory_timing["status"] = str(match.get("status", "UNAVAILABLE"))

        def with_memory_timing(outcome):
            if isinstance(outcome, dict):
                detail = outcome.setdefault("detail", {})
                if isinstance(detail, dict):
                    detail["verified_memory_timing"] = dict(memory_timing)
            return outcome

        mode = canonical_memory_lighting(info.get("lighting_mode", "SIDE"))
        if match["status"] == "UNAVAILABLE":
            return with_memory_timing(_review_result(
                "Verificação exata indisponível: " + str(match.get("reason", "")),
                mode,
            ))
        if match["status"] == "KNOWN":
            return with_memory_timing(_recognized_result(match, mode))
        if match["status"] == "CONFLICT":
            return with_memory_timing(_review_result(match["reason"], mode))

        category = canonical_memory_category(info.get("category", ""))
        if uses_faltando_v2(category):
            route = "NEW_CNN"
            # O FaltandoCNNLive é o wrapper anterior, não consulta KNN.
            requested_info = dict(info)
        else:
            route = "NEW_EXPERTS"
            requested_info = dict(info)
            # A extensão física aceita este marcador: dispensa KNN.
            # Em replay offline original o flag é preservado.
            requested_info["_replay_without_memory"] = True
        result = original(
            self, full_gab, full_test, raw_anomalies,
            requested_info, global_box_info, aoi_epicenters,
        )
        if not isinstance(result, dict):
            return with_memory_timing(_review_result(
                "O especialista não retornou um resultado", mode
            ))
        detail = result.setdefault("detail", {})
        if not isinstance(detail, dict):
            detail = {}
            result["detail"] = detail
        if type(index) is VerifiedKNNMemory:
            # Persistência humana e KNN C de PNGs são bancos distintos:
            # recuperar agora também a experiência humana APROXIMADA,
            # antes do gate SIDE, sem transformar similaridade em KNOWN_KNN.
            human_recall = index.recall_confirmed_visual(
                full_gab, full_test, info,
            )
            detail["confirmed_visual_recall"] = human_recall
        detail.update({
            "specialist_candidate": (
                None
            ),
            "recognition_route": route,
            "cnn_v2_aoi_category": category if route == "NEW_CNN" else None,
            "recognition_schema": RECOGNITION_SCHEMA,
            "recognition_match": "NOT_FOUND",
            "recognition_memory_verified": False,
            "recognition_specialists_skipped": False,
            "recognition_reason": str(match.get("reason", "")),
        })
        trace = detail.get("decision_trace")
        if isinstance(trace, dict):
            trace["recognition_route"] = route
        if route == "NEW_CNN":
            result["reason"] = "CASO NOVO • CNN FALTANDO v2. " + str(result.get("reason", ""))
        else:
            result["reason"] = "CASO NOVO • motores da categoria. " + str(result.get("reason", ""))
        return with_memory_timing(result)

    def _get_verified_memory_index(self):
        existing = getattr(self, "_memory_first_index", None)
        if existing is None:
            existing = memory if memory is not None else VerifiedKNNMemory()
            self._memory_first_index = existing
        return existing

    def prepare_verified_memory_index(self):
        index = self._get_verified_memory_index()
        if type(index) is VerifiedKNNMemory:
            return index.warmup(getattr(self, "experts", {}).get("knn"))
        return {"state": "unsupported"}

    def ingest_saved_memory(self, json_paths):
        """Atualização incremental dos JSONs criados pelo operador."""
        knn = getattr(self, "experts", {}).get("knn")
        if knn is None or not hasattr(knn, "ingest_saved_samples"):
            return self.reload_memory()
        paths = tuple(str(p) for p in json_paths if p)
        cached = getattr(self, "_memory_first_index", None)
        if cached is not None and type(cached) is VerifiedKNNMemory:
            # Sem intervalo entre atualizar as listas KNN e o índice:
            # SIDE/TOP/MID sempre consultam uma geração completa.
            with cached._lock:
                knn.ingest_saved_samples(paths)
                cached.apply_saved_legacy(knn, paths)
        else:
            knn.ingest_saved_samples(paths)

    def reload_memory(self, *args, **kwargs):
        outcome = original_reload(self, *args, **kwargs)
        cached = getattr(self, "_memory_first_index", None)
        if cached is not None:
            cached.invalidate(reason="orchestrator_reload_memory")
        return outcome

    orchestrator_cls.inspect = inspect
    orchestrator_cls._get_verified_memory_index = _get_verified_memory_index
    orchestrator_cls.prepare_verified_memory_index = prepare_verified_memory_index
    orchestrator_cls.reload_memory = reload_memory
    orchestrator_cls.ingest_saved_memory = ingest_saved_memory
    orchestrator_cls._memory_first_router_installed = True


__all__ = ["VerifiedKNNMemory", "install_memory_first_router", "RECOGNITION_SCHEMA"]
