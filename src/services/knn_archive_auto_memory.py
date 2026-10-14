"""Sincronização incremental e automática da memória KNN dos arquivos OK/NG.

Independe de Modo Teste/Produção/Sombra. Observa DIRETÓRIOS, não eventos
de botão: tudo que entrar em public/ok_archive ou public/ng_archive passa por
estabilidade do arquivo, validação PNG e extração AOI em thread independente.
O snapshot append-only sobrevive ao sumiço das imagens originais.

O worker nunca aperta 0/1, nunca altera os PNGs e nunca confia no
veredito atual da IA para produzir o rótulo: o rótulo vem da pasta física.
Isso não torna autogerados verdade independente. Casos inválidos continuam
pendentes com retentativa e mensagem explícita, sem promover a OK.
"""
from __future__ import annotations

from hashlib import sha256
import logging
from pathlib import Path
from threading import Event, Lock, Thread
import time
from typing import Callable

from src.config.settings import BASE_DIR
from src.core.anomaly_signature import build_anomaly_signature
from src.core.archive_persistent_memory import PersistentArchiveKNN, make_record
from src.core.strict_category_memory import canonical_memory_category
from src.services.faltando_cross_category_audit import _archive_image
from src.services.startup_regression.archive_inventory import (
    _inspect_png, filename_hints,
)
from src.utils.text_normalizer import normalize_aoi_text

LOG = logging.getLogger(__name__)
LABEL_DIRS = (("OK", "ok_archive"), ("NG", "ng_archive"))
_active_lock = Lock()
_active_monitor = None


def _extract_entry(root: Path, path: Path, label: str, *,
                   extractor, frame_loader) -> dict:
    """Valida PNG, bytes e categoria; retorna um registro durável íntegro."""
    relative = path.relative_to(root).as_posix()
    item = _inspect_png(path, root, label)
    if item["status"] != "VALID_PNG":
        reason = "; ".join(i["detail"] for i in item.get("issues", ()))
        raise ValueError(f"PNG ainda incompleto ou inválido: {reason}")
    frame = frame_loader(root, item)  # relê SHA; detecta alteração durante importação
    reference, test, info = extractor(frame)
    if not isinstance(info, dict):
        raise ValueError("AOI não extraiu metadados")
    category, normalized_value = normalize_aoi_text(info.get("value", ""))
    cat = canonical_memory_category(category)
    expected = canonical_memory_category(filename_hints(path)["category_hint"])
    if not cat or cat in {"UNKNOWN", "UNKNOW"} or cat != expected:
        raise ValueError(
            f"Categoria AOI {cat!r} não confere com nome {expected!r}"
        )
    signature = build_anomaly_signature(
        reference, test, {}, {**info, "category": cat,
                              "value": normalized_value}
    )
    return make_record(
        label=label, source_path=relative,
        source_sha256=item["file_sha256"],
        category=cat, lighting=item["lighting_mode"],
        reference=reference, test=test,
        anomaly_signature=signature, aoi_info=info,
    )


class KNNArchiveAutoMemory:
    """Observador de pasta seguro, sem importar no caminho crítico da AOI."""

    def __init__(
        self, root: Path | None = None, *, store=None,
        extractor: Callable | None = None,
        frame_loader: Callable | None = None,
        poll_interval: float = 1.0,
        settle_seconds: float = 0.5,
        batch_size: int = 24,
        on_committed: Callable | None = None,
    ):
        self.root = Path(root or BASE_DIR).expanduser().resolve()
        self.store = store if store is not None else PersistentArchiveKNN(self.root)
        self._extractor = extractor
        self._frame_loader = frame_loader or _archive_image
        self.poll_interval = max(.1, float(poll_interval))
        self.settle_seconds = max(0.0, float(settle_seconds))
        self.batch_size = max(1, int(batch_size))
        self.on_committed = on_committed
        self._wake = Event()
        self._stop = Event()
        self._thread: Thread | None = None
        self._observed: dict[str, tuple[tuple[int, int], float]] = {}
        self._completed: set[tuple[str, tuple[int, int]]] = set()
        self._attempts: dict[str, tuple[int, float, tuple[int, int]]] = {}
        self._errors: dict[str, str] = {}
        self._refresh_pending = False
        self._lock = Lock()
        self._total_added = 0
        self._last_pass = None
        # Telemetria passiva: mede tempo de parede do observador em scan
        # (não é tempo de CPU nem autoriza alterar inspeção ou memória).
        self._worker_scan_active = False
        self._worker_scan_started = 0.0
        self._worker_busy_total_s = 0.0
        self._worker_scan_count = 0
        self._worker_last_scan_ms = None
        self._worker_phase = "idle"

    def _set_worker_phase(self, phase: str) -> None:
        with self._lock:
            if self._worker_scan_active:
                self._worker_phase = str(phase)

    def activity(self) -> dict:
        """Snapshot sem caminhos, PNGs, identificação AOI ou dados sensíveis."""
        with self._lock:
            now = time.monotonic()
            active = self._worker_scan_active
            total = self._worker_busy_total_s + (
                max(0.0, now - self._worker_scan_started)
                if active else 0.0
            )
            return {
                "running": bool(self._thread is not None
                                and self._thread.is_alive()),
                "scan_active": bool(active),
                "phase": self._worker_phase,
                "busy_total_ms": round(total * 1000.0, 2),
                "scan_count": self._worker_scan_count,
                "last_scan_ms": self._worker_last_scan_ms,
                "imports_this_session": self._total_added,
            }

    def _ensure_extractor(self):
        if self._extractor is None:
            from src.services.faltando_neural_dataset import AOIPairExtractor
            self._extractor = AOIPairExtractor()
        return self._extractor

    def _list_files(self) -> list[tuple[Path, str]]:
        paths = []
        for label, dirname in LABEL_DIRS:
            archive = self.root / "public" / dirname
            if not archive.is_dir():
                LOG.error("Arquivo KNN ausente: %s", archive)
                continue
            for path in archive.rglob("*"):
                if not path.is_file() or path.suffix.lower() != ".png":
                    continue
                if path.is_symlink():
                    LOG.warning("KNN ignora link simbólico: %s", path)
                    continue
                try:
                    if archive.resolve() not in path.resolve().parents:
                        continue
                except OSError:
                    continue
                paths.append((path, label))
        return sorted(paths, key=lambda p: str(p[0]).casefold())

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._thread is not None and self._thread.is_alive(),
                "imports_this_session": self._total_added,
                "pending_or_failed": dict(self._errors),
                "last_scan": self._last_pass,
            }

    def scan_once(self, *, now: float | None = None) -> dict:
        """Uma passagem de verificação (também usada pelos testes offline)."""
        instant = time.monotonic() if now is None else float(now)
        try:
            # Checksum inválido não é tratado como snapshot vazio.
            current = self.store.records()
        except (OSError, ValueError, TypeError) as exc:
            LOG.error("KNN não pode sincronizar: snapshot inválido: %s", exc)
            with self._lock:
                self._errors["SNAPSHOT"] = str(exc)
            return {"imported": 0, "pending": 0, "errors": 1, "fatal": True}

        imported_sources = {
            (src["source_path"], src["source_sha256"])
            for r in current for src in r["sources"]
        }
        self._set_worker_phase("enumerating_files")
        candidates = []
        live = set()
        pending = 0
        errors = 0
        for path, label in self._list_files():
            relative = path.relative_to(self.root).as_posix()
            live.add(relative)
            try:
                stat = path.stat()
                stamp = (stat.st_size, stat.st_mtime_ns)
            except OSError as exc:
                with self._lock:
                    self._errors[relative] = str(exc)
                errors += 1
                continue
            previous = self._observed.get(relative)
            if previous is None or previous[0] != stamp:
                self._observed[relative] = (stamp, instant)
                self._attempts.pop(relative, None)
                self._completed.discard((relative, stamp))
                with self._lock:
                    self._errors.pop(relative, None)
                pending += 1
                continue
            if (instant - previous[1]) < self.settle_seconds:
                pending += 1
                continue
            if (relative, stamp) in self._completed:
                continue
            attempts, due, previous_stamp = self._attempts.get(
                relative, (0, 0.0, stamp)
            )
            if previous_stamp == stamp and instant < due:
                pending += 1
                continue
            if len(candidates) < self.batch_size:
                candidates.append((path, label, relative, stamp))
            else:
                pending += 1

        proposed = []
        accepted = []
        for path, label, relative, stamp in candidates:
            try:
                self._set_worker_phase("hashing_png")
                # Fast path de reinicialização: sem OCR dos PNGs já importados.
                raw_sha = sha256(path.read_bytes()).hexdigest()
                if (relative, raw_sha) in imported_sources:
                    self._completed.add((relative, stamp))
                    self._attempts.pop(relative, None)
                    with self._lock:
                        self._errors.pop(relative, None)
                    continue
                self._set_worker_phase("extracting_png")
                entry = _extract_entry(
                    self.root, path, label,
                    extractor=self._ensure_extractor(),
                    frame_loader=self._frame_loader,
                )
                if (relative, entry["sources"][0]["source_sha256"]) in imported_sources:
                    self._completed.add((relative, stamp))
                    continue
                proposed.append(entry)
                accepted.append((relative, stamp))
            except Exception as exc:
                attempts, _, _ = self._attempts.get(relative, (0, 0, stamp))
                tries = attempts + 1
                delay = min(60.0, float(2 ** min(tries, 6)))
                self._attempts[relative] = (tries, instant + delay, stamp)
                with self._lock:
                    self._errors[relative] = f"{type(exc).__name__}: {exc}"
                LOG.warning(
                    "KNN automática: pendente %s (%s); nova tentativa em %.0fs",
                    relative, exc, delay,
                )
                errors += 1

        newly_added = 0
        if proposed:
            try:
                self._set_worker_phase("persisting_memory")
                result = self.store.add(proposed)
                newly_added = result["new"]
                for relative, stamp in accepted:
                    self._completed.add((relative, stamp))
                    self._attempts.pop(relative, None)
                    with self._lock:
                        self._errors.pop(relative, None)
                if result["changed"] and self.on_committed is not None:
                    self._refresh_pending = True
                LOG.info(
                    "KNN memória permanente: %d novas, total %d, OK %d / NG %d",
                    result["new"], result["total"],
                    result["by_label"]["OK"], result["by_label"]["NG"],
                )
            except Exception as exc:
                with self._lock:
                    self._errors["SNAPSHOT_WRITE"] = str(exc)
                LOG.exception("KNN falhou na gravação atômica da memória: %s", exc)
                errors += len(proposed)
        # Memória persistida já é consultável; se o refresh de RAM falhar,
        # tentar de novo nas próximas varreduras sem reaprender PNGs.
        if self._refresh_pending and self.on_committed is not None:
            try:
                self._set_worker_phase("refreshing_knn")
                self.on_committed({"path": str(self.store.path)})
                self._refresh_pending = False
                with self._lock:
                    self._errors.pop("HOT_REFRESH", None)
            except Exception as exc:
                with self._lock:
                    self._errors["HOT_REFRESH"] = f"{type(exc).__name__}: {exc}"
                LOG.warning("KNN snapshot salvo; refresh de RAM pendente: %s", exc)
        self._observed = {
            p: observation for p, observation in self._observed.items()
            if p in live
        }
        self._completed = {(p, stamp) for p, stamp in self._completed
                           if p in live}
        self._attempts = {p: attempt for p, attempt in self._attempts.items()
                          if p in live}
        with self._lock:
            self._total_added += newly_added
            self._last_pass = instant
        return {
            "imported": newly_added,
            "pending": pending + max(0, len(candidates) - len(accepted)),
            "errors": errors,
            "fatal": False,
        }

    def _run(self):
        LOG.info("KNN sincronização automática de arquivos OK/NG iniciada")
        while not self._stop.is_set():
            started = time.monotonic()
            with self._lock:
                self._worker_scan_active = True
                self._worker_scan_started = started
                self._worker_phase = "reading_snapshot"
            try:
                self.scan_once()
            except Exception:
                LOG.exception("KNN observador: erro; retentativa na próxima passagem")
            finally:
                finished = time.monotonic()
                with self._lock:
                    elapsed = max(0.0, finished - started)
                    self._worker_busy_total_s += elapsed
                    self._worker_last_scan_ms = round(elapsed * 1000.0, 2)
                    self._worker_scan_count += 1
                    self._worker_scan_active = False
                    self._worker_scan_started = 0.0
                    self._worker_phase = "idle"
            self._wake.wait(self.poll_interval)
            self._wake.clear()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = Thread(target=self._run, name="ODINKNNArchiveMemory",
                              daemon=True)
        self._thread.start()
        return self

    def notify(self, _path: Path | None = None):
        """Somente acorda worker; não roda OCR no thread de arquivamento."""
        self._wake.set()

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        self._wake.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)


def activate_archive_auto_memory(watcher: KNNArchiveAutoMemory | None):
    global _active_monitor
    with _active_lock:
        _active_monitor = watcher


def notify_archive_image_saved(path: Path):
    with _active_lock:
        worker = _active_monitor
    if worker is not None:
        worker.notify(path)


def archive_auto_memory_activity() -> dict | None:
    """Apenas status da thread KNN; não consulta arquivos ou imagens."""
    with _active_lock:
        worker = _active_monitor
    return worker.activity() if worker is not None else None


__all__ = [
    "KNNArchiveAutoMemory", "activate_archive_auto_memory",
    "notify_archive_image_saved", "archive_auto_memory_activity",
]
