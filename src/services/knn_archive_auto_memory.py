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
import os
import sys
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


def _set_background_thread_mode(enable: bool) -> bool:
    """Prioriza captura/UI ao reduzir CPU/I/O/memória SOMENTE do watcher.

    THREAD_MODE_BACKGROUND_BEGIN/END (Win32) só pode ser aplicado na
    própria thread. O modo de prioridade do processo ODIN nunca muda.
    Não requer privilégios administrativos. Falha ou outro SO: modo normal.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        current_thread = kernel32.GetCurrentThread
        current_thread.restype = ctypes.c_void_p
        set_priority = kernel32.SetThreadPriority
        set_priority.argtypes = (ctypes.c_void_p, ctypes.c_int)
        set_priority.restype = ctypes.c_int
        mode = 0x00010000 if enable else 0x00020000
        succeeded = bool(set_priority(current_thread(), mode))
        if not succeeded:
            LOG.warning(
                "Prioridade KNN de background nao aplicada (enable=%s; winerr=%s)",
                enable, ctypes.get_last_error(),
            )
        return succeeded
    except (OSError, AttributeError, TypeError, ValueError) as exc:
        LOG.warning("KNN mantem prioridade normal: %s", exc)
        return False


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
        post_inspection_quiet_seconds: float = 3.0,
        max_idle_deferral_seconds: float = 30.0,
    ):
        self.root = Path(root or BASE_DIR).expanduser().resolve()
        self.store = store if store is not None else PersistentArchiveKNN(self.root)
        self._extractor = extractor
        self._frame_loader = frame_loader or _archive_image
        self.poll_interval = max(.1, float(poll_interval))
        self.settle_seconds = max(0.0, float(settle_seconds))
        self.batch_size = max(1, int(batch_size))
        self.on_committed = on_committed
        # Ao liberar o gate, a próxima SIDE normalmente chega em ~2 s.
        # Aguardar brevemente evita iniciar Tesseract de acervo justo antes
        # da próxima peça. Limite anti-starvation garante progresso na folga.
        self.post_inspection_quiet_seconds = max(
            0.0, float(post_inspection_quiet_seconds)
        )
        self.max_idle_deferral_seconds = max(
            self.post_inspection_quiet_seconds,
            float(max_idle_deferral_seconds),
        )
        self._idle_grace_until = 0.0
        self._idle_grace_deferrals = 0
        self._last_scan_started_at = time.monotonic()
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
        self._worker_background_mode = False
        # A inspeção SIDE/TOP/MID tem precedência sobre OCR do acervo.
        # Event é sinalizado pela UI sem aguardar o worker nem bloquear a AOI.
        self._inspection_priority = Event()
        self._deferred_scans = 0
        self._deferred_candidates = 0

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
                "background_priority_applied": self._worker_background_mode,
                "inspection_priority_active": self._inspection_priority.is_set(),
                "deferred_scans": self._deferred_scans,
                "deferred_candidates": self._deferred_candidates,
                "idle_grace_active": (
                    now < self._idle_grace_until
                    and now - self._last_scan_started_at
                    < self.max_idle_deferral_seconds
                    and not self._inspection_priority.is_set()
                ),
                "idle_grace_remaining_ms": round(
                    max(0.0, self._idle_grace_until - now) * 1000.0, 2,
                ),
                "idle_grace_deferrals": self._idle_grace_deferrals,
            }

    def pause_for_inspection(self) -> None:
        """Avisa o worker para NÃO iniciar mais OCR enquanto a peça é analisada.

        Um OCR de PNG já iniciado não pode ser cancelado sem arriscar falha ou
        perda de estado. A pausa é cooperativa no próximo arquivo; não bloqueia
        a thread da interface nem altera os dados persistidos.
        """
        self._inspection_priority.set()

    def resume_after_inspection(self) -> None:
        """Retoma backlog após breve janela para chegada da próxima SIDE.

        A janela é finita: após max_idle_deferral_seconds sem uma passagem,
        o observador aproveita a próxima folga mesmo que os ciclos continuem.
        Liberações repetidas sem pause não prorrogam a janela.
        """
        with self._lock:
            if self._inspection_priority.is_set():
                self._idle_grace_until = max(
                    self._idle_grace_until,
                    time.monotonic() + self.post_inspection_quiet_seconds,
                )
            self._inspection_priority.clear()
        self._wake.set()

    def _defer_scan(self) -> dict:
        with self._lock:
            self._deferred_scans += 1
        return {
            "imported": 0, "pending": 0, "errors": 0,
            "fatal": False, "deferred_for_inspection": True,
        }

    def _ensure_extractor(self):
        if self._extractor is None:
            from src.services.faltando_neural_dataset import AOIPairExtractor
            self._extractor = AOIPairExtractor()
        return self._extractor

    def _list_files(self) -> list[tuple[Path, str]]:
        """Enumera só arquivos regulares sem resolver milhares de caminhos.

        scandir reutiliza metadados do sistema de arquivos e evita o custo
        de Path.rglob + is_file + resolve por PNG (especialmente no Windows).
        Diretórios/arquivos simbólicos nunca são seguidos. Cada caminho
        provém exclusivamente de entradas dentro da raiz selecionada; não
        pode conter '../' injetado. A validação forte do PNG continua em
        _inspect_png e _archive_image antes de persistir qualquer registro.
        A ordem final é idêntica à enumeração anterior.
        """
        paths = []
        for label, dirname in LABEL_DIRS:
            archive = self.root / "public" / dirname
            if not archive.is_dir():
                LOG.error("Arquivo KNN ausente: %s", archive)
                continue
            stack = [archive]
            while stack:
                directory = stack.pop()
                try:
                    with os.scandir(directory) as entries:
                        for item in entries:
                            try:
                                if item.is_symlink():
                                    LOG.warning(
                                        "KNN ignora link simbólico: %s", item.path
                                    )
                                    continue
                                if item.is_dir(follow_symlinks=False):
                                    stack.append(Path(item.path))
                                elif (
                                    item.name.lower().endswith(".png")
                                    and item.is_file(follow_symlinks=False)
                                ):
                                    paths.append((Path(item.path), label))
                            except OSError as exc:
                                LOG.warning(
                                    "KNN nao conseguiu consultar entrada %s: %s",
                                    item.path, exc,
                                )
                except OSError as exc:
                    LOG.warning(
                        "KNN nao conseguiu enumerar pasta %s: %s", directory, exc
                    )
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
        if self._inspection_priority.is_set():
            return self._defer_scan()
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
        for candidate_index, (path, label, relative, stamp) in enumerate(candidates):
            # Um ciclo de produção pode começar durante uma varredura já em
            # andamento. Terminar o PNG corrente e adiar os próximos OCRs,
            # preservando registros propostos para commit normal.
            if self._inspection_priority.is_set():
                with self._lock:
                    self._deferred_candidates += len(candidates) - candidate_index
                break
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
        background = _set_background_thread_mode(True)
        with self._lock:
            self._worker_background_mode = background
        try:
            while not self._stop.is_set():
                if self._inspection_priority.is_set():
                    # Sem enumerar pastas, abrir snapshot ou lançar Tesseract.
                    # A thread acorda imediatamente quando o gate é liberado.
                    with self._lock:
                        self._deferred_scans += 1
                        self._worker_phase = "deferred_inspection"
                    self._wake.wait(self.poll_interval)
                    self._wake.clear()
                    continue
                started = time.monotonic()
                with self._lock:
                    remaining = max(0.0, self._idle_grace_until - started)
                    overdue = (
                        started - self._last_scan_started_at
                        >= self.max_idle_deferral_seconds
                    )
                    if remaining > 0.0 and not overdue:
                        self._idle_grace_deferrals += 1
                        self._worker_phase = "deferred_between_inspections"
                if remaining > 0.0 and not overdue:
                    self._wake.wait(min(self.poll_interval, remaining))
                    self._wake.clear()
                    continue
                with self._lock:
                    self._last_scan_started_at = started
                    self._worker_scan_active = True
                    self._worker_scan_started = started
                    self._worker_phase = "reading_snapshot"
                try:
                    self.scan_once()
                except Exception:
                    LOG.exception(
                        "KNN observador: erro; retentativa na próxima passagem"
                    )
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
        finally:
            if background:
                _set_background_thread_mode(False)
            with self._lock:
                self._worker_background_mode = False

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
        # Nunca manter o worker pausado ao fechar o programa.
        self._inspection_priority.clear()
        with self._lock:
            self._idle_grace_until = 0.0
        self._stop.set()
        self._wake.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)


def activate_archive_auto_memory(watcher: KNNArchiveAutoMemory | None):
    global _active_monitor
    with _active_lock:
        _active_monitor = watcher


def pause_archive_auto_memory_for_inspection() -> None:
    """Sinaliza a prioridade da inspeção sem esperar OCR ou IO de background."""
    with _active_lock:
        worker = _active_monitor
    if worker is not None:
        worker.pause_for_inspection()


def resume_archive_auto_memory_after_inspection() -> None:
    """O observador retoma automaticamente no encerramento da peça."""
    with _active_lock:
        worker = _active_monitor
    if worker is not None:
        worker.resume_after_inspection()


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
    "pause_archive_auto_memory_for_inspection",
    "resume_archive_auto_memory_after_inspection",
]
