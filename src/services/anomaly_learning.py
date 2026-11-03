"""Aprendizado humano: decisão imediata, persistência em segundo plano."""

from __future__ import annotations

import numpy as np

from src.services.decision_persistence import DecisionPersistenceQueue
from src.services.live_session_learning import immediate_confirmed_memory


def _image_snapshot(value):
    if isinstance(value, np.ndarray) and value.size > 0:
        return value.copy()
    return value


LIGHTING_ORDER = ("SIDE", "TOP", "MID")


def _multilight_learning_snapshot(panel) -> list[dict]:
    """Copia SIDE/TOP/MID antes de a interface limpar a peça julgada."""
    samples = getattr(
        panel,
        "adhesive_multilight_learning_samples",
        {},
    )
    if not isinstance(samples, dict):
        return []

    resolved = []
    for mode in LIGHTING_ORDER:
        item = samples.get(mode)
        if not isinstance(item, dict):
            return []

        sample_image = item.get("sample_image")
        test_image = item.get("test_image")
        analysis = item.get("analysis")
        if (
            not isinstance(sample_image, np.ndarray)
            or sample_image.size == 0
            or not isinstance(test_image, np.ndarray)
            or test_image.size == 0
            or not isinstance(analysis, dict)
        ):
            return []

        resolved.append(
            {
                "lighting_mode": mode,
                "sample_image": sample_image.copy(),
                "test_image": test_image.copy(),
                "source_frame": _image_snapshot(item.get("source_frame")),
                "analysis": analysis,
            }
        )
    return resolved


def _partial_shadow_samples(panel) -> list[dict]:
    """Prefixo SIDE/TOP/MID efetivamente obtido ANTES do 0/1 humano.

    Não aguardar novas capturas após o julgamento: a máquina XP pode
    já ter avançado a peça. Nunca salvar frames tardios como a mesma peça.
    """
    samples = getattr(panel, "adhesive_multilight_learning_samples", {})
    if not isinstance(samples, dict):
        return []
    kept = []
    for mode in LIGHTING_ORDER:
        item = samples.get(mode)
        if not isinstance(item, dict):
            break
        ref, test, analysis = (
            item.get("sample_image"),
            item.get("test_image"),
            item.get("analysis"),
        )
        if not (
            isinstance(ref, np.ndarray) and ref.size
            and isinstance(test, np.ndarray) and test.size
            and isinstance(analysis, dict)
        ):
            break
        kept.append({
            "lighting_mode": mode,
            "sample_image": ref.copy(),
            "test_image": test.copy(),
            "source_frame": _image_snapshot(item.get("source_frame")),
            "analysis": analysis,
        })
    return kept


def _decision_task(panel, normalized: str, source: str, ai_decision: str) -> dict:
    event_id = str(
        getattr(panel, "adhesive_multilight_primary_event_id", "")
        or getattr(panel, "adhesive_multilight_last_event_id", "")
        or ""
    ).strip()
    multilight_samples = _multilight_learning_snapshot(panel)

    info = dict(getattr(panel, "current_aoi_info", {}) or {})
    analysis = getattr(panel, "current_analysis", None) or {}
    # Capturas NOVAS confirmadas por humano alimentam a CNN inclusive
    # quando o operador concorda com a IA (não só hard negatives).
    from src.services.neural_online_learning import eligible_online_case
    eligible = eligible_online_case({
        "aoi_info": info, "analysis": analysis,
        "source": source, "label": normalized,
    })

    # Sombra é aquisição supervisionada: TODO evento confirmado no XP
    # deve entrar no dataset mesmo quando a IA concorda. A deduplicação
    # da persistência impede registrar imagens repetidas.
    shadow = (
        getattr(getattr(panel, "combo_mode", None), "currentText", lambda: "")()
        == "Modo Sombra"
    )
    # A tela completa pode ser idêntica ao PNG do archive mesmo quando
    # um OCR ou recorte da AOI variar na nova execução.
    from src.services.capture_evidence import (
        capture_debug_event_id, current_copy_image_event_id,
        current_copy_image_snapshot,
    )
    cycle_source = str(getattr(panel, "capture_cycle_source", "") or "").lower()
    if cycle_source == "local":
        # Captura MSS não pode ser vinculada à última imagem de rede
        # (possivelmente de um ciclo anterior).
        active_id = str(capture_debug_event_id(panel) or "")
    else:
        active_id = str(
            getattr(panel, "adhesive_multilight_primary_event_id", "")
            or getattr(panel, "network_intake_last_image_event_id", "")
            or ""
        )
    evidence_id = str(current_copy_image_event_id(panel) or "")
    source_frame = (
        _image_snapshot(current_copy_image_snapshot(panel))
        if active_id and evidence_id == active_id else None
    )
    if cycle_source == "local":
        # Um estado multilight XP anterior não pode prevalecer sobre o
        # evento MSS atual ao registrar a confirmação humana.
        event_id = evidence_id if source_frame is not None else ""
    elif not event_id and source_frame is not None:
        event_id = evidence_id
    return {
        "source_frame": source_frame,
        "ng_image": _image_snapshot(getattr(panel, "current_ng", None)),
        "label": normalized,
        "sample_image": _image_snapshot(getattr(panel, "current_sample", None)),
        "aoi_info": info,
        "analysis": analysis,
        "save_images": bool(shadow or eligible or ai_decision != normalized),
        "source": source,
        "ai_decision": ai_decision,
        "event_id": event_id,
        "multilight_samples": multilight_samples,
        "shadow_partial_samples": (
            _partial_shadow_samples(panel)
            if shadow and not multilight_samples else []
        ),
    }


def _submit_decision_persistence(panel, task: dict) -> None:
    """Enfileira sem bloquear a thread Qt.

    Testes podem fornecer um submitter substituto para observar a submissão sem
    iniciar trabalho real em background.
    """
    submitter = getattr(panel, "_anomaly_persistence_submitter", None)
    if callable(submitter):
        submitter(task)
        return

    queue = getattr(panel, "_decision_persistence_queue", None)
    if queue is None:
        queue = DecisionPersistenceQueue(getattr(panel, "orchestrator", None))
        panel._decision_persistence_queue = queue
    queue.submit(task)


def _finish_operator_decision_ui(panel, normalized: str, source: str) -> None:
    """Conclui visualmente a peça sem esperar disco/KNN."""
    for name in ("btn_save_ok", "btn_save_ng", "btn_skip"):
        button = getattr(panel, name, None)
        if button is not None:
            button.setEnabled(False)

    panel.is_locked = False

    if getattr(panel, "capture_cycle_source", None) == "network":
        prepare_next = getattr(panel, "prepare_for_next_network_image", None)
        if callable(prepare_next):
            prepare_next()
    else:
        start = getattr(panel, "btn_start", None)
        if start is not None:
            start.setText("Capturar Local (MSS)")
            start.setEnabled(True)
        update_status = getattr(panel, "update_brain_status", None)
        if callable(update_status):
            update_status("Sistema Ocioso", False)

    update_history = getattr(panel, "update_history_status", None)
    if callable(update_history):
        update_history(normalized, source)


def install_anomaly_learning(control_panel_cls) -> None:
    if getattr(control_panel_cls, "_anomaly_learning_installed", False):
        return

    original_save_label = control_panel_cls.save_label

    def save_label(self, user_decision: str, source="button"):
        normalized = str(user_decision or "").strip().upper()
        if normalized not in {"OK", "NG"} or self.current_ng is None:
            return None

        # A saída automática só é gravada como memória OPERACIONAL após
        # confirmação real do envio ao XP. Ela nunca recebe o selo de
        # confirmação humana nem entra no treinamento CNN supervisionado.
        normalized_source = str(source or "").strip().lower()
        if normalized_source == "production_auto":
            from src.services.decision_persistence import (
                _UNVERIFIED_AUTOMATIC_REASONS,
            )
            analysis = getattr(self, "current_analysis", None)
            detail = analysis.get("detail") if isinstance(analysis, dict) else {}
            policy = detail.get("production_decision_policy") if isinstance(
                detail, dict,
            ) else {}
            quarantine = bool(
                isinstance(policy, dict)
                and policy.get("reason") in _UNVERIFIED_AUTOMATIC_REASONS
                and policy.get("resolution") == "automatic"
            )
            # As imagens atuais desaparecem ao concluir o 0/1:
            # copiar cada captura antes de chamar a cadeia original.
            task = _decision_task(
                self, normalized=normalized, source=source,
                ai_decision="NG" if isinstance(analysis, dict)
                and analysis.get("is_defect") else "OK",
            )
            outcome = original_save_label(self, normalized, source=source)
            if (
                outcome is not False
                and getattr(self, "last_decision_command_success", None) is True
            ):
                if quarantine:
                    _submit_decision_persistence(self, task)
                try:
                    from src.core.production_auto_side_recall import (
                        ProductionAutoSideMemory,
                    )
                    saved = ProductionAutoSideMemory().record(task)
                    if not saved:
                        print(
                            "Memória AUTO SIDE: evento sem prova completa "
                            "ou repetição já indexada; sem autoaprendizado."
                        )
                except Exception as exc:
                    # Persistência falha não pode inverter a ordem XP já
                    # transmitida. O incidente permanece auditável no log.
                    print("Memória AUTO SIDE indisponível: " + str(exc))
            return outcome
        if normalized_source == "auto":
            return original_save_label(self, normalized, source=source)

        if source == "button":
            sent = self.send_command_to_xp("0" if normalized == "OK" else "1")
            # O Modo Teste mantém sua semântica existente. Em Produção,
            # não gravar 'verdade humana' se o comando XP local falhou.
            combo = getattr(self, "combo_mode", None)
            mode = combo.currentText() if combo is not None else ""
            if mode == "Modo Produção" and not sent:
                self.last_decision_command_success = False
                notice = getattr(self, "update_brain_status", None)
                if callable(notice):
                    notice(
                        "Correção manual não confirmada pelo XP; "
                        "captura preservada para nova tentativa.", True,
                    )
                return False
            if mode == "Modo Produção":
                self.last_decision_command_success = True

        ai_decision = (
            "NG"
            if self.current_analysis
            and self.current_analysis.get("is_defect", False)
            else "OK"
        )

        # Copia tudo o que o worker precisa ANTES de limpar a captura atual.
        # A fila é serial: decisões sucessivas são persistidas na mesma ordem.
        task = _decision_task(
            self,
            normalized=normalized,
            source=source,
            ai_decision=ai_decision,
        )

        # Commit da memória visual ANTES da próxima inspeção.
        # É o único ponto síncrono para garantir que 2ª imagem idêntica
        # já encontre o 1º rótulo humano, independentemente de worker/OCR.
        # Treinamento CNN pesado continua em processo separado.
        try:
            memory_writer = getattr(self, "_live_memory_writer", None)
            learned = (
                memory_writer(task) if callable(memory_writer)
                else immediate_confirmed_memory(task)
            )
            if learned.get("committed"):
                knn = getattr(
                    getattr(self, "orchestrator", None), "experts", {}
                ).get("knn")
                refresh = getattr(knn, "refresh_archive_memory", None)
                if callable(refresh):
                    refresh()
            else:
                print("KNN live: confirmação não importada: " +
                      str(learned.get("reason")))
        except Exception as exc:
            # Uma falha de disco NUNCA deve criar sucesso fictício.
            print("KNN live: memória NÃO persistida: " + str(exc))

        # Caminho crítico termina aqui: a peça já está julgada. A interface pode
        # limpar o resultado e o gate externo pode liberar a próxima imagem.
        _finish_operator_decision_ui(self, normalized, source)

        # Disco + recarga KNN deixam de bloquear o operador e o receptor XP.
        _submit_decision_persistence(self, task)
        return None

    control_panel_cls.save_label = save_label
    control_panel_cls._anomaly_learning_installed = True


__all__ = [
    "LIGHTING_ORDER",
    "_decision_task",
    "_multilight_learning_snapshot",
    "_finish_operator_decision_ui",
    "_submit_decision_persistence",
    "install_anomaly_learning",
]
