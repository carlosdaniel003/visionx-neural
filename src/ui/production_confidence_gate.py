"""Política de intervenção humana do Modo Produção.

A versão atual não usa mais confiança mínima de 99% para autorizar OK
automático. O veredito final já renderizado é a referência:

- FALHA FALSA -> pode ser automatizado como 0/OK;
- DEFEITO REAL/NG -> intervenção humana;
- REVISÃO OBRIGATÓRIA -> intervenção humana.

A apresentação visual e o momento do envio automático pertencem ao
ProductionAutonomyController. Este módulo mantém apenas a trava operacional de
revisão e a compatibilidade com decisões 0/1 do operador.
"""

from __future__ import annotations

from typing import Any

from src.core.corroborated_ng_verdict import corroborated_ng_verdict
from src.core.knn_c_supervised_policy import supervised_knn_c_decision
from src.core.neural.faltando_category_scope import uses_faltando_v2
from src.core.neural.faltando_checkpoint_contract import TRIAL_97P70_SHA256
from src.core.side_early_decision import (
    SIDE_EARLY_EXIT_THRESHOLD,
    side_automatic_decision,
    side_trial_automatic_decision,
)


# Mantido por compatibilidade com imports antigos. Não participa mais da política.
PRODUCTION_AUTO_CONFIDENCE_THRESHOLD = None


def normalized_confidence(analysis: dict | None) -> float:
    """Mantém a confiança como telemetria, sem usá-la como gate de decisão."""
    try:
        value = float((analysis or {}).get("confidence", 0.0))
    except (TypeError, ValueError):
        value = 0.0
    if value != value:
        value = 0.0
    return max(0.0, min(1.0, value))


def _review_required(analysis: dict | None) -> bool:
    if not isinstance(analysis, dict):
        return True

    detail = analysis.get("detail", {})
    detail = detail if isinstance(detail, dict) else {}
    trace = detail.get("decision_trace", {})
    trace = trace if isinstance(trace, dict) else {}

    return bool(
        analysis.get("production_review_required", False)
        or detail.get("operator_review_required", False)
        or trace.get("operator_review_required", False)
    )


def _verdict(analysis: dict | None) -> str:
    if not isinstance(analysis, dict):
        return "REVISÃO OBRIGATÓRIA"

    verdict = str(analysis.get("verdict", "") or "").strip().upper()
    if verdict in {
        "FALHA FALSA",
        "DEFEITO REAL",
        "DEFEITO",
        "REVISÃO OBRIGATÓRIA",
    }:
        return "DEFEITO REAL" if verdict == "DEFEITO" else verdict

    if _review_required(analysis):
        return "REVISÃO OBRIGATÓRIA"

    if "is_defect" in analysis:
        return "DEFEITO REAL" if bool(analysis.get("is_defect")) else "FALHA FALSA"

    return "REVISÃO OBRIGATÓRIA"


def production_decision_policy(analysis: dict | None) -> dict[str, Any]:
    """Resolve a política v1 sem limiar mínimo de confiança."""
    verdict = _verdict(analysis)
    confidence = normalized_confidence(analysis)

    detail = (analysis or {}).get("detail", {}) if isinstance(analysis, dict) else {}
    detail = detail if isinstance(detail, dict) else {}
    # Contraprova física independente: uma inconsistência ou ausência dos
    # pares visuais proíbe 0/OK até confirmação humana. Esse segundo gate
    # continua válido mesmo se uma tela intermediária mudar o veredito.
    # Histórico trial permanece registrado na auditoria. Uma exceção de
    # Produção explicitamente validada só se aplica à SIDE final com contrato
    # de inferência, SHA, direção OK/NG, memória e episódio íntegros.
    diagnostics = detail.get("cnn_v2_light_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    trial_used = bool(
        detail.get("cnn_v2_checkpoint_sha256") == TRIAL_97P70_SHA256
        or detail.get("cnn_v2_trial_manual_only") is True
        or any(
            isinstance(data, dict) and (
                data.get("checkpoint_sha256") == TRIAL_97P70_SHA256
                or data.get("trial_manual_only") is True
            ) for data in diagnostics.values()
        )
    )
    trial_side = side_trial_automatic_decision(analysis) if trial_used else None
    if trial_side in {"OK", "NG"}:
        return {
            "mode": "production", "confidence": confidence,
            "threshold": SIDE_EARLY_EXIT_THRESHOLD,
            "verdict": "FALHA FALSA" if trial_side == "OK" else "DEFEITO REAL",
            "auto_allowed": True, "operator_review_required": False,
            "proposed_decision": trial_side,
            "reason": "side_trial_verified_auto_" + trial_side.lower(),
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }
    if trial_used:
        # Uma referência OK de captura recorrente pode recuperar a
        # CLASSIFICAÇÃO do operador na interface, mas nunca transmitir 0.
        # Validar novamente a identidade da fonte e as evidências brutas;
        # o flag calculado pela fusão não basta para liberar a placa.
        from src.core.knn_visual_corroboration import (
            _operator_confirmed_ok_pattern_review, _recognized_recurrent_ok,
            _recognized_two_light_operator_ok,
        )
        recalled = detail.get("recognized_recurrent_ok")
        pattern = detail.get("operator_confirmed_ok_pattern_review")
        counterevidence = detail.get("knn_ok_counterevidence")
        two_light = detail.get("recognized_two_light_operator_ok")
        trace = detail.get("decision_trace")
        verified_recall = _recognized_recurrent_ok(detail, counterevidence)
        verified_pattern = _operator_confirmed_ok_pattern_review(
            detail, counterevidence,
        )
        valid_known_ok = bool(
            isinstance(recalled, dict)
            and verified_recall is not None
            and recalled == verified_recall
            and detail.get("fusion_rule")
            == "operator_confirmed_recurrent_ok_three_light_review_v1"
        )
        valid_pattern = bool(
            isinstance(pattern, dict)
            and verified_pattern is not None
            and pattern == verified_pattern
            and detail.get("fusion_rule")
            == "operator_confirmed_ok_pattern_three_light_review_v1"
        )
        verified_two_light = _recognized_two_light_operator_ok(
            {
                **analysis, "verdict": detail.get("cnn_ng_verdict_before_knn_review"),
                "is_defect": detail.get("cnn_ng_verdict_before_knn_review")
                == "DEFEITO REAL",
            } if isinstance(analysis, dict) else {},
            detail.get("knn_visual_corroboration"), "FALTANDO",
        )
        # Recompute using original CNN NG evidence rather than trusting
        # a final UI label; the helper accepts only raw NG multilight.
        valid_two_light = bool(
            isinstance(two_light, dict)
            and verified_two_light is not None
            and two_light == verified_two_light
            and detail.get("fusion_rule")
            == "operator_recurrent_ok_top_mid_review_v1"
        )
        if (
            verdict == "FALHA FALSA"
            and (valid_known_ok or valid_pattern or valid_two_light)
            and detail.get("operator_review_required") is True
            and isinstance(analysis, dict)
            and analysis.get("production_review_required") is True
            and analysis.get("eligible_for_final_decision") is False
            and isinstance(trace, dict)
            and trace.get("operator_review_required") is True
            and trace.get("verdict") == "FALHA FALSA"
            and detail.get("physical_score") == 0.0
            and detail.get("cnn_v2_consensus") == "REVIEW"
        ):
            return {
                "mode": "production",
                "confidence": confidence,
                "threshold": None,
                "verdict": "FALHA FALSA",
                "auto_allowed": False,
                "operator_review_required": True,
                "proposed_decision": "",
                "reason": (
                    "operator_confirmed_recurring_ok_requires_operator"
                    if valid_known_ok
                    else "operator_recurrent_ok_top_mid_requires_operator"
                    if valid_two_light
                    else "operator_confirmed_ok_pattern_requires_operator"
                ),
                "operator_shortcuts": {"0": "OK", "1": "NG"},
            }
        return {
            "mode": "production", "confidence": confidence, "threshold": None,
            "verdict": verdict if verdict == "DEFEITO REAL"
                       else "REVISÃO OBRIGATÓRIA",
            "auto_allowed": False, "operator_review_required": True,
            "proposed_decision": "NG" if verdict == "DEFEITO REAL" else "",
            "reason": "cnn_97p70_unqualified_operator_review_only",
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }
    physical_veto = detail.get("physical_missing_veto", {})
    physical_veto = physical_veto if isinstance(physical_veto, dict) else {}
    # A apresentação de NG corroborado e a emissão 1/NG são dimensões
    # diferentes: mesmo sem SUSPECT_ABSENCE a convergência KNN tricolor
    # pode classificar NG, mas NUNCA autoriza XP automático.
    if corroborated_ng_verdict(analysis):
        return {
            "mode": "production", "confidence": confidence,
            "threshold": None, "verdict": "DEFEITO REAL",
            "auto_allowed": False, "operator_review_required": True,
            "proposed_decision": "NG",
            "reason": "corroborated_ng_requires_human_confirmation",
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }
    physical_blocked = bool(
        detail.get("multilight_category") == "FALTANDO"
        and physical_veto.get("status") in {"SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW"}
    )
    # Um caminho futuro que esqueça de executar a sonda física também não
    # pode reutilizar três votos CNN OK como se tivessem contraprova.
    physical_required = bool(
        isinstance(analysis, dict)
        and analysis.get("multilight_final") is True
        and detail.get("multilight_category") == "FALTANDO"
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("cnn_v2_consensus") == "OK"
        and detail.get("cnn_v2_supervised_auto_eligible") is True
        and physical_veto.get("status") != "NO_STRONG_EVIDENCE"
    )
    if physical_blocked or physical_required:
        corroborated_ng = corroborated_ng_verdict(analysis)
        return {
            "mode": "production",
            "confidence": confidence,
            "threshold": None,
            "verdict": "DEFEITO REAL" if corroborated_ng
                       else "REVISÃO OBRIGATÓRIA",
            "auto_allowed": False,
            "operator_review_required": True,
            "proposed_decision": "NG" if corroborated_ng else "",
            "reason": (
                "corroborated_ng_requires_human_confirmation" if corroborated_ng
                else "cnn_faltando_independent_physical_veto" if physical_blocked
                else "cnn_faltando_physical_evidence_required"
            ),
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }

    # Divergência forte entre vizinhos NG e CNN OK veta AUTO-OK inclusive
    # se outra camada visual reescrever o veredito de apresentação.
    neighbors = detail.get("knn_visual_corroboration")
    neighbors = neighbors if isinstance(neighbors, dict) else {}
    if (
        neighbors.get("status") == "NG_CORROBORATED"
        and detail.get("cnn_v2_consensus") == "OK"
    ):
        return {
            "mode": "production", "confidence": confidence, "threshold": None,
            "verdict": "REVISÃO OBRIGATÓRIA",
            "auto_allowed": False, "operator_review_required": True,
            "proposed_decision": "",
            "reason": "cnn_knn_visual_ng_disagreement",
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }

    # Estratégia C: somente o operador pode armá-la por sessão.
    # Executa DEPOIS dos vetos físicos, checkpoint trial e corroboração NG,
    # e ANTES das políticas genéricas. O controlador revalida no envio.
    selected_c = supervised_knn_c_decision(analysis)
    if selected_c in {"OK", "NG"}:
        return {
            "mode": "production", "confidence": confidence,
            "threshold": None,
            "verdict": "FALHA FALSA" if selected_c == "OK" else "DEFEITO REAL",
            "auto_allowed": True,
            "operator_review_required": False,
            "proposed_decision": selected_c,
            "reason": "knn_c_supervised_three_light_" + selected_c.lower(),
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }

    # A SIDE antecipada é uma alternativa explícita ao consenso tricolor.
    # O contrato completo da SIDE deve ser validado novamente aqui; todos
    # os vetos operacionais anteriores continuam tendo precedência.
    early_decision = side_automatic_decision(analysis)
    if early_decision in {"OK", "NG"}:
        return {
            "mode": "production",
            "confidence": confidence,
            "threshold": SIDE_EARLY_EXIT_THRESHOLD,
            "verdict": verdict,
            "auto_allowed": True,
            "operator_review_required": False,
            "proposed_decision": early_decision,
            "reason": "side_97pct_auto_" + early_decision.lower(),
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }

    # Um marcador SIDE existente, porém inconsistente, nunca pode cair
    # no auto-OK legado nem ser reinterpretado como consenso tricolor.
    if "side_early_exit" in detail:
        return {
            "mode": "production",
            "confidence": confidence,
            "threshold": SIDE_EARLY_EXIT_THRESHOLD,
            "verdict": (
                "DEFEITO REAL" if verdict == "DEFEITO REAL"
                else "REVISÃO OBRIGATÓRIA"
            ),
            "auto_allowed": False,
            "operator_review_required": True,
            "proposed_decision": "NG" if verdict == "DEFEITO REAL" else "",
            "reason": "side_early_exit_invalid_contract",
            "operator_shortcuts": {"0": "OK", "1": "NG"},
        }

    # AUTO-OK ou AUTO-NG pela CNN experimental somente se TRÊS luzes
    # verificadas concordarem, com mesmo SHA de modelo e sem revisão.
    # Uma pontuação isolada, até extrema, não autoriza nenhum comando.
    multilight_routes = detail.get("recognition_light_routes", {})
    multilight_routes = (
        multilight_routes if isinstance(multilight_routes, dict) else {}
    )
    cnn_experimental = bool(
        (detail.get("cnn_v2_active") and detail.get("cnn_v2_experimental"))
        or any(route == "NEW_CNN" for route in multilight_routes.values())
        or detail.get("recognition_route") == "NEW_CNN"
    )
    consensus = detail.get("cnn_v2_consensus")
    votes = detail.get("cnn_v2_light_votes", {})
    verified_lights = detail.get("cnn_v2_verified_lights", [])
    verified_lights = verified_lights if isinstance(verified_lights, list) else []
    expected_vote = "OK" if verdict == "FALHA FALSA" else (
        "NG" if verdict == "DEFEITO REAL" else ""
    )
    cnn_auto = bool(
        cnn_experimental
        and isinstance(analysis, dict)
        and analysis.get("multilight_final") is True
        and analysis.get("eligible_for_final_decision") is True
        and detail.get("multilight_final") is True
        and detail.get("eligible_for_final_decision") is True
        and uses_faltando_v2(detail.get("multilight_category", ""))
        and detail.get("recognition_route") == "NEW_CNN"
        and detail.get("cnn_v2_supervised_auto_eligible") is True
        and consensus == expected_vote
        and expected_vote in {"OK", "NG"}
        and isinstance(votes, dict)
        and set(votes) == {"SIDE", "TOP", "MID"}
        and all(votes[light] == expected_vote for light in votes)
        and set(verified_lights) == {"SIDE", "TOP", "MID"}
        and not _review_required(analysis)
    )
    if cnn_auto:
        proposed_decision = expected_vote
        auto_allowed = True
        operator_review_required = False
        reason = "faltando_cnn_v2_supervised_three_light_consensus"
    elif cnn_experimental:
        auto_allowed = False
        operator_review_required = True
        if verdict == "DEFEITO REAL":
            proposed_decision = "NG"
            reason = "faltando_cnn_v2_ng_requires_operator"
        else:
            proposed_decision = ""
            verdict = "REVISÃO OBRIGATÓRIA"
            reason = "faltando_cnn_v2_experimental_requires_operator"
    elif verdict == "FALHA FALSA" and not _review_required(analysis):
        proposed_decision = "OK"
        auto_allowed = True
        operator_review_required = False
        reason = "false_failure_auto_ok"
    elif verdict == "DEFEITO REAL":
        proposed_decision = "NG"
        auto_allowed = False
        operator_review_required = True
        reason = "real_defect_manual_review"
    else:
        proposed_decision = ""
        auto_allowed = False
        operator_review_required = True
        reason = "mandatory_manual_review"

    return {
        "mode": "production",
        "confidence": confidence,
        "threshold": None,
        "verdict": verdict,
        "auto_allowed": bool(auto_allowed),
        "operator_review_required": bool(operator_review_required),
        "proposed_decision": proposed_decision,
        "reason": reason,
        "operator_shortcuts": {"0": "OK", "1": "NG"},
    }


def _record_policy(panel, policy: dict, resolution: str = "pending") -> None:
    panel.production_review_policy = dict(policy)
    analysis = getattr(panel, "current_analysis", None)
    if isinstance(analysis, dict):
        detail = analysis.setdefault("detail", {})
        detail["production_decision_policy"] = {
            **policy,
            "resolution": str(resolution),
        }
        # Uma peça classificada NG também pode precisar do operador para
        # enviar 1 ao XP. O flag sinaliza ação PENDENTE, não a classe.
        analysis["production_review_required"] = bool(
            policy.get("operator_review_required", False)
            and resolution == "pending"
        )


def _clear_pending(panel) -> None:
    panel.production_review_pending = False
    analysis = getattr(panel, "current_analysis", None)
    if isinstance(analysis, dict):
        analysis["production_review_required"] = False


def _show_pending_message(panel) -> None:
    policy = getattr(panel, "production_review_policy", None)
    if not isinstance(policy, dict):
        policy = production_decision_policy(
            getattr(panel, "current_analysis", None)
        )

    verdict = str(policy.get("verdict", "") or "REVISÃO OBRIGATÓRIA")
    panel.update_brain_status(
        f"Intervenção necessária: {verdict}. "
        "Aguardando operador: 0=OK | 1=NG",
        True,
    )
    if hasattr(panel, "_operational_controls"):
        panel._operational_controls.sync(force=True)


def enter_production_review(
    panel,
    analysis: dict | None = None,
    *,
    verdict_override: str = "",
) -> dict[str, Any]:
    """Congela a peça e libera somente 0/1 humano."""
    target = analysis if isinstance(analysis, dict) else getattr(
        panel,
        "current_analysis",
        None,
    )
    policy = production_decision_policy(target)

    override = str(verdict_override or "").strip().upper()
    if override:
        if override not in {"DEFEITO REAL", "REVISÃO OBRIGATÓRIA"}:
            override = "REVISÃO OBRIGATÓRIA"
        policy = {
            **policy,
            "verdict": override,
            "auto_allowed": False,
            "operator_review_required": True,
            "proposed_decision": "NG" if override == "DEFEITO REAL" else "",
            "reason": "forced_manual_intervention",
        }

    panel.production_review_pending = True
    panel.is_locked = True
    _record_policy(panel, policy, resolution="pending")
    _show_pending_message(panel)
    return policy


def install_production_confidence_gate(control_panel_cls, presenter_cls) -> None:
    """Mantém a trava humana enquanto o controlador gerencia a autonomia."""
    if getattr(control_panel_cls, "_production_confidence_gate_installed", False):
        return

    from PyQt6.QtCore import Qt

    original_init = control_panel_cls.__init__
    original_start_monitoring = control_panel_cls.start_monitoring
    original_handle_network_image = control_panel_cls.handle_network_image
    original_skip_image = control_panel_cls.skip_image
    original_save_label = control_panel_cls.save_label
    original_handle_physical_keyboard = control_panel_cls.handle_physical_keyboard
    original_key_press = control_panel_cls.keyPressEvent
    original_presenter_sync = presenter_cls.sync

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.production_review_pending = False
        self.production_review_policy = None

    def wrapped_start_monitoring(self, *args, **kwargs):
        if getattr(self, "production_review_pending", False):
            _show_pending_message(self)
            return None
        return original_start_monitoring(self, *args, **kwargs)

    def wrapped_handle_network_image(self, *args, **kwargs):
        if getattr(self, "production_review_pending", False):
            _show_pending_message(self)
            return None
        return original_handle_network_image(self, *args, **kwargs)

    def wrapped_skip_image(self, *args, **kwargs):
        if getattr(self, "production_review_pending", False):
            _show_pending_message(self)
            return None
        return original_skip_image(self, *args, **kwargs)

    def wrapped_save_label(self, user_decision: str, source="button"):
        mode = self.combo_mode.currentText() if hasattr(self, "combo_mode") else ""
        is_production = str(mode).strip() == "Modo Produção"
        normalized_source = str(source or "").strip().lower()
        had_pending = bool(getattr(self, "production_review_pending", False))

        if is_production and normalized_source in {"auto", "production_auto"}:
            policy = production_decision_policy(
                getattr(self, "current_analysis", None)
            )
            if not (
                policy["auto_allowed"]
                and str(user_decision).strip().upper() == policy["proposed_decision"]
                and not bool(getattr(
                    getattr(self, "production_autonomy_controller", None),
                    "paused", False,
                ))
            ):
                enter_production_review(
                    self,
                    getattr(self, "current_analysis", None),
                )
                return None
            _record_policy(self, policy, resolution="automatic")

        elif is_production and had_pending:
            policy = production_decision_policy(
                getattr(self, "current_analysis", None)
            )
            resolution = (
                "operator_ok"
                if str(user_decision).upper() == "OK"
                else "operator_ng"
            )
            _record_policy(self, policy, resolution=resolution)

        _clear_pending(self)
        result = original_save_label(self, user_decision, source=source)

        if (
            is_production
            and had_pending
            and not bool(getattr(self, "is_locked", False))
        ):
            callback = getattr(
                self,
                "production_operator_decision_completed",
                None,
            )
            if callable(callback):
                callback(user_decision, source=source)

        return result

    def wrapped_handle_physical_keyboard(self, comando_xp: str):
        command = str(comando_xp or "").strip().upper()
        mode = self.combo_mode.currentText() if hasattr(self, "combo_mode") else ""
        is_production = str(mode).strip() == "Modo Produção"
        pending = bool(getattr(self, "production_review_pending", False))

        if command in {"0", "OK"}:
            if is_production and not pending:
                return None
            return self.save_label("OK", source="xp_keyboard")
        if command in {"1", "NG"}:
            if is_production and not pending:
                return None
            return self.save_label("NG", source="xp_keyboard")
        if pending:
            _show_pending_message(self)
            return None
        return original_handle_physical_keyboard(self, comando_xp)

    def wrapped_key_press(self, event):
        pending = bool(getattr(self, "production_review_pending", False))
        if pending:
            if event.key() == Qt.Key.Key_0 and self.btn_save_ok.isEnabled():
                self.save_label("OK", source="button")
                event.accept()
                return
            if event.key() == Qt.Key.Key_1 and self.btn_save_ng.isEnabled():
                self.save_label("NG", source="button")
                event.accept()
                return
            event.accept()
            _show_pending_message(self)
            return
        return original_key_press(self, event)

    def wrapped_presenter_sync(self, force: bool = False):
        original_presenter_sync(self, force=force)
        panel = self.panel
        mode = panel.combo_mode.currentText() if hasattr(panel, "combo_mode") else ""
        is_production = str(mode).strip() == "Modo Produção"
        pending = bool(getattr(panel, "production_review_pending", False))

        decision_buttons_visible = (not is_production) or pending
        panel.btn_save_ok.setVisible(decision_buttons_visible)
        panel.btn_save_ng.setVisible(decision_buttons_visible)

        if not (is_production and pending):
            return

        self._set_enabled(panel.btn_start, False)
        self._set_enabled(panel.btn_skip, False)
        self._set_enabled(panel.btn_save_ok, True)
        self._set_enabled(panel.btn_save_ng, True)

        for button in (
            panel.btn_light_mid,
            panel.btn_light_side,
            panel.btn_light_top,
        ):
            self._set_enabled(button, False)

        if hasattr(panel, "btn_clear_dataset"):
            self._set_enabled(panel.btn_clear_dataset, False)

        self._set_text(
            panel.btn_start,
            "Captura congelada — aguardando operador",
        )
        self._set_text(panel.btn_save_ok, "0 - Aprovar como OK")
        self._set_text(panel.btn_save_ng, "1 - Confirmar defeito NG")

        policy = getattr(panel, "production_review_policy", {}) or {}
        verdict = str(
            policy.get("verdict", "REVISÃO OBRIGATÓRIA") or
            "REVISÃO OBRIGATÓRIA"
        )

        panel.lbl_operation_state.setText("INTERVENÇÃO NECESSÁRIA")
        panel.lbl_operation_state.setProperty("tone", "attention")
        self._refresh_style(panel.lbl_operation_state)
        panel.lbl_operation_hint.setText(
            f"{verdict}. Pressione 0 para OK ou 1 para NG."
        )
        panel.lbl_operation_actions.setText("2 AÇÕES DISPONÍVEIS")
        self.last_state_name = "production_review"

    control_panel_cls.__init__ = wrapped_init
    control_panel_cls.start_monitoring = wrapped_start_monitoring
    control_panel_cls.handle_network_image = wrapped_handle_network_image
    control_panel_cls.skip_image = wrapped_skip_image
    control_panel_cls.save_label = wrapped_save_label
    control_panel_cls.handle_physical_keyboard = wrapped_handle_physical_keyboard
    control_panel_cls.keyPressEvent = wrapped_key_press
    presenter_cls.sync = wrapped_presenter_sync
    control_panel_cls._production_confidence_gate_installed = True


__all__ = [
    "PRODUCTION_AUTO_CONFIDENCE_THRESHOLD",
    "enter_production_review",
    "install_production_confidence_gate",
    "normalized_confidence",
    "production_decision_policy",
]
