"""Instala o painel explicável de decisão sem alterar o controller principal."""

from __future__ import annotations

from types import MethodType

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy

from src.ui.decision_model import (
    decision_summary,
    decision_trace_from_analysis,
    fusion_summary,
    memory_summary,
)
from src.ui.widgets.decision_influence import DecisionInfluenceWidget


def _render_panel(panel, analysis: dict | None) -> None:
    trace = decision_trace_from_analysis(analysis)

    panel.frame_decision_influence.update_data(analysis)

    if not trace:
        panel.lbl_decision_summary.setText("Aguardando rastreamento da decisão.")
        panel.lbl_decision_rule.setText("Sem regra de fusão disponível.")
        panel.lbl_memory_role.setText("SEM MEMÓRIA")
        return

    panel.lbl_decision_summary.setText(decision_summary(trace))
    panel.lbl_decision_rule.setText(fusion_summary(trace))

    primary, role = memory_summary(trace)
    panel.lbl_db_info.setText(primary)
    panel.lbl_memory_role.setText(role)

    is_defect = bool((analysis or {}).get("is_defect", False))
    verdict_color = "#ff6262" if is_defect else "#4ade80"
    panel.lbl_verdict.setStyleSheet(
        f"color: {verdict_color}; font-size: 17px; "
        "font-weight: 800; border: none; background: transparent;"
    )

    rule = str(trace.get("fusion_rule", "physical_only"))
    role_color = "#ff6262" if rule in {"memory_veto", "memory_override"} else "#f5c518"
    panel.lbl_memory_role.setStyleSheet(
        f"color: {role_color}; font-size: 11px; font-weight: 800; "
        "border: 1px solid #3a3a3a; border-radius: 5px; "
        "padding: 4px 7px; background: #101010;"
    )


def install_decision_panel(panel) -> None:
    """Reconstrói a seção inferior e mantém os breakpoints já existentes."""
    if getattr(panel, "_decision_panel_installed", False):
        return

    builder = panel.ui_builder
    grid = builder.footer_grid

    old_cards = list(builder.footer_cards)
    for card in old_cards:
        grid.removeWidget(card)

    verdict_card, verdict_layout = builder._create_footer_card(
        "VEREDITO E REGRA APLICADA"
    )
    panel.lbl_verdict.setAlignment(
        Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
    )
    panel.lbl_verdict.setWordWrap(True)
    panel.lbl_verdict.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Preferred,
    )

    panel.lbl_decision_summary = QLabel("Aguardando rastreamento da decisão.")
    panel.lbl_decision_summary.setObjectName("decisionSummary")
    panel.lbl_decision_summary.setWordWrap(True)
    panel.lbl_decision_summary.setAlignment(Qt.AlignmentFlag.AlignCenter)
    panel.lbl_decision_summary.setStyleSheet(
        "color: #f5c518; font-size: 11px; font-weight: 800; "
        "border: none; background: transparent;"
    )

    panel.lbl_decision_rule = QLabel("Sem regra de fusão disponível.")
    panel.lbl_decision_rule.setObjectName("decisionRule")
    panel.lbl_decision_rule.setWordWrap(True)
    panel.lbl_decision_rule.setAlignment(Qt.AlignmentFlag.AlignCenter)
    panel.lbl_decision_rule.setStyleSheet(
        "color: #a6a6a6; font-size: 10px; font-weight: 700; "
        "border: none; background: transparent;"
    )

    panel.lbl_reason.setAlignment(Qt.AlignmentFlag.AlignCenter)
    panel.lbl_reason.setWordWrap(True)

    verdict_layout.addWidget(panel.lbl_verdict)
    verdict_layout.addWidget(panel.lbl_decision_summary)
    verdict_layout.addWidget(panel.lbl_decision_rule)
    verdict_layout.addWidget(panel.lbl_reason, stretch=1)

    influence_card, influence_layout = builder._create_footer_card(
        "INFLUÊNCIA DOS MOTORES"
    )
    panel.frame_decision_influence = DecisionInfluenceWidget()
    panel.frame_decision_influence.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding,
    )
    influence_layout.addWidget(panel.frame_decision_influence, stretch=1)

    memory_card, memory_layout = builder._create_footer_card(
        "MEMÓRIA LOCAL • KNN"
    )
    panel.lbl_memory_role = QLabel("SEM MEMÓRIA")
    panel.lbl_memory_role.setAlignment(Qt.AlignmentFlag.AlignCenter)
    panel.lbl_memory_role.setObjectName("memoryRole")
    panel.lbl_db_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
    panel.lbl_db_info.setWordWrap(True)
    panel.frame_knn.setMinimumHeight(130)
    panel.frame_knn.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding,
    )

    memory_layout.addWidget(panel.lbl_memory_role)
    memory_layout.addWidget(panel.lbl_db_info)
    memory_layout.addWidget(panel.frame_knn, stretch=1)

    builder.footer_cards = [verdict_card, influence_card, memory_card]
    panel.metric_labels = {}

    for old_card in old_cards:
        old_card.hide()
        old_card.deleteLater()

    builder.apply_layout_profile(panel, max(panel.width(), 1), force=True)

    original_reference_update = panel._update_reference_panel
    original_reset = panel._reset_confidence_panel

    def wrapped_reference_update(self, analysis):
        result = original_reference_update(analysis)
        _render_panel(self, analysis)
        return result

    def wrapped_reset(self):
        result = original_reset()
        self.frame_decision_influence.update_data({})
        self.lbl_decision_summary.setText("Aguardando rastreamento da decisão.")
        self.lbl_decision_rule.setText("Sem regra de fusão disponível.")
        self.lbl_memory_role.setText("SEM MEMÓRIA")
        return result

    panel._update_reference_panel = MethodType(wrapped_reference_update, panel)
    panel._reset_confidence_panel = MethodType(wrapped_reset, panel)
    panel._decision_panel_installed = True
