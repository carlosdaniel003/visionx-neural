"""Resumo visual das novas camadas de memória no painel principal."""

from __future__ import annotations

import json
from pathlib import Path

from src.ui.memory_status_model import memory_status_from_detail, memory_summary_text


def install_memory_status_ui(control_panel_cls) -> None:
    """Atualiza apenas textos da UI; não interfere na decisão da inspeção."""
    if getattr(control_panel_cls, "_memory_status_ui_installed", False):
        return

    original_update_confidence = control_panel_cls._update_confidence_panel

    def update_confidence_panel(self, analysis: dict):
        original_update_confidence(self, analysis)
        detail = (analysis or {}).get("detail", {})
        model = memory_status_from_detail(detail)

        # Rota explicitada na própria linha existente e no tooltip.
        route = str(detail.get("recognition_route", "") or "")
        routes = detail.get("recognition_light_routes", {})
        route_labels = {
            "KNOWN_KNN": "CASO CONHECIDO • MEMÓRIA KNN",
            "NEW_CNN": "CASO NOVO • CNN FALTANDO v2",
            "NEW_EXPERTS": "CASO NOVO • MOTORES DA CATEGORIA",
            "MEMORY_CONFLICT": "MEMÓRIA CONTRADITÓRIA • REVISÃO",
            "MULTILIGHT_MIXED": "MULTILIGHT • KNN + ESPECIALISTAS",
        }
        if route:
            tooltip_lines = [
                "ROTEAMENTO DA INSPEÇÃO",
                route_labels.get(route, route),
                "",
                "Um caso só é conhecido quando gabarito e teste são",
                "idênticos aos PNGs de um registro humano OK/NG na",
                "mesma placa, componente, categoria e iluminação.",
                "Similaridade KNN aproximada não libera a peça.",
            ]
            if route == "KNOWN_KNN":
                tooltip_lines += [
                    "",
                    "KNN: recuperação do rótulo humano de um par exato.",
                    "CNN e demais motores não foram executados.",
                    "Rótulo: " + str(detail.get("recognition_known_label", "-")),
                    "Registro: " + str(detail.get("recognition_memory_path", "-")),
                ]
            elif route == "NEW_CNN":
                tooltip_lines += [
                    "", "Memória KNN ignorada nesta decisão.",
                    "Motor: CNN FALTANDO v2, com referência e teste.",
                    "Em Produção, o OK da CNN experimental exige operador.",
                    "",
                    "APRENDIZADO INCREMENTAL:",
                    "Após OK/NG confirmado por operador (Teste/Produção/Sombra),",
                    "o treino CPU é enfileirado sem interromper a AOI.",
                    "O checkpoint ativo muda só quando o candidato passa",
                    "a regressão do arquivo e dos novos casos confirmados.",
                ]
                try:
                    status_path = (
                        Path(__file__).resolve().parents[2]
                        / "reports" / "neural_online" / "latest_event.json"
                    )
                    if status_path.is_file():
                        latest = json.loads(status_path.read_text(encoding="utf-8"))
                        status = str(latest.get("state", ""))
                        display = {
                            "QUEUED": "NA FILA",
                            "TRAINING": "TREINANDO EM SEGUNDO PLANO",
                            "PROMOTED": "NOVOS PESOS ATIVADOS",
                            "REJECTED": "CANDIDATO REPROVADO (PESOS ANTIGOS ATIVOS)",
                            "FAILED": "FALHA NO TREINAMENTO",
                            "COMPLETED": "EXECUÇÃO CONCLUÍDA",
                        }
                        if status in display:
                            tooltip_lines += [
                                "Último treino: " + display[status],
                                "Evento: " + str(latest.get("event_id", "")),
                            ]
                except (OSError, ValueError, TypeError):
                    pass
            elif route == "NEW_EXPERTS":
                tooltip_lines += [
                    "", "Memória KNN ignorada nesta decisão.",
                    "Motores da categoria consultados como primeira ocorrência.",
                ]
                if detail.get("specialist_candidate") == (
                    "DESLOCADO_CNN_V1_BOOTSTRAP_NOT_ACTIVE"
                ):
                    tooltip_lines += [
                        "", "CNN DESLOCADO: EM TREINAMENTO EXPERIMENTAL.",
                        "NG DESLOCADO reais conhecidos: nenhum no acervo inicial.",
                        "OK/NG humano novo é enfileirado para treino candidato.",
                        "A CNN DESLOCADO NÃO substitui os motores atuais.",
                        "A ativação exigirá NG real e avaliação independente.",
                    ]
            if isinstance(routes, dict) and routes:
                tooltip_lines += [
                    "", "POR ILUMINAÇÃO:",
                    *[
                        f"{mode}: {route_labels.get(routes.get(mode), routes.get(mode, '-'))}"
                        for mode in ("SIDE", "TOP", "MID")
                    ],
                ]
            tooltip = "\n".join(tooltip_lines)
            # O label é visível sem passar o mouse; tooltip guarda os detalhes.
            if hasattr(self, "lbl_db_info"):
                self.lbl_db_info.setText(route_labels.get(route, route))
                self.lbl_db_info.setToolTip(tooltip)
            if hasattr(self, "lbl_reason"):
                self.lbl_reason.setToolTip(tooltip)
            if hasattr(self, "lbl_verdict"):
                self.lbl_verdict.setToolTip(tooltip)

        if hasattr(self, "lbl_db_info") and not route:
            self.lbl_db_info.setText(memory_summary_text(detail))
            if model["hard_missing_override"]:
                self.lbl_db_info.setStyleSheet(
                    "color: #ff7b72; font-size: 12px; font-weight: 800; "
                    "border: none; background: transparent;"
                )
            elif model["conflict"]:
                self.lbl_db_info.setStyleSheet(
                    "color: #ffb454; font-size: 12px; font-weight: 800; "
                    "border: none; background: transparent;"
                )
            elif model["leading_hypothesis"] == "NG" and model["has_memory"]:
                self.lbl_db_info.setStyleSheet(
                    "color: #ff7b72; font-size: 12px; font-weight: 700; "
                    "border: none; background: transparent;"
                )
            elif model["leading_hypothesis"] == "OK" and model["has_memory"]:
                self.lbl_db_info.setStyleSheet(
                    "color: #3fb950; font-size: 12px; font-weight: 700; "
                    "border: none; background: transparent;"
                )
            else:
                self.lbl_db_info.setStyleSheet(
                    "color: #8b949e; font-size: 12px; font-weight: 600; "
                    "border: none; background: transparent;"
                )

        # Os textos preexistentes da memória aproximada não devem apagar a
        # distinção explícita NOVO/CONHECIDO quando o roteador estiver ativo.
        if route and hasattr(self, "lbl_db_info"):
            if route == "KNOWN_KNN":
                self.lbl_db_info.setStyleSheet(
                    "color: #3fb950; font-size: 12px; font-weight: 800;"
                    " border: none; background: transparent;"
                )
            elif route == "MEMORY_CONFLICT":
                self.lbl_db_info.setStyleSheet(
                    "color: #ff7b72; font-size: 12px; font-weight: 800;"
                    " border: none; background: transparent;"
                )
            else:
                self.lbl_db_info.setStyleSheet(
                    "color: #ffd33d; font-size: 12px; font-weight: 800;"
                    " border: none; background: transparent;"
                )

        # Destaque exclusivamente visual. O analysis original permanece intacto.
        if model["conflict"] and hasattr(self, "lbl_verdict"):
            self.lbl_verdict.setText("CONFLITO DE MEMÓRIA • REVISÃO OBRIGATÓRIA")
            self.lbl_verdict.setStyleSheet(
                "color: #ffb454; font-size: 16px; font-weight: 800; border: none;"
            )

    control_panel_cls._update_confidence_panel = update_confidence_panel
    control_panel_cls._memory_status_ui_installed = True


__all__ = ["install_memory_status_ui"]
