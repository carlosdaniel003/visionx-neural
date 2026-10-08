"""Resumo visual das novas camadas de memória no painel principal."""

from __future__ import annotations

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
                ]
            elif route == "NEW_EXPERTS":
                tooltip_lines += [
                    "", "Memória KNN ignorada nesta decisão.",
                    "Motores da categoria consultados como primeira ocorrência.",
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
