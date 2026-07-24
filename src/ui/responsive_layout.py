"""Perfis de layout responsivo para a interface desktop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutProfile:
    name: str
    outer_margin: int
    section_spacing: int
    info_columns: int
    footer_columns: int
    action_columns: int
    splitter_vertical: bool
    image_min_height: int
    debugger_min_width: int
    debugger_max_width: int


def profile_for_width(width: int) -> LayoutProfile:
    """Retorna o perfil visual apropriado para a largura disponível."""
    if width < 1100:
        return LayoutProfile(
            name="compact",
            outer_margin=8,
            section_spacing=8,
            info_columns=2,
            footer_columns=1,
            action_columns=2,
            splitter_vertical=True,
            image_min_height=105,
            debugger_min_width=340,
            debugger_max_width=460,
        )

    if width < 1600:
        return LayoutProfile(
            name="standard",
            outer_margin=10,
            section_spacing=10,
            info_columns=3,
            footer_columns=2,
            action_columns=4,
            splitter_vertical=False,
            image_min_height=120,
            debugger_min_width=420,
            debugger_max_width=540,
        )

    return LayoutProfile(
        name="wide",
        outer_margin=14,
        section_spacing=12,
        info_columns=5,
        footer_columns=3,
        action_columns=4,
        splitter_vertical=False,
        image_min_height=145,
        debugger_min_width=500,
        debugger_max_width=620,
    )
