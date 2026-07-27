"""Modelo puro dos estados dos controles operacionais."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationalState:
    name: str
    badge: str
    hint: str
    tone: str
    capture_enabled: bool
    discard_enabled: bool
    approve_enabled: bool
    reject_enabled: bool
    lighting_enabled: bool
    dataset_clear_enabled: bool


def operational_state(
    *,
    mode: str,
    is_locked: bool,
    has_analysis: bool,
) -> OperationalState:
    """Resolve permissões e mensagem visual sem depender do PyQt."""
    normalized_mode = str(mode or "").strip()

    if is_locked and not has_analysis:
        return OperationalState(
            name="processing",
            badge="PROCESSANDO",
            hint="Aguarde a captura e a análise dos especialistas.",
            tone="busy",
            capture_enabled=False,
            discard_enabled=False,
            approve_enabled=False,
            reject_enabled=False,
            lighting_enabled=False,
            dataset_clear_enabled=False,
        )

    if is_locked and has_analysis and normalized_mode == "Modo Teste":
        return OperationalState(
            name="review_test",
            badge="DECISÃO PENDENTE",
            hint="Revise os diagnósticos e escolha OK, NG ou descarte a captura.",
            tone="attention",
            capture_enabled=True,
            discard_enabled=True,
            approve_enabled=True,
            reject_enabled=True,
            lighting_enabled=True,
            dataset_clear_enabled=False,
        )

    if is_locked and has_analysis and normalized_mode == "Modo Sombra":
        return OperationalState(
            name="review_shadow",
            badge="AGUARDANDO AOI",
            hint="A decisão deve ser enviada pelo teclado físico da AOI.",
            tone="attention",
            capture_enabled=True,
            discard_enabled=True,
            approve_enabled=False,
            reject_enabled=False,
            lighting_enabled=True,
            dataset_clear_enabled=False,
        )

    if is_locked and has_analysis and normalized_mode == "Modo Produção":
        return OperationalState(
            name="production_auto",
            badge="EMISSÃO AUTOMÁTICA",
            hint="A decisão está sendo enviada automaticamente à estação.",
            tone="busy",
            capture_enabled=False,
            discard_enabled=False,
            approve_enabled=False,
            reject_enabled=False,
            lighting_enabled=False,
            dataset_clear_enabled=False,
        )

    return OperationalState(
        name="idle",
        badge="PRONTO",
        hint="Selecione a iluminação ou inicie uma nova captura.",
        tone="ready",
        capture_enabled=True,
        discard_enabled=False,
        approve_enabled=False,
        reject_enabled=False,
        lighting_enabled=True,
        dataset_clear_enabled=normalized_mode == "Modo Teste",
    )


def available_action_count(state: OperationalState) -> int:
    """Conta somente ações operacionais atualmente executáveis."""
    return (
        int(state.capture_enabled)
        + int(state.discard_enabled)
        + int(state.approve_enabled)
        + int(state.reject_enabled)
        + (3 if state.lighting_enabled else 0)
        + int(state.dataset_clear_enabled)
    )
