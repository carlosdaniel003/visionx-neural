"""Contrato leve do ponteiro da CNN FALTANDO v2.

Não importa PyTorch. O candidato de 97,70% teve **19 falsos NG** no replay
histórico e só pode ser usado em ensaio assistido, com operador obrigatório.
"""

POINTER_SCHEMA = "visionx.specialist_active_pointer.v1"

TRIAL_97P70_MODE = "97p70_unqualified_operator_review_only"
TRIAL_97P70_SHA256 = (
    "c13ca3296eee5e7741c58f29a6ef035296c0fe84a5651a9481462d72042b120c"
)
TRIAL_97P70_CHECKPOINT = (
    "reports/neural_online/checkpoints/"
    "faltando_v2_97p70_operator_trial.pt"
)

__all__ = [
    "POINTER_SCHEMA", "TRIAL_97P70_MODE", "TRIAL_97P70_SHA256",
    "TRIAL_97P70_CHECKPOINT",
]
