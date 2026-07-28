"""Compatibilidade de importação para o especialista de fluxo de adesivo."""

from src.core.experts.adhesive_shift_expert import ShiftExpert as _AdhesiveShiftExpert


class ShiftExpert(_AdhesiveShiftExpert):
    """Publica aliases exclusivos e restringe o motor à categoria correta."""

    ADHESIVE_CATEGORIES = frozenset({"MUCH ADHESIVE", "MUITO ADESIVO"})

    @classmethod
    def _is_adhesive_context(cls, info: dict | None) -> bool:
        if not isinstance(info, dict):
            return False
        category = " ".join(
            str(info.get("category", "")).strip().upper().split()
        )
        return category in cls.ADHESIVE_CATEGORIES

    def analyze(self, *args, **kwargs):
        result = super().analyze(*args, **kwargs)
        result.update(
            {
                "adhesive_comparison_mode": result.get(
                    "comparison_mode",
                    self.MODE,
                ),
                "adhesive_roi_width": result.get("roi_width", 0),
                "adhesive_roi_height": result.get("roi_height", 0),
                "adhesive_reference_centroid": result.get("reference_centroid"),
                "adhesive_test_centroid": result.get("test_centroid"),
                "adhesive_reference_mask": result.get("reference_mask"),
                "adhesive_test_mask": result.get("test_mask"),
                "adhesive_excess_mask": result.get("excess_mask"),
                "adhesive_padding_overlap_mask": result.get(
                    "padding_overlap_mask"
                ),
                "adhesive_reference_view": result.get("reference_view"),
                "adhesive_test_view": result.get("test_view"),
                "adhesive_flow_view": result.get("flow_view"),
            }
        )
        return result


__all__ = ["ShiftExpert"]
