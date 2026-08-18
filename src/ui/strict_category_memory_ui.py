"""Telemetria visual da memória KNN isolada por categoria."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter


CATEGORY_LABELS = {
    "INVERTIDO": "INVERTIDO",
    "FALTANDO": "FALTANDO",
    "MUITOADESIVO": "MUITO ADESIVO",
}


def install_strict_category_memory_ui(widget_cls) -> None:
    if getattr(widget_cls, "_strict_category_ui_installed", False):
        return

    original_update_data = widget_cls.update_data
    original_mode_label = widget_cls._mode_label
    original_paint_event = widget_cls.paintEvent

    def update_data(self, detail):
        original_update_data(self, detail)
        raw_category = str((detail or {}).get("memory_category", ""))
        self.memory_category = raw_category.upper()
        self.memory_filter_strict = bool(
            (detail or {}).get("memory_filter_strict", False)
        )
        self.memory_candidate_count = int(
            (detail or {}).get("memory_candidate_count", 0) or 0
        )
        self.memory_reason = str((detail or {}).get("memory_reason", ""))

    def mode_label(self):
        base = original_mode_label(self)
        category = CATEGORY_LABELS.get(
            getattr(self, "memory_category", ""),
            getattr(self, "memory_category", ""),
        )
        return f"{base} • {category}" if category else base

    def paint_event(self, event):
        category = CATEGORY_LABELS.get(
            getattr(self, "memory_category", ""),
            getattr(self, "memory_category", ""),
        )
        # Só substitui o painel quando realmente não existe memória da categoria.
        # Conflito e correspondência insuficiente possuem memória disponível e
        # precisam continuar visíveis no painel hierárquico novo.
        if (
            getattr(self, "is_active", False)
            and not getattr(self, "has_memory", False)
            and not getattr(self, "memory_available", False)
            and not getattr(self, "memory_conflict", False)
            and category
        ):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(0, 0, self.width(), self.height(), QColor("#101010"))
            painter.setPen(QColor("#f5c518"))
            painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                f"Sem JSON de anomalia para {category}",
            )
            painter.end()
            return
        return original_paint_event(self, event)

    widget_cls.update_data = update_data
    widget_cls._mode_label = mode_label
    widget_cls.paintEvent = paint_event
    widget_cls._strict_category_ui_installed = True


__all__ = ["install_strict_category_memory_ui"]
