"""Visão mult-iluminação da categoria de adesivo.

Esta camada é exclusivamente de aquisição/visualização. Ela não altera o motor
FLUXO DE ADESIVO, score, memória, KNN ou decisão final. A primeira inspeção
SIDE continua passando pelo pipeline normal; frames posteriores da mesma peça
podem preencher TOP/MID sem substituir a análise ativa.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.core.epicenter_extractor import EpicenterExtractor
from src.core.inspection import detect_anomalies
from src.utils.text_normalizer import normalize_aoi_text


ADHESIVE_CATEGORY = "MUITO ADESIVO"
LIGHTING_ORDER = ("SIDE", "TOP", "MID")
LIGHTING_SYMBOL = {
    "SIDE": "↓",
    "TOP": "←",
    "MID": "→",
}
WIDE_COLUMNS_BREAKPOINT = 1000


def category_from_aoi_info(aoi_info: dict | None) -> str:
    """Retorna a categoria canônica sem depender de o controller já ter normalizado."""
    info = aoi_info if isinstance(aoi_info, dict) else {}
    candidates = (
        info.get("category", ""),
        info.get("value", ""),
    )
    for candidate in candidates:
        normalized, _value = normalize_aoi_text(str(candidate or ""))
        if normalized != "Unknown":
            return normalized
    return str(info.get("category", "") or "").strip().upper()


def is_adhesive_category(aoi_info: dict | None) -> bool:
    return category_from_aoi_info(aoi_info) == ADHESIVE_CATEGORY


def _valid_image(value: Any) -> bool:
    return bool(
        isinstance(value, np.ndarray)
        and value.size > 0
        and value.ndim >= 2
    )


def _crop_box(image: np.ndarray, box: Any) -> np.ndarray:
    if not _valid_image(image) or box is None:
        return np.array([])

    if isinstance(box, dict):
        if box.get("detected") is False:
            return np.array([])
        x = int(box.get("x", 0) or 0)
        y = int(box.get("y", 0) or 0)
        w = int(box.get("w", 0) or 0)
        h = int(box.get("h", 0) or 0)
    else:
        try:
            x, y, w, h = [int(value) for value in box[:4]]
        except Exception:
            return np.array([])

    height, width = image.shape[:2]
    x1 = max(0, min(width, x))
    y1 = max(0, min(height, y))
    x2 = max(x1, min(width, x + max(0, w)))
    y2 = max(y1, min(height, y + max(0, h)))
    if x2 <= x1 or y2 <= y1:
        return np.array([])
    return image[y1:y2, x1:x2].copy()


def build_adhesive_view_payload(
    sample_crop: np.ndarray,
    ng_crop: np.ndarray,
) -> dict:
    """Monta as três imagens visuais de uma iluminação.

    main:
        imagem TESTE recebida da AOI.
    large:
        recorte da maior caixa verde detectada.
    small:
        recorte do epicentro/menor caixa verde selecionada pelo radar.
    """
    payload = {
        "main": ng_crop.copy() if _valid_image(ng_crop) else np.array([]),
        "large": np.array([]),
        "small": np.array([]),
        "large_box": None,
        "small_box": None,
    }
    if not _valid_image(sample_crop) or not _valid_image(ng_crop):
        return payload

    try:
        (
            _raw_anomalies,
            old_epicenters,
            global_box_info,
            _gab_focus,
            _test_focus,
        ) = detect_anomalies(sample_crop, ng_crop)

        real_epicenters, _focus_gab, focus_ng = EpicenterExtractor.extract_focus(
            sample_crop,
            ng_crop,
            old_epicenters,
            global_box_info,
        )

        large = _crop_box(ng_crop, global_box_info)
        if _valid_image(large):
            payload["large"] = large
            payload["large_box"] = dict(global_box_info or {})

        if _valid_image(focus_ng):
            payload["small"] = focus_ng.copy()
            if real_epicenters:
                payload["small_box"] = tuple(
                    int(value) for value in real_epicenters[0]
                )
    except Exception as exc:
        # O recurso é visual. Uma falha de preview nunca pode interromper
        # a inspeção produtiva que continua no pipeline original.
        print(f"Falha não fatal ao montar preview multilight de adesivo: {exc}")

    return payload


class _ResponsiveImageViewport(QLabel):
    """QLabel que preserva a imagem original e recalcula escala no resize."""

    def __init__(self, placeholder: str, focused: bool = False):
        super().__init__(placeholder)
        self._source_pixmap = QPixmap()
        self._placeholder = str(placeholder)
        self.setObjectName("focusViewport" if focused else "imageViewport")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumWidth(0)
        self.setMinimumHeight(92 if focused else 150)

    def set_bgr_image(self, image: np.ndarray, placeholder: str | None = None) -> bool:
        if not _valid_image(image):
            self.clear_view(placeholder or self._placeholder)
            return False

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimage = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        if pixmap.isNull():
            self.clear_view(placeholder or self._placeholder)
            return False

        self._source_pixmap = pixmap
        self._render()
        return True

    def clear_view(self, placeholder: str | None = None) -> None:
        self._source_pixmap = QPixmap()
        self.clear()
        self.setText(str(placeholder or self._placeholder))

    def _render(self) -> None:
        if self._source_pixmap.isNull():
            return
        target = self.contentsRect().size()
        if target.width() <= 1 or target.height() <= 1:
            return
        scaled = self._source_pixmap.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()


class _LightingInspectionCard(QFrame):
    """Uma iluminação com imagem completa + caixa maior + caixa menor."""

    def __init__(self, mode: str):
        super().__init__()
        self.mode = str(mode).upper()
        self.setObjectName("imageCard")
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)

        title = QLabel(
            f"{self.mode}  {LIGHTING_SYMBOL.get(self.mode, '')}"
        )
        title.setObjectName("eyebrowLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        main_title = QLabel("IMAGEM RECEBIDA")
        main_title.setObjectName("imageTitle")
        self.main_view = _ResponsiveImageViewport(
            f"Aguardando {self.mode}"
        )
        root.addWidget(main_title)
        root.addWidget(self.main_view, stretch=3)

        crop_grid = QGridLayout()
        crop_grid.setContentsMargins(0, 0, 0, 0)
        crop_grid.setHorizontalSpacing(7)
        crop_grid.setVerticalSpacing(4)

        large_title = QLabel("RETÂNGULO MAIOR")
        large_title.setObjectName("imageTitle")
        small_title = QLabel("RETÂNGULO MENOR")
        small_title.setObjectName("imageTitle")
        self.large_view = _ResponsiveImageViewport(
            "Aguardando caixa maior",
            focused=True,
        )
        self.small_view = _ResponsiveImageViewport(
            "Aguardando caixa menor",
            focused=True,
        )

        crop_grid.addWidget(large_title, 0, 0)
        crop_grid.addWidget(small_title, 0, 1)
        crop_grid.addWidget(self.large_view, 1, 0)
        crop_grid.addWidget(self.small_view, 1, 1)
        crop_grid.setColumnStretch(0, 1)
        crop_grid.setColumnStretch(1, 1)
        root.addLayout(crop_grid, stretch=2)

    def set_payload(self, payload: dict) -> None:
        data = payload if isinstance(payload, dict) else {}
        self.main_view.set_bgr_image(
            data.get("main"),
            f"Aguardando {self.mode}",
        )
        self.large_view.set_bgr_image(
            data.get("large"),
            "Caixa maior não detectada",
        )
        self.small_view.set_bgr_image(
            data.get("small"),
            "Caixa menor não detectada",
        )

    def clear_payload(self) -> None:
        self.main_view.clear_view(f"Aguardando {self.mode}")
        self.large_view.clear_view("Aguardando caixa maior")
        self.small_view.clear_view("Aguardando caixa menor")


class AdhesiveMultiLightView(QWidget):
    """Layout responsivo das nove imagens da inspeção de adesivo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("adhesiveMultiLightView")
        self.setMinimumWidth(0)
        self._columns = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(7)

        hint = QLabel(
            "ADESIVO • SIDE / TOP / MID • imagem recebida + caixas maior e menor"
        )
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(8)

        self.cards = {
            mode: _LightingInspectionCard(mode)
            for mode in LIGHTING_ORDER
        }
        root.addLayout(self.grid)
        self._reflow(WIDE_COLUMNS_BREAKPOINT + 1)

    @staticmethod
    def columns_for_width(width: int) -> int:
        return 3 if int(width) >= WIDE_COLUMNS_BREAKPOINT else 1

    def _reflow(self, width: int) -> None:
        columns = self.columns_for_width(width)
        if columns == self._columns and self.grid.count() == len(self.cards):
            return

        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self)

        for index, mode in enumerate(LIGHTING_ORDER):
            row = index // columns
            column = index % columns
            self.grid.addWidget(self.cards[mode], row, column)

        for column in range(3):
            self.grid.setColumnStretch(column, 1 if column < columns else 0)
        self._columns = columns

    def set_lighting_payload(self, mode: str, payload: dict) -> bool:
        normalized = str(mode or "").strip().upper()
        card = self.cards.get(normalized)
        if card is None:
            return False
        card.set_payload(payload)
        return True

    def clear_all(self) -> None:
        for card in self.cards.values():
            card.clear_payload()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow(event.size().width())


def _set_receiver_auxiliary_mode(panel, enabled: bool) -> None:
    receiver = getattr(panel, "network_receiver", None)
    method = getattr(receiver, "set_auxiliary_image_mode", None)
    if callable(method):
        try:
            method(bool(enabled))
        except Exception as exc:
            print(f"Falha não fatal ao alternar recepção multilight: {exc}")


def _switch_inspection_view(panel, adhesive: bool) -> None:
    builder = getattr(panel, "ui_builder", None)
    method = getattr(builder, "set_adhesive_inspection_mode", None)
    if callable(method):
        method(panel, bool(adhesive))


def _reset_session(panel, *, show_normal: bool = True) -> None:
    panel.adhesive_multilight_active = False
    panel.adhesive_multilight_aux_mode = None
    panel.adhesive_multilight_views = {}
    panel.adhesive_multilight_primary_event_id = None
    _set_receiver_auxiliary_mode(panel, False)

    view = getattr(panel, "adhesive_multilight_view", None)
    if view is not None and hasattr(view, "clear_all"):
        view.clear_all()
    if show_normal:
        _switch_inspection_view(panel, False)


def _store_view(panel, mode: str, sample_crop, ng_crop) -> bool:
    normalized = str(mode or "").strip().upper()
    if normalized not in LIGHTING_ORDER:
        return False

    payload = build_adhesive_view_payload(sample_crop, ng_crop)
    panel.adhesive_multilight_views[normalized] = payload

    view = getattr(panel, "adhesive_multilight_view", None)
    if view is not None and hasattr(view, "set_lighting_payload"):
        view.set_lighting_payload(normalized, payload)
    return True


def _current_lighting(panel) -> str:
    label = getattr(panel, "lbl_light_value", None)
    try:
        value = str(label.text()).strip().upper()
    except Exception:
        value = ""
    return value if value in LIGHTING_ORDER else "SIDE"


def install_adhesive_multilight_inspection(control_panel_cls) -> None:
    """Liga o layout às imagens recebidas sem alterar o julgamento atual."""
    if getattr(
        control_panel_cls,
        "_adhesive_multilight_inspection_installed",
        False,
    ):
        return

    original_init = control_panel_cls.__init__
    original_handle_network_image = control_panel_cls.handle_network_image
    original_process_aoi_images = control_panel_cls.process_aoi_images
    original_save_label = control_panel_cls.save_label
    original_skip_image = control_panel_cls.skip_image

    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.adhesive_multilight_active = False
        self.adhesive_multilight_aux_mode = None
        self.adhesive_multilight_views = {}
        self.adhesive_multilight_primary_event_id = None
        _switch_inspection_view(self, False)

    def handle_network_image(self, img_bgr, ip: str):
        # Enquanto uma peça de adesivo permanece ativa, frames adicionais são
        # previews auxiliares da mesma peça. Eles não passam pelo gate normal,
        # não recebem novo event_id e não substituem current_analysis.
        if (
            bool(getattr(self, "adhesive_multilight_active", False))
            and getattr(self, "current_analysis", None) is not None
            and is_adhesive_category(
                getattr(self, "current_aoi_info", None)
            )
        ):
            mode = _current_lighting(self)
            self.last_xp_ip = str(ip)
            self.adhesive_multilight_aux_mode = mode
            try:
                self.processor_monitor.process_external_image(img_bgr)
                return True
            finally:
                # process_external_image emite layout_detected de forma direta
                # neste caminho. Se falhar antes do sinal, não deixamos estado
                # auxiliar pendurado para a próxima peça.
                self.adhesive_multilight_aux_mode = None

        return original_handle_network_image(self, img_bgr, ip)

    def process_aoi_images(self, sample_crop, ng_crop, aoi_info):
        aux_mode = getattr(self, "adhesive_multilight_aux_mode", None)
        if aux_mode in LIGHTING_ORDER and bool(
            getattr(self, "adhesive_multilight_active", False)
        ):
            stored = _store_view(
                self,
                aux_mode,
                sample_crop,
                ng_crop,
            )
            if stored:
                _switch_inspection_view(self, True)
                try:
                    self.update_brain_status(
                        f"Imagem {aux_mode} recebida para visão multilight de adesivo.",
                        False,
                    )
                except Exception:
                    pass
            return stored

        adhesive = is_adhesive_category(aoi_info)
        if not adhesive:
            _reset_session(self, show_normal=True)
            return original_process_aoi_images(
                self,
                sample_crop,
                ng_crop,
                aoi_info,
            )

        # A primeira imagem SIDE continua usando o pipeline atual integralmente.
        # Só depois do resultado pronto ativamos a visualização auxiliar.
        result = original_process_aoi_images(
            self,
            sample_crop,
            ng_crop,
            aoi_info,
        )

        if (
            getattr(self, "current_analysis", None) is None
            or not _valid_image(getattr(self, "current_ng", None))
        ):
            return result

        self.adhesive_multilight_active = True
        self.adhesive_multilight_aux_mode = None
        self.adhesive_multilight_views = {}
        self.adhesive_multilight_primary_event_id = getattr(
            self,
            "network_intake_last_image_event_id",
            None,
        )

        # Contrato da AOI: a iluminação padrão/primeira recebida é SIDE.
        _store_view(self, "SIDE", sample_crop, ng_crop)
        _switch_inspection_view(self, True)
        _set_receiver_auxiliary_mode(self, True)
        return result

    def save_label(self, *args, **kwargs):
        # Fecha primeiro a passagem de previews para não haver corrida com a
        # liberação do gate da próxima peça.
        _set_receiver_auxiliary_mode(self, False)
        result = original_save_label(self, *args, **kwargs)
        if not bool(getattr(self, "is_locked", False)):
            _reset_session(self, show_normal=True)
        return result

    def skip_image(self, *args, **kwargs):
        _set_receiver_auxiliary_mode(self, False)
        result = original_skip_image(self, *args, **kwargs)
        if not bool(getattr(self, "is_locked", False)):
            _reset_session(self, show_normal=True)
        return result

    control_panel_cls.__init__ = wrapped_init
    control_panel_cls.handle_network_image = handle_network_image
    control_panel_cls.process_aoi_images = process_aoi_images
    control_panel_cls.save_label = save_label
    control_panel_cls.skip_image = skip_image
    control_panel_cls._adhesive_multilight_inspection_installed = True


__all__ = [
    "ADHESIVE_CATEGORY",
    "LIGHTING_ORDER",
    "WIDE_COLUMNS_BREAKPOINT",
    "AdhesiveMultiLightView",
    "build_adhesive_view_payload",
    "category_from_aoi_info",
    "install_adhesive_multilight_inspection",
    "is_adhesive_category",
]
