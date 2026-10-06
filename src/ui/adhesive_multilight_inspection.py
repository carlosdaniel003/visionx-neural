"""Aquisição, visualização, análise e fusão multilight da categoria de adesivo.

SIDE, TOP e MID pertencem à mesma peça. Cada iluminação mantém sua análise
independente para auditoria, e o julgamento final só é promovido depois que as
três análises estão disponíveis.
"""

from __future__ import annotations

from functools import wraps
import time
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import QEventLoop, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from src.core.adhesive_multilight_analysis import (
    analyze_lighting,
    build_lighting_context,
    valid_image,
)
from src.core.adhesive_multilight_fusion import (
    fuse_adhesive_multilight,
)
from src.services.capture_debug_payload import decision_record
from src.services.capture_evidence import (
    capture_debug_record,
    update_capture_debug_record,
)
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
    return valid_image(value)


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
    context: dict | None = None,
) -> dict:
    """Monta as três imagens visuais de uma iluminação.

    O contexto geométrico pode ser compartilhado com a análise dos especialistas
    para evitar repetir detect_anomalies/EpicenterExtractor no mesmo frame.
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
        frame_context = (
            context
            if isinstance(context, dict)
            else build_lighting_context(sample_crop, ng_crop)
        )
        if not frame_context.get("valid", False):
            return payload

        global_box_info = frame_context.get("global_box_info", {})
        real_epicenters = frame_context.get("real_epicenters", [])
        focus_ng = frame_context.get("focus_ng", np.array([]))

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


def _sync_multilight_debug_controls(panel) -> None:
    """Atualiza Copiar debug/Copiar imagem sem criar dependência de import cíclica."""
    try:
        from src.ui.network_xp_debug import sync_network_debug_controls

        sync_network_debug_controls(panel)
    except Exception as exc:
        print(f"Falha não fatal ao sincronizar debug multilight: {exc}")


def _finalize_multilight_decision(panel) -> dict | None:
    """Promove SIDE/TOP/MID a um único julgamento final de adesivo."""
    analyses = getattr(panel, "adhesive_multilight_analyses", {})
    fused = fuse_adhesive_multilight(analyses)
    if not isinstance(fused, dict):
        try:
            panel.update_network_status(
                "Fusão multilight não executada: SIDE/TOP/MID incompletos."
            )
        except Exception:
            pass
        return None

    panel.current_analysis = fused
    panel.adhesive_multilight_final_analysis = fused
    panel.adhesive_multilight_last_final_analysis = fused

    # O painel principal e os overlays recebem somente a decisão já fundida.
    try:
        panel._update_confidence_panel(fused)
        panel._update_reference_panel(fused)
    except Exception as exc:
        print(f"Falha não fatal ao exibir fusão multilight: {exc}")

    QApplication.processEvents(
        QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
    )
    completed_at = time.perf_counter()
    started_at = float(
        getattr(panel, "capture_start_time", 0.0) or 0.0
    )
    elapsed = max(0.0, completed_at - started_at) if started_at > 0.0 else 0.0
    panel.last_analysis_time_seconds = elapsed

    detail = fused.setdefault("detail", {})
    detail["analysis_time_seconds"] = float(elapsed)
    detail["analysis_time_start_source"] = str(
        getattr(panel, "capture_start_source", "") or ""
    )
    detail["analysis_time_contract"] = (
        "primeira imagem SIDE recebida/capturada -> "
        "SIDE+TOP+MID analisadas e resultado multilight pintado"
    )

    try:
        panel.lbl_timer.setText(f"{elapsed:.2f} s")
        panel.lbl_timer.setToolTip(
            "Tempo real da categoria adesivo: do recebimento da SIDE até "
            "a conclusão das análises SIDE/TOP/MID e da fusão final."
        )
    except Exception:
        pass

    # O debug principal deixa de apontar para a decisão provisória SIDE.
    # Mantemos o mesmo event_id e apenas promovemos a decisão final já fundida.
    try:
        debug_record = capture_debug_record(panel)
        if debug_record:
            debug_record["decision"] = decision_record(
                fused,
                getattr(panel, "current_aoi_info", None),
            )
            update_capture_debug_record(panel, debug_record)
    except Exception as exc:
        print(f"Falha não fatal ao atualizar debug da fusão multilight: {exc}")

    mode = ""
    try:
        mode = str(panel.combo_mode.currentText() or "")
    except Exception:
        pass

    try:
        if mode == "Modo Produção":
            if fused.get("production_review_required", False):
                panel.update_brain_status(
                    "REVISÃO OBRIGATÓRIA • fusão multilight concluída.",
                    True,
                )
            else:
                panel.update_brain_status(
                    "Análise multilight concluída • emissão autônoma.",
                    False,
                )
        elif mode == "Modo Sombra":
            panel.update_brain_status(
                "Fusão multilight concluída • aguardando decisão humana no XP.",
                True,
            )
            panel.btn_start.setEnabled(True)
            panel.btn_skip.setEnabled(True)
        else:
            panel.update_brain_status(
                "Fusão multilight concluída • aguardando operador.",
                True,
            )
            panel.btn_start.setEnabled(True)
            panel.btn_save_ok.setEnabled(True)
            panel.btn_save_ng.setEnabled(True)
            panel.btn_skip.setEnabled(True)
    except Exception:
        pass

    _sync_multilight_debug_controls(panel)
    return fused


def _reset_session(panel, *, show_normal: bool = True) -> None:
    automation = getattr(
        panel,
        "adhesive_multilight_automation",
        None,
    )
    cancel_cycle = getattr(automation, "cancel_for_cycle_end", None)
    if callable(cancel_cycle):
        cancel_cycle()

    panel.adhesive_multilight_active = False
    panel.adhesive_multilight_aux_mode = None
    panel.adhesive_multilight_views = {}
    panel.adhesive_multilight_analyses = {}
    panel.adhesive_multilight_source_frames = {}
    panel.adhesive_multilight_pending_source_frame = None
    panel.adhesive_multilight_pending_source_mode = ""
    panel.adhesive_multilight_primary_event_id = None
    panel.adhesive_multilight_pending_start = False
    panel.adhesive_multilight_deferred_auto_decision = ""
    _set_receiver_auxiliary_mode(panel, False)

    view = getattr(panel, "adhesive_multilight_view", None)
    if view is not None and hasattr(view, "clear_all"):
        view.clear_all()

    analysis_view = getattr(
        panel,
        "adhesive_multilight_analysis_view",
        None,
    )
    if analysis_view is not None and hasattr(analysis_view, "clear_all"):
        analysis_view.clear_all()

    if show_normal:
        _switch_inspection_view(panel, False)


def _store_view(
    panel,
    mode: str,
    sample_crop,
    ng_crop,
    context: dict | None = None,
) -> bool:
    normalized = str(mode or "").strip().upper()
    if normalized not in LIGHTING_ORDER:
        return False

    payload = build_adhesive_view_payload(
        sample_crop,
        ng_crop,
        context=context,
    )
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
        self.adhesive_multilight_analyses = {}
        self.adhesive_multilight_source_frames = {}
        self.adhesive_multilight_pending_source_frame = None
        self.adhesive_multilight_pending_source_mode = ""
        self.adhesive_multilight_primary_event_id = None
        self.adhesive_multilight_pending_start = False
        self.adhesive_multilight_deferred_auto_decision = ""
        self.adhesive_multilight_final_analysis = None

        # Último conjunto multilight completo/parcial permanece disponível para
        # Copiar debug/Copiar imagem mesmo depois que o ciclo produtivo termina.
        self.adhesive_multilight_last_analyses = {}
        self.adhesive_multilight_last_source_frames = {}
        self.adhesive_multilight_last_event_id = ""
        self.adhesive_multilight_last_category = ""
        self.adhesive_multilight_last_final_analysis = None
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
            automation = getattr(
                self,
                "adhesive_multilight_automation",
                None,
            )
            expected_mode = ""
            expected_getter = getattr(
                automation,
                "expected_frame_mode",
                None,
            )
            if callable(expected_getter):
                expected_mode = str(
                    expected_getter() or ""
                ).strip().upper()

            mode = (
                expected_mode
                if expected_mode in LIGHTING_ORDER
                else _current_lighting(self)
            )
            self.last_xp_ip = str(ip)
            self.adhesive_multilight_aux_mode = mode
            self.adhesive_multilight_pending_source_mode = mode
            self.adhesive_multilight_pending_source_frame = (
                img_bgr.copy()
                if _valid_image(img_bgr)
                else None
            )
            try:
                self.processor_monitor.process_external_image(img_bgr)
                return True
            finally:
                # process_external_image emite layout_detected de forma direta
                # neste caminho. Se falhar antes do sinal, não deixamos estado
                # auxiliar pendurado para a próxima peça.
                self.adhesive_multilight_aux_mode = None
                self.adhesive_multilight_pending_source_mode = ""
                self.adhesive_multilight_pending_source_frame = None

        return original_handle_network_image(self, img_bgr, ip)

    def process_aoi_images(self, sample_crop, ng_crop, aoi_info):
        aux_mode = getattr(self, "adhesive_multilight_aux_mode", None)
        if aux_mode in LIGHTING_ORDER and bool(
            getattr(self, "adhesive_multilight_active", False)
        ):
            try:
                frame_context = build_lighting_context(
                    sample_crop,
                    ng_crop,
                )
                if not frame_context.get("valid", False):
                    return False

                stored = _store_view(
                    self,
                    aux_mode,
                    sample_crop,
                    ng_crop,
                    context=frame_context,
                )
                if not stored:
                    return False

                # TOP/MID passam pelo mesmo conjunto de especialistas usado por
                # SIDE, mas o resultado fica isolado por iluminação e não toca
                # current_analysis nem o veredito final da peça.
                lighting_analysis = analyze_lighting(
                    getattr(self, "orchestrator", None),
                    sample_crop,
                    ng_crop,
                    getattr(self, "current_aoi_info", None),
                    aux_mode,
                    context=frame_context,
                )
                if not isinstance(lighting_analysis, dict):
                    try:
                        self.update_network_status(
                            f"Falha na análise visual {aux_mode}; "
                            "aguardando nova captura da mesma iluminação."
                        )
                    except Exception:
                        pass
                    return False

                self.adhesive_multilight_analyses[aux_mode] = (
                    lighting_analysis
                )

                pending_source_mode = str(
                    getattr(
                        self,
                        "adhesive_multilight_pending_source_mode",
                        "",
                    )
                    or ""
                ).strip().upper()
                pending_source_frame = getattr(
                    self,
                    "adhesive_multilight_pending_source_frame",
                    None,
                )
                if (
                    pending_source_mode == aux_mode
                    and _valid_image(pending_source_frame)
                ):
                    source_copy = pending_source_frame.copy()
                    self.adhesive_multilight_source_frames[aux_mode] = (
                        source_copy
                    )
                    self.adhesive_multilight_last_source_frames[aux_mode] = (
                        source_copy.copy()
                    )

                self.adhesive_multilight_last_analyses[aux_mode] = (
                    lighting_analysis
                )
                self.adhesive_multilight_last_event_id = str(
                    getattr(
                        self,
                        "adhesive_multilight_primary_event_id",
                        "",
                    )
                    or ""
                )
                self.adhesive_multilight_last_category = ADHESIVE_CATEGORY

                analysis_view = getattr(
                    self,
                    "adhesive_multilight_analysis_view",
                    None,
                )
                if analysis_view is not None:
                    analysis_view.set_analysis(
                        aux_mode,
                        lighting_analysis,
                    )

                _switch_inspection_view(self, True)

                # A automação só avança depois que a imagem e sua análise
                # visual foram concluídas para a iluminação esperada.
                automation = getattr(
                    self,
                    "adhesive_multilight_automation",
                    None,
                )
                frame_stored = getattr(
                    automation,
                    "frame_stored",
                    None,
                )
                if callable(frame_stored):
                    frame_stored(aux_mode)

                _sync_multilight_debug_controls(self)

                try:
                    self.update_brain_status(
                        f"Imagem e análise {aux_mode} concluídas.",
                        False,
                    )
                except Exception:
                    pass
                return True
            except Exception as exc:
                try:
                    self.update_network_status(
                        f"Falha na análise multilight {aux_mode}: {exc}"
                    )
                except Exception:
                    pass
                return False

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
        # Em Produção, save_label(auto) pode ocorrer dentro desse processamento;
        # por isso marcamos a futura sessão multilight e adiamos somente o envio
        # automático 0/1 até TOP/MID serem recebidas.
        self.adhesive_multilight_pending_start = True
        try:
            result = original_process_aoi_images(
                self,
                sample_crop,
                ng_crop,
                aoi_info,
            )
        finally:
            self.adhesive_multilight_pending_start = False

        if (
            getattr(self, "current_analysis", None) is None
            or not _valid_image(getattr(self, "current_ng", None))
        ):
            self.adhesive_multilight_deferred_auto_decision = ""
            return result

        self.adhesive_multilight_active = True
        self.adhesive_multilight_aux_mode = None
        self.adhesive_multilight_views = {}
        self.adhesive_multilight_analyses = {}
        self.adhesive_multilight_source_frames = {}
        self.adhesive_multilight_last_analyses = {}
        self.adhesive_multilight_last_source_frames = {}
        self.adhesive_multilight_last_category = ADHESIVE_CATEGORY
        self.adhesive_multilight_final_analysis = None
        self.adhesive_multilight_last_final_analysis = None
        self.adhesive_multilight_primary_event_id = getattr(
            self,
            "network_intake_last_image_event_id",
            None,
        )
        self.adhesive_multilight_last_event_id = str(
            self.adhesive_multilight_primary_event_id or ""
        )

        side_source = getattr(
            self,
            "network_intake_last_image",
            None,
        )
        if not _valid_image(side_source):
            side_source = getattr(
                self,
                "capture_debug_last_image",
                None,
            )
        if _valid_image(side_source):
            side_copy = side_source.copy()
            self.adhesive_multilight_source_frames["SIDE"] = side_copy
            self.adhesive_multilight_last_source_frames["SIDE"] = (
                side_copy.copy()
            )

        # Contrato da AOI: a iluminação padrão/primeira recebida é SIDE.
        _store_view(self, "SIDE", sample_crop, ng_crop)

        # SIDE preserva a análise principal existente. TOP/MID serão analisados
        # separadamente quando seus frames auxiliares chegarem.
        self.adhesive_multilight_analyses["SIDE"] = getattr(
            self,
            "current_analysis",
            None,
        )
        self.adhesive_multilight_last_analyses["SIDE"] = getattr(
            self,
            "current_analysis",
            None,
        )

        analysis_view = getattr(
            self,
            "adhesive_multilight_analysis_view",
            None,
        )
        if analysis_view is not None:
            analysis_view.clear_all()
            analysis_view.set_analysis(
                "SIDE",
                getattr(self, "current_analysis", None),
            )

        _switch_inspection_view(self, True)
        _set_receiver_auxiliary_mode(self, True)

        # SIDE não é mais julgamento final para adesivo. Até TOP/MID terminarem
        # nenhum botão 0/1 do ODIN deve ficar disponível.
        try:
            self.lbl_verdict.setText("ADESIVO • AGUARDANDO TOP/MID")
            self.lbl_verdict.setStyleSheet(
                "color: #ffd33d; font-size: 16px; font-weight: bold; "
                "border: none;"
            )
            self.lbl_reason.setText(
                "SIDE concluída. Coletando TOP e MID para o único "
                "julgamento final."
            )
            self.lbl_timer.setText("Coletando TOP/MID...")
            self.btn_save_ok.setEnabled(False)
            self.btn_save_ng.setEnabled(False)
            self.btn_skip.setEnabled(False)
        except Exception:
            pass

        _sync_multilight_debug_controls(self)

        automation = getattr(
            self,
            "adhesive_multilight_automation",
            None,
        )
        start_automation = getattr(automation, "start", None)
        if callable(start_automation):
            start_automation()

        return result

    def save_label(self, *args, **kwargs):
        decision = ""
        if args:
            decision = str(args[0] or "").strip().upper()
        elif "user_decision" in kwargs:
            decision = str(
                kwargs.get("user_decision") or ""
            ).strip().upper()

        source = str(kwargs.get("source", "button") or "button").strip().lower()
        automation = getattr(
            self,
            "adhesive_multilight_automation",
            None,
        )
        automation_active = bool(
            getattr(automation, "active", False)
        )
        pending_start = bool(
            getattr(
                self,
                "adhesive_multilight_pending_start",
                False,
            )
        )

        # A peça precisa permanecer na AOI até TOP e MID chegarem. Em Produção
        # guardamos a decisão automática já calculada; em ações do ODIN apenas
        # recusamos o julgamento enquanto a sequência estiver ativa.
        if (
            is_adhesive_category(
                getattr(self, "current_aoi_info", None)
            )
            and (pending_start or automation_active)
        ):
            if source == "auto" and decision in {"OK", "NG"}:
                self.adhesive_multilight_deferred_auto_decision = decision
                try:
                    self.update_brain_status(
                        "Decisão automática pronta; aguardando fotos TOP/MID.",
                        True,
                    )
                except Exception:
                    pass
                return False

            try:
                self.update_brain_status(
                    "Aguarde a captura automática SIDE/TOP/MID antes de julgar.",
                    True,
                )
            except Exception:
                pass
            return False

        cancel_cycle = getattr(automation, "cancel_for_cycle_end", None)
        if callable(cancel_cycle):
            cancel_cycle()

        # Fecha primeiro a passagem de previews para não haver corrida com a
        # liberação do gate da próxima peça.
        _set_receiver_auxiliary_mode(self, False)
        result = original_save_label(self, *args, **kwargs)
        if not bool(getattr(self, "is_locked", False)):
            _reset_session(self, show_normal=True)
        return result

    def skip_image(self, *args, **kwargs):
        automation = getattr(
            self,
            "adhesive_multilight_automation",
            None,
        )
        cancel_cycle = getattr(automation, "cancel_for_cycle_end", None)
        if callable(cancel_cycle):
            cancel_cycle()

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
    control_panel_cls.finalize_adhesive_multilight_decision = (
        _finalize_multilight_decision
    )
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
