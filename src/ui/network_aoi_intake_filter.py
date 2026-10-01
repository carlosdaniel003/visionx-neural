"""Filtro final para imagens AOI recebidas pela rede.

O Windows XP pode enviar a tela da central durante a transição entre peças. O
receptor confirma estabilidade temporal; este módulo confirma o conteúdo da
inspeção antes que ControlPanel.process_aoi_images substitua a peça atual.
Capturas MSS locais não passam por este filtro.

A telemetria de diagnóstico registrada aqui é somente observabilidade: ela não
altera os critérios de aceitação, o EpicenterExtractor ou a decisão produtiva.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4
from typing import Any

import cv2
import numpy as np

from src.config.settings import settings
from src.core.epicenter_extractor import EpicenterExtractor
from src.core.inspection import detect_anomalies
from src.ui.network_xp_debug import (
    DEBUG_SCHEMA,
    set_network_debug_available,
)


MIN_FOCUS_SIDE = 4
DEBUG_BOX_LIMIT = 20
RADAR_GREEN_LOWER = np.array([50, 150, 100], dtype=np.uint8)
RADAR_GREEN_UPPER = np.array([75, 255, 255], dtype=np.uint8)


def _valid_image(value: Any) -> bool:
    return bool(
        isinstance(value, np.ndarray)
        and value.size > 0
        and value.ndim in {2, 3}
        and value.shape[0] >= MIN_FOCUS_SIDE
        and value.shape[1] >= MIN_FOCUS_SIDE
    )


def _json_safe(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _box_list(values: Any) -> list[list[int]]:
    output = []
    if not isinstance(values, (list, tuple)):
        return output
    for item in values[:DEBUG_BOX_LIMIT]:
        try:
            if len(item) < 4:
                continue
            output.append([int(round(float(value))) for value in item[:4]])
        except Exception:
            continue
    return output


def _image_summary(image: Any) -> dict:
    if not isinstance(image, np.ndarray) or image.size == 0:
        return {
            "valid": False,
            "shape": [],
            "dtype": "",
            "min": None,
            "max": None,
            "mean": None,
        }

    summary = {
        "valid": True,
        "shape": [int(value) for value in image.shape],
        "dtype": str(image.dtype),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "mean": round(float(np.mean(image)), 3),
    }
    if image.ndim == 3 and image.shape[2] >= 3:
        means = np.mean(image[:, :, :3], axis=(0, 1))
        summary["mean_bgr"] = [round(float(value), 3) for value in means]
    return summary


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image[:, :, :3].copy()


def _green_diagnostics(
    image: Any,
    lower: Any,
    upper: Any,
    *,
    min_width: int = 10,
    min_height: int = 10,
    max_width_ratio: float | None = None,
    max_height_ratio: float | None = None,
    morphology_close: bool = False,
) -> dict:
    if not _valid_image(image):
        return {
            "valid": False,
            "hsv_lower": _json_safe(lower),
            "hsv_upper": _json_safe(upper),
            "green_pixels": 0,
            "green_ratio": 0.0,
            "contour_count": 0,
            "valid_box_count": 0,
            "boxes": [],
            "valid_boxes": [],
        }

    try:
        bgr = _as_bgr(image)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower_array = np.asarray(lower, dtype=np.uint8).reshape(3)
        upper_array = np.asarray(upper, dtype=np.uint8).reshape(3)
        mask = cv2.inRange(hsv, lower_array, upper_array)
        if morphology_close:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            )
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes.sort(key=lambda box: box[2] * box[3], reverse=True)

        height, width = bgr.shape[:2]
        valid_boxes = []
        for box in boxes:
            x, y, box_width, box_height = box
            if box_width <= min_width or box_height <= min_height:
                continue
            oversized_width = bool(
                max_width_ratio is not None
                and box_width >= width * max_width_ratio
            )
            oversized_height = bool(
                max_height_ratio is not None
                and box_height >= height * max_height_ratio
            )

            if max_width_ratio is not None and max_height_ratio is not None:
                # Espelha o Radar real: a moldura só é descartada por tamanho
                # quando é gigante simultaneamente nos dois eixos.
                if oversized_width and oversized_height:
                    continue
            elif oversized_width or oversized_height:
                continue

            valid_boxes.append((x, y, box_width, box_height))

        green_pixels = int(cv2.countNonZero(mask))
        total_pixels = max(1, int(mask.shape[0] * mask.shape[1]))
        return {
            "valid": True,
            "hsv_lower": [int(value) for value in lower_array],
            "hsv_upper": [int(value) for value in upper_array],
            "green_pixels": green_pixels,
            "green_ratio": round(float(green_pixels / total_pixels), 8),
            "contour_count": int(len(contours)),
            "valid_box_count": int(len(valid_boxes)),
            "boxes": _box_list(boxes),
            "valid_boxes": _box_list(valid_boxes),
        }
    except Exception as exc:
        return {
            "valid": False,
            "error": str(exc),
            "hsv_lower": _json_safe(lower),
            "hsv_upper": _json_safe(upper),
            "green_pixels": 0,
            "green_ratio": 0.0,
            "contour_count": 0,
            "valid_box_count": 0,
            "boxes": [],
            "valid_boxes": [],
        }


def _radar_green_diagnostics(sample_crop: Any) -> dict:
    diagnostics = _green_diagnostics(
        sample_crop,
        RADAR_GREEN_LOWER,
        RADAR_GREEN_UPPER,
        min_width=15,
        min_height=15,
        max_width_ratio=0.85,
        max_height_ratio=0.85,
        morphology_close=True,
    )
    if not diagnostics.get("valid"):
        return diagnostics

    try:
        height, width = sample_crop.shape[:2]
        center_x, center_y = width / 2.0, height / 2.0
        ranked = []
        for box in diagnostics.get("valid_boxes", []):
            x, y, box_width, box_height = box
            box_center_x = x + box_width / 2.0
            box_center_y = y + box_height / 2.0
            distance = float(
                np.hypot(center_x - box_center_x, center_y - box_center_y)
            )
            ranked.append(
                {
                    "box": [x, y, box_width, box_height],
                    "distance_to_center": round(distance, 3),
                    "width_ratio": round(float(box_width / max(width, 1)), 6),
                    "height_ratio": round(float(box_height / max(height, 1)), 6),
                    "area_ratio": round(
                        float(
                            (box_width * box_height)
                            / max(width * height, 1)
                        ),
                        6,
                    ),
                }
            )
        ranked.sort(key=lambda item: item["distance_to_center"])
        diagnostics["ranked_center_candidates"] = ranked[:DEBUG_BOX_LIMIT]
        diagnostics["candidate_selected_by_radar"] = (
            ranked[0]["box"] if ranked else None
        )
    except Exception as exc:
        diagnostics["ranking_error"] = str(exc)
    return diagnostics


def _diagnostic_hints(
    old_epicenters: list,
    real_epicenters: list,
    test_green: dict,
    sample_radar_green: dict,
) -> list[str]:
    hints: list[str] = []
    if real_epicenters:
        return hints

    if int(test_green.get("green_pixels", 0) or 0) == 0:
        hints.append(
            "O recorte TESTE não contém pixels verdes dentro do HSV configurado "
            "para a hierarquia de caixas da AOI."
        )
    elif int(test_green.get("valid_box_count", 0) or 0) == 0:
        hints.append(
            "Há pixels verdes no TESTE, mas nenhum contorno verde passou pelo "
            "filtro mínimo de tamanho usado na detecção inicial."
        )

    if int(sample_radar_green.get("green_pixels", 0) or 0) == 0:
        hints.append(
            "O GABARITO não contém pixels verdes dentro da faixa fixa usada pelo "
            "Radar Euclidiano do EpicenterExtractor."
        )
    elif int(sample_radar_green.get("valid_box_count", 0) or 0) == 0:
        hints.append(
            "O GABARITO contém verde, mas nenhuma caixa passou pelo Radar: "
            "é necessário ter mais de 15 px por lado e não ocupar 85% ou mais "
            "da largura E da altura simultaneamente."
        )

    if not old_epicenters and int(test_green.get("valid_box_count", 0) or 0) > 0:
        hints.append(
            "Foram encontradas caixas verdes no TESTE, porém detect_anomalies não "
            "produziu uma caixa menor de epicentro; pode ter restado apenas a "
            "caixa global ou caixas fora da hierarquia esperada."
        )

    if (
        int(test_green.get("valid_box_count", 0) or 0) > 0
        and int(sample_radar_green.get("valid_box_count", 0) or 0) == 0
    ):
        hints.append(
            "A marcação verde aparece no TESTE, mas o Radar que escolhe o "
            "epicentro procura candidatos no GABARITO e não encontrou um válido."
        )

    if not hints:
        hints.append(
            "A imagem chegou utilizável, mas nenhum dos caminhos de fallback "
            "resultou em epicentro válido; compare as caixas registradas no JSON."
        )
    return hints


def _base_validation_audit(
    sample_crop: Any,
    ng_crop: Any,
) -> dict:
    return {
        "sample_crop": _image_summary(sample_crop),
        "test_crop": _image_summary(ng_crop),
        "green_detection": {
            "inspection_test_settings": _green_diagnostics(
                ng_crop,
                settings.COLOR_GREEN_LOWER,
                settings.COLOR_GREEN_UPPER,
                min_width=10,
                min_height=10,
            ),
            "epicenter_radar_sample": _radar_green_diagnostics(sample_crop),
        },
    }


def validate_network_inspection(
    sample_crop: np.ndarray,
    ng_crop: np.ndarray,
) -> tuple[bool, str, dict]:
    """Exige recortes utilizáveis e o epicentro menor escolhido pelo sistema."""
    audit = _base_validation_audit(sample_crop, ng_crop)

    if not _valid_image(sample_crop) or not _valid_image(ng_crop):
        audit.update(
            {
                "valid": False,
                "reason": "invalid_crops",
                "diagnostic_hints": [
                    "O par gabarito/teste produzido a partir da imagem de rede "
                    "está vazio ou possui dimensão abaixo do mínimo operacional."
                ],
            }
        )
        return False, "gabarito ou teste vazio/inválido", audit

    try:
        (
            raw_anomalies,
            old_epicenters,
            global_box_info,
            _gab_focus,
            _test_focus,
        ) = detect_anomalies(sample_crop, ng_crop)
        real_epicenters, focus_gab, focus_ng = EpicenterExtractor.extract_focus(
            sample_crop,
            ng_crop,
            old_epicenters,
            global_box_info,
        )
    except Exception as exc:
        audit.update(
            {
                "valid": False,
                "reason": "validation_exception",
                "error": str(exc),
                "diagnostic_hints": [
                    "A validação lançou uma exceção antes de concluir a seleção "
                    "do epicentro."
                ],
            }
        )
        return False, f"falha ao validar epicentro: {exc}", audit

    raw_boxes = _box_list(raw_anomalies or [])
    old_boxes = _box_list(old_epicenters or [])
    real_boxes = _box_list(real_epicenters or [])
    audit.update(
        {
            "raw_anomaly_count": int(len(raw_anomalies or [])),
            "raw_anomalies": raw_boxes,
            "old_epicenter_count": int(len(old_epicenters or [])),
            "old_epicenters": old_boxes,
            "global_box_info": _json_safe(global_box_info or {}),
            "real_epicenter_count": int(len(real_epicenters or [])),
            "real_epicenters": real_boxes,
        }
    )

    if not real_epicenters:
        audit.update(
            {
                "valid": False,
                "reason": "missing_epicenter",
                "epicenter_count": 0,
                "diagnostic_hints": _diagnostic_hints(
                    old_epicenters or [],
                    real_epicenters or [],
                    audit["green_detection"]["inspection_test_settings"],
                    audit["green_detection"]["epicenter_radar_sample"],
                ),
            }
        )
        return False, "tela sem epicentro de anomalia", audit

    try:
        x, y, width, height = (
            int(round(float(value))) for value in real_epicenters[0][:4]
        )
    except Exception:
        audit.update(
            {
                "valid": False,
                "reason": "invalid_epicenter_box",
                "diagnostic_hints": [
                    "O EpicenterExtractor retornou uma caixa, mas suas "
                    "coordenadas não puderam ser convertidas em números inteiros."
                ],
            }
        )
        return False, "coordenadas do epicentro inválidas", audit

    audit["focus_box"] = [x, y, width, height]

    if width < MIN_FOCUS_SIDE or height < MIN_FOCUS_SIDE:
        audit.update(
            {
                "valid": False,
                "reason": "epicenter_too_small",
                "diagnostic_hints": [
                    "O epicentro existe, porém ficou abaixo do mínimo operacional "
                    f"de {MIN_FOCUS_SIDE}px por lado."
                ],
            }
        )
        return False, "epicentro menor que o mínimo operacional", audit

    if not _valid_image(focus_gab) or not _valid_image(focus_ng):
        audit.update(
            {
                "valid": False,
                "reason": "empty_focus_pair",
                "focus_reference": _image_summary(focus_gab),
                "focus_test": _image_summary(focus_ng),
                "diagnostic_hints": [
                    "A caixa do epicentro foi encontrada, mas o recorte não gerou "
                    "um par gabarito/teste utilizável."
                ],
            }
        )
        return False, "epicentro não gerou o par gabarito/teste", audit

    audit.update(
        {
            "valid": True,
            "reason": "valid_epicenter",
            "focus_shape": [
                int(focus_ng.shape[1]),
                int(focus_ng.shape[0]),
            ],
            "epicenter_count": int(len(real_epicenters)),
            "diagnostic_hints": [],
        }
    )
    return True, "epicentro válido", audit


def _gate_snapshot(panel) -> dict:
    receiver = getattr(panel, "network_receiver", None)
    if receiver is None or not hasattr(receiver, "image_gate_snapshot"):
        return {}
    try:
        snapshot = receiver.image_gate_snapshot()
        return {
            "accepting_images": bool(snapshot.accepting_images),
            "generation": int(snapshot.generation),
            "ignored_images": int(snapshot.ignored_images),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _mode(panel) -> str:
    try:
        return str(panel.combo_mode.currentText()).strip()
    except Exception:
        return ""


def _transport_record(panel, image: Any, ip: str, event_id: str) -> dict:
    receiver = getattr(panel, "network_receiver", None)
    return {
        "schema": DEBUG_SCHEMA,
        "event_id": str(event_id),
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "source": "windows_xp",
        "source_ip": str(ip or ""),
        "stage": "network_image_received",
        "mode": _mode(panel),
        "transport": {
            "image": _image_summary(image),
            "stable_required_frames": int(
                getattr(receiver, "STABLE_REQUIRED_FRAMES", 0) or 0
            ),
        },
        "cycle": _gate_snapshot(panel),
        "validation_message": "Imagem recebida; aguardando extração da AOI.",
        "validation": {
            "valid": None,
            "reason": "pending_aoi_validation",
            "diagnostic_hints": [],
        },
    }


def _enrich_validation_record(
    panel,
    reason: str,
    audit: dict,
    aoi_info: dict | None,
) -> dict:
    previous = getattr(panel, "network_intake_last_validation", {})
    transport = (
        dict(previous.get("transport", {}))
        if isinstance(previous, dict)
        else {}
    )
    timestamp = (
        str(previous.get("timestamp", ""))
        if isinstance(previous, dict)
        else ""
    ) or datetime.now().isoformat(timespec="milliseconds")
    source_ip = (
        str(previous.get("source_ip", ""))
        if isinstance(previous, dict)
        else ""
    ) or str(getattr(panel, "last_xp_ip", "") or "")

    event_id = (
        str(previous.get("event_id", ""))
        if isinstance(previous, dict)
        else ""
    )

    return {
        "schema": DEBUG_SCHEMA,
        "event_id": event_id,
        "timestamp": timestamp,
        "source": "windows_xp",
        "source_ip": source_ip,
        "stage": "aoi_intake_validation",
        "mode": _mode(panel),
        "transport": transport,
        "cycle": _gate_snapshot(panel),
        "aoi_info": _json_safe(aoi_info or {}),
        "validation_message": str(reason or ""),
        "validation": _json_safe(audit or {}),
    }


def _decision_record(analysis: Any, aoi_info: dict | None) -> dict:
    if not isinstance(analysis, dict):
        return {}

    detail = analysis.get("detail", {})
    detail = detail if isinstance(detail, dict) else {}
    trace = detail.get("decision_trace", {})
    trace = trace if isinstance(trace, dict) else {}
    memory = trace.get("memory", {})
    memory = memory if isinstance(memory, dict) else {}

    missing_fields = (
        "missing_active",
        "missing_is_defect",
        "missing_score",
        "missing_tolerance",
        "missing_classification",
        "missing_changed_coverage",
        "missing_residual_mean",
        "missing_structure_loss",
        "missing_background_exposure",
        "missing_best_similarity",
        "missing_direct_similarity",
        "missing_appearance_loss",
        "missing_edge_mismatch",
        "missing_residual_p90",
        "missing_hard_absence",
        "missing_hard_absence_reason",
        "missing_cross_category_guard",
        "missing_guard_policy",
        "missing_guard_source_category",
        "missing_guard_physical_support",
    )
    missing = {
        key: _json_safe(detail.get(key))
        for key in missing_fields
        if key in detail
    }

    raw_memory_conflict = bool(
        memory.get(
            "memory_conflict",
            detail.get("memory_conflict", False),
        )
    )
    raw_memory_review = bool(
        memory.get(
            "operator_review_required",
            detail.get("operator_review_required", False),
        )
    )
    hard_missing = bool(
        trace.get("hard_missing_evidence", False)
        or memory.get("suppressed_by_hard_missing", False)
        or detail.get("missing_hard_absence", False)
        or str(trace.get("fusion_rule", "")) == "missing_hard_absence"
    )
    effective_memory_conflict = bool(
        raw_memory_conflict and not hard_missing
    )
    effective_memory_review = bool(
        raw_memory_review and not hard_missing
    )

    return {
        "category": str((aoi_info or {}).get("category", "") or ""),
        "is_defect": bool(analysis.get("is_defect", False)),
        "verdict": str(analysis.get("verdict", "") or ""),
        "confidence": _json_safe(analysis.get("confidence")),
        "reason": str(analysis.get("reason", "") or ""),
        "final_score": _json_safe(detail.get("final_score")),
        "physical_score": _json_safe(detail.get("physical_score")),
        "fusion_rule": str(detail.get("fusion_rule", "") or ""),
        "dominant_engine": str(detail.get("dominant_engine", "") or ""),
        "operator_review_required": bool(
            analysis.get("production_review_required", False)
            or trace.get("operator_review_required", False)
            or effective_memory_review
        ),
        "hard_missing_evidence": bool(
            trace.get("hard_missing_evidence", False)
            or detail.get("missing_hard_absence", False)
        ),
        "missing": missing,
        "memory": {
            "has_memory": bool(
                memory.get("has_memory", detail.get("has_memory", False))
            ),
            "memory_available": bool(
                memory.get(
                    "memory_available",
                    detail.get("memory_available", False),
                )
            ),
            "best_match_label": str(
                memory.get(
                    "best_match_label",
                    detail.get("best_match_label", ""),
                )
                or ""
            ),
            "best_similarity": _json_safe(
                memory.get(
                    "best_similarity",
                    detail.get("best_similarity"),
                )
            ),
            "best_ok_similarity": _json_safe(
                memory.get(
                    "best_ok_similarity",
                    detail.get("best_ok_similarity"),
                )
            ),
            "best_ng_similarity": _json_safe(
                memory.get(
                    "best_ng_similarity",
                    detail.get("best_ng_similarity"),
                )
            ),
            "memory_conflict": effective_memory_conflict,
            "raw_memory_conflict": raw_memory_conflict,
            "operator_review_required": effective_memory_review,
            "raw_operator_review_required": raw_memory_review,
            "role": str(
                memory.get("role", detail.get("memory_reason", "")) or ""
            ),
            "suppressed_by_hard_missing": hard_missing,
        },
    }


def _safe_button(button, *, enabled: bool | None = None, text: str | None = None):
    if button is None:
        return
    try:
        if enabled is not None:
            button.setEnabled(bool(enabled))
        if text is not None:
            button.setText(str(text))
    except Exception:
        pass


def reject_invalid_network_capture(panel, reason: str, audit: dict | None = None) -> None:
    """Libera a procura sem permitir que a central ocupe uma peça ativa."""
    receiver = getattr(panel, "network_receiver", None)
    if receiver is not None and hasattr(receiver, "mark_reserved_image_rejected"):
        try:
            receiver.mark_reserved_image_rejected(reason)
        except Exception as exc:
            print(f"Falha não fatal ao registrar tela rejeitada: {exc}")

    panel.capture_cycle_network_generation = int(
        getattr(panel, "capture_cycle_network_generation", 0)
    ) + 1
    panel.capture_cycle_active = False
    panel.capture_cycle_ignored_signals = 0
    panel.capture_cycle_source = None
    panel.current_analysis = None
    panel.current_sample = None
    panel.current_ng = None
    panel.current_aoi_info = {}
    panel.is_locked = False
    panel.network_intake_last_validation = dict(audit or {})
    set_network_debug_available(panel, bool(audit))

    if hasattr(panel, "production_review_pending"):
        panel.production_review_pending = False

    for method_name in (
        "_reset_confidence_panel",
        "_reset_reference_panel",
        "_reset_aoi_info",
    ):
        method = getattr(panel, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:
                print(f"Falha não fatal em {method_name}: {exc}")

    _safe_button(
        getattr(panel, "btn_start", None),
        enabled=True,
        text="Capturar local (MSS)",
    )
    for name in ("btn_save_ok", "btn_save_ng", "btn_skip"):
        _safe_button(getattr(panel, name, None), enabled=False)

    if receiver is not None and hasattr(receiver, "release_image_gate"):
        try:
            receiver.release_image_gate()
        except Exception as exc:
            print(f"Falha não fatal ao liberar receptor após rejeição: {exc}")

    presenter = getattr(panel, "_operational_controls", None)
    if presenter is not None:
        try:
            presenter.sync(force=True)
        except Exception as exc:
            print(f"Falha não fatal ao sincronizar controles: {exc}")

    try:
        panel.update_brain_status(
            "Imagem XP ignorada antes do julgamento. "
            f"Motivo: {reason}. Use 'Copiar debug XP' para o diagnóstico completo.",
            False,
        )
    except Exception:
        pass


def install_network_aoi_intake_filter(control_panel_cls) -> None:
    """Protege somente o caminho de imagens externas recebidas pela rede."""
    if getattr(control_panel_cls, "_network_aoi_intake_filter_installed", False):
        return

    original_handle_network_image = control_panel_cls.handle_network_image
    original_process_aoi_images = control_panel_cls.process_aoi_images

    def handle_network_image(self, img_bgr, ip: str):
        event_id = uuid4().hex
        self.network_intake_last_image = (
            img_bgr.copy()
            if isinstance(img_bgr, np.ndarray) and img_bgr.size > 0
            else None
        )
        self.network_intake_last_image_event_id = event_id
        self.network_intake_last_validation = _transport_record(
            self,
            img_bgr,
            ip,
            event_id,
        )
        set_network_debug_available(self, True)
        return original_handle_network_image(self, img_bgr, ip)

    def process_aoi_images(self, sample_crop, ng_crop, aoi_info):
        if getattr(self, "capture_cycle_source", None) != "network":
            return original_process_aoi_images(
                self,
                sample_crop,
                ng_crop,
                aoi_info,
            )

        valid, reason, audit = validate_network_inspection(
            sample_crop,
            ng_crop,
        )
        record = _enrich_validation_record(
            self,
            reason,
            audit,
            aoi_info,
        )
        self.network_intake_last_validation = record
        set_network_debug_available(self, True)

        if not valid:
            reject_invalid_network_capture(self, reason, record)
            return False

        receiver = getattr(self, "network_receiver", None)
        mode = _mode(self)

        if mode == "Modo Produção" and receiver is not None and hasattr(
            receiver,
            "confirm_reserved_image",
        ):
            receiver.confirm_reserved_image()

        result = original_process_aoi_images(
            self,
            sample_crop,
            ng_crop,
            aoi_info,
        )

        if mode != "Modo Produção" and receiver is not None and hasattr(
            receiver,
            "confirm_reserved_image",
        ):
            receiver.confirm_reserved_image()

        analysis = getattr(self, "current_analysis", None)
        if isinstance(analysis, dict):
            # A categoria só é normalizada dentro do processamento principal.
            # Atualize o registro depois da análise para que o debug carregue
            # tanto o intake quanto a decisão efetivamente tomada.
            record["aoi_info"] = _json_safe(aoi_info or {})
            record["decision"] = _decision_record(analysis, aoi_info)
            self.network_intake_last_validation = record
            set_network_debug_available(self, True)

            detail = analysis.setdefault("detail", {})
            detail["network_intake_validation"] = dict(audit)
            detail["network_intake_debug"] = dict(record)
            detail["network_intake_stable_required"] = 2
            detail["network_intake_source"] = "windows_xp"

        return result

    control_panel_cls.handle_network_image = handle_network_image
    control_panel_cls.process_aoi_images = process_aoi_images
    control_panel_cls._network_aoi_intake_filter_installed = True


__all__ = [
    "MIN_FOCUS_SIDE",
    "install_network_aoi_intake_filter",
    "reject_invalid_network_capture",
    "validate_network_inspection",
]
