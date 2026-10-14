"""Telemetria passiva dos tempos do ciclo XP SIDE/TOP/MID.

Nunca interfere em comandos, liberação do gate, inspeção ou julgamento.
Usa um único relógio monotônico do PC ODIN; captura interna ao XP, antes
do primeiro pacote, não é mensurável sem instrumentação no agente.
"""
from __future__ import annotations

import time


LIGHTS = ("SIDE", "TOP", "MID")
STEPS = ("layout_crop_ms", "bars_detect_ms", "sample_crop_ms",
         "test_crop_ms", "debug_png_write_ms", "debug_png_write_count",
         "ocr_ms", "extraction_ms", "validation_ms", "geometry_ms",
         "preview_ms", "inference_ms", "render_ms", "processing_ms")
NETWORK_STEPS = ("tcp_payload_ms", "decode_ms", "stability_ms",
                 "last_stable_pair_gap_ms", "configured_settle_ms",
                 "payload_to_dispatch_ms", "dispatch_to_ui_ms")
NETWORK_COUNTS = ("stable_frame_count", "pre_settle_frames", "backlog_deferrals")
NETWORK_FLAGS = ("backlog_release_with_pending",)


def _ms(start, end):
    if isinstance(start, (float, int)) and isinstance(end, (float, int)):
        if start > 0 and end >= start:
            return round((end - start) * 1000.0, 2)
    return None


def _state(panel):
    value = getattr(panel, "_inspection_timing", None)
    return value if isinstance(value, dict) else None


def _archive_activity():
    """Somente snapshot não bloqueante do estado de varredura da thread KNN."""
    try:
        from src.services.knn_archive_auto_memory import archive_auto_memory_activity
        return archive_auto_memory_activity()
    except Exception:
        return None


def begin_cycle(
    panel, event_id, mode, transport=None, fast_ack=None,
    previous_gate_release_at=None,
):
    """Abre registro isolado pela identidade da primeira imagem SIDE."""
    transport = transport if isinstance(transport, dict) else {}
    started = float(transport.get("payload_at") or 0.0)
    panel._inspection_timing = {
        "event_id": str(event_id),
        "mode": str(mode),
        "start_at": started if started > 0 else time.perf_counter(),
        "finished_at": None,
        "fast_ack": fast_ack if fast_ack in (True, False) else None,
        "release_to_side_ms": _ms(previous_gate_release_at, started),
        "lights": {light: {"steps": {}, "network": {}, "command": {}}
                   for light in LIGHTS},
        "presentation": {},
    }
    note_frame(panel, "SIDE", transport)


def note_frame(panel, light, transport=None):
    state = _state(panel)
    if state is None or light not in LIGHTS:
        return
    data = transport if isinstance(transport, dict) else {}
    slot = state["lights"][light]
    activity = _archive_activity()
    if isinstance(activity, dict):
        slot["archive_worker"] = {"at_frame": activity}
    slot["network"] = {
        key: data.get(key)
        for key in (*NETWORK_STEPS, *NETWORK_COUNTS)
        if isinstance(data.get(key), (float, int))
    }
    slot["network"].update({
        key: data[key]
        for key in NETWORK_FLAGS
        if isinstance(data.get(key), bool)
    })
    payload = data.get("payload_at")
    ready = data.get("ready_at")
    if isinstance(payload, (float, int)) and payload > 0:
        slot["payload_at"] = float(payload)
    if isinstance(ready, (float, int)) and ready > 0:
        slot["ready_at"] = float(ready)
        slot["network"]["dispatch_to_ui_ms"] = _ms(
            ready, time.perf_counter()
        )
    command_started = slot["command"].get("started_at")
    if command_started:
        slot["command"]["to_payload_ms"] = _ms(command_started, payload)
        slot["command"]["to_ready_ms"] = _ms(command_started, ready)


def note_command(panel, light, started_at, completed_at, success):
    state = _state(panel)
    if state is None or light not in LIGHTS:
        return
    state["lights"][light]["command"] = {
        "started_at": float(started_at),
        "send_ms": _ms(started_at, completed_at),
        "success": bool(success),
    }


def note_steps(panel, light, **durations):
    state = _state(panel)
    if state is None or light not in LIGHTS:
        return
    light_slot = state["lights"][light]
    activity = _archive_activity()
    if isinstance(activity, dict):
        worker = light_slot.setdefault("archive_worker", {})
        worker["after_step"] = activity
        start = worker.get("at_frame")
        if isinstance(start, dict):
            a = start.get("busy_total_ms")
            b = activity.get("busy_total_ms")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                worker["busy_overlap_ms"] = round(max(0.0, b - a), 2)
    slot = light_slot["steps"]
    for key, value in durations.items():
        if key in STEPS and isinstance(value, (float, int)):
            slot[key] = (
                max(0, int(value)) if key == "debug_png_write_count"
                else round(max(0.0, float(value)), 2)
            )


def note_extraction(panel, light, monitor):
    timing = getattr(monitor, "last_external_timing", None)
    if isinstance(timing, dict):
        note_steps(panel, light, **timing)
        state = _state(panel)
        profile = timing.get("ocr_profile")
        if state is not None and light in LIGHTS and isinstance(profile, dict):
            state["lights"][light]["ocr_profile"] = {
                **profile,
                "calls": [dict(item) for item in profile.get("calls", [])
                          if isinstance(item, dict)],
            }


def finish_cycle(panel, when=None):
    state = _state(panel)
    if state is not None:
        state["finished_at"] = float(
            when if when is not None else time.perf_counter()
        )


def presentation_mark(panel, marker, when=None):
    state = _state(panel)
    if state is None or marker not in ("ready", "presented", "decision"):
        return
    state["presentation"][marker] = float(
        when if when is not None else time.perf_counter()
    )


def snapshot(panel, event_id):
    """Cópia JSON-safe; nunca expõe timestamps monotônicos brutos."""
    state = _state(panel)
    if state is None or not event_id or state["event_id"] != str(event_id):
        return {}
    lights = {}
    for light in LIGHTS:
        slot = state["lights"][light]
        command = slot["command"]
        lights[light] = {
            "network": dict(slot["network"]),
            "archive_worker": {
                key: dict(value) if isinstance(value, dict) else value
                for key, value in slot.get("archive_worker", {}).items()
            },
            "steps": dict(slot["steps"]),
            "ocr_profile": {
                **slot["ocr_profile"],
                "calls": [dict(item) for item in
                          slot["ocr_profile"].get("calls", [])
                          if isinstance(item, dict)],
            } if isinstance(slot.get("ocr_profile"), dict) else {},
            "command": {
                key: command[key]
                for key in ("send_ms", "to_payload_ms", "to_ready_ms", "success")
                if key in command
            },
            "payload_to_complete_ms": (
                _ms(slot.get("payload_at"), state["finished_at"])
                if light == "MID" else None
            ),
        }
    ready = state["presentation"].get("ready")
    presented = state["presentation"].get("presented")
    decision = state["presentation"].get("decision")
    return {
        "schema": "visionx.inspection_timing.v1",
        "event_id": state["event_id"],
        "mode": state["mode"],
        "xp_fast_ack": state["fast_ack"],
        "previous_gate_release_to_side_ms": state["release_to_side_ms"],
        "cycle_side_to_result_ms": _ms(
            state["start_at"], state["finished_at"]
        ),
        "lights": lights,
        "production_presentation_ms": _ms(ready, presented),
        "production_post_presentation_decision_ms": _ms(presented, decision),
        "note": (
            "Duracoes podem se sobrepor: nao somar etapas. "
            "Tempo anterior ao primeiro pacote SIDE e tempo interno do XP "
            "nao sao observaveis no ODIN."
        ),
    }


def format_report(data):
    if not isinstance(data, dict) or not data:
        return []
    lines = ["", "PERFIL DE TEMPO POR ETAPA (ms)", "-" * 72,
             "Ciclo SIDE recebido -> resultado final: {}".format(
                 _show(data.get("cycle_side_to_result_ms"))
             ),
             "XP captura rapida confirmada (ACK): {}".format(
                 _show(data.get("xp_fast_ack"))
             ),
             "Gate liberado -> proxima SIDE estavel: {} ms".format(
                 _show(data.get("previous_gate_release_to_side_ms"))
             )]
    names = {
        "tcp_payload_ms": "Transferencia do payload TCP",
        "decode_ms": "Descompressao/decode",
        "stability_ms": "Intervalo de estabilidade/coalescing de frames",
        "last_stable_pair_gap_ms": "Intervalo do ultimo par estavel",
        "configured_settle_ms": "Janela minima apos comando de luz",
        "payload_to_dispatch_ms": "Recebimento ate despacho",
        "dispatch_to_ui_ms": "Fila Qt: despacho ate UI",
        "layout_crop_ms": "Localizacao AOI / recortes",
        "bars_detect_ms": "Copia inicial / deteccao de barras",
        "sample_crop_ms": "Recorte gabarito azul (inclui PNGs debug)",
        "test_crop_ms": "Recorte teste vermelho (inclui PNGs debug)",
        "debug_png_write_ms": "Gravacao de PNGs de debug (dentro dos recortes)",
        "debug_png_write_count": "PNGs de debug gravados",
        "ocr_ms": "OCR",
        "extraction_ms": "Extracao total (inclui OCR)",
        "validation_ms": "Validacao de entrada AOI",
        "geometry_ms": "Geometria / epicentros",
        "preview_ms": "Preparo visual / miniaturas",
        "inference_ms": "Inferencia dos especialistas",
        "render_ms": "Renderizacao / debug visual",
        "processing_ms": "Processamento local total",
    }
    for light in LIGHTS:
        item = data.get("lights", {}).get(light, {})
        if not isinstance(item, dict):
            continue
        lines.extend(["", "ILUMINACAO " + light])
        command = item.get("command", {})
        if command:
            lines.append("Comando TCP: {} ms | Sucesso: {}".format(
                _show(command.get("send_ms")), _show(command.get("success"))
            ))
            lines.append("Comando -> payload completo: {} ms".format(
                _show(command.get("to_payload_ms"))
            ))
            lines.append("Comando -> frame estavel entregue: {} ms".format(
                _show(command.get("to_ready_ms"))
            ))
        for group_name in ("network", "steps"):
            group = item.get(group_name, {})
            for key in (NETWORK_STEPS if group_name == "network" else STEPS):
                if key in group:
                    lines.append("{}: {}{}".format(
                        names[key], _show(group[key]),
                        "" if key == "debug_png_write_count" else " ms",
                    ))
            if group_name == "network":
                if "stable_frame_count" in group:
                    lines.append(
                        "Frames consecutivos estaveis confirmados: {}".format(
                            _show(group["stable_frame_count"])
                        )
                    )
                if "pre_settle_frames" in group:
                    lines.append(
                        "Frames antes da janela de troca de luz: {}".format(
                            _show(group["pre_settle_frames"])
                        )
                    )
                if "backlog_deferrals" in group:
                    lines.append(
                        "Frames adiados por conexoes XP pendentes: {}".format(
                            _show(group["backlog_deferrals"])
                        )
                    )
                if "backlog_release_with_pending" in group:
                    lines.append(
                        "Entrega XP com fila ainda ocupada apos estabilidade reforcada: {}".format(
                            _show(group["backlog_release_with_pending"])
                        )
                    )
        ocr_profile = item.get("ocr_profile", {})
        if isinstance(ocr_profile, dict) and ocr_profile:
            lines.extend([
                "OCR fase/rota: {}".format(ocr_profile.get("route", "-")),
                "OCR recuperacao Parts: {}".format(
                    ocr_profile.get("parts_reason", "-")
                ),
                "OCR releitura zona vermelha: {}".format(
                    ocr_profile.get("general_retry", False)
                ),
                "OCR subprocessos Tesseract: {}".format(
                    ocr_profile.get("call_count", "-")
                ),
                "OCR soma das chamadas Tesseract: {} ms".format(
                    _show(ocr_profile.get("tesseract_calls_ms"))
                ),
            ])
            for call in ocr_profile.get("calls", ()):
                if isinstance(call, dict):
                    lines.append(
                        "  Tesseract {}: {} ms | {}".format(
                            call.get("phase", "-"),
                            _show(call.get("ms")),
                            call.get("status", "-"),
                        )
                    )
    if data.get("production_presentation_ms") is not None:
        lines.append("Apresentacao da Producao apos resultado: {} ms".format(
            _show(data["production_presentation_ms"])
        ))
    if data.get("production_post_presentation_decision_ms") is not None:
        lines.append("Espera apos apresentacao ate decisao automatica: {} ms".format(
            _show(data["production_post_presentation_decision_ms"])
        ))
    candidates = []
    for light in LIGHTS:
        entry = data.get("lights", {}).get(light, {})
        if not isinstance(entry, dict):
            continue
        for group, keys in (
            ("network", ("tcp_payload_ms", "decode_ms", "dispatch_to_ui_ms")),
            ("steps", ("validation_ms", "geometry_ms", "ocr_ms",
                       "inference_ms", "preview_ms", "render_ms")),
        ):
            values = entry.get(group, {})
            for key in keys:
                number = values.get(key)
                if isinstance(number, (float, int)) and number > 0:
                    candidates.append((float(number), "{}: {}".format(
                        light, names[key])))
        command_wait = entry.get("command", {}).get("to_ready_ms")
        if isinstance(command_wait, (float, int)) and command_wait > 0:
            candidates.append((float(command_wait), light + ": comando ate frame"))
    prev_gap = data.get("previous_gate_release_to_side_ms")
    if isinstance(prev_gap, (float, int)) and prev_gap > 0:
        candidates.append((float(prev_gap), "Entre pecas: gate ate proxima SIDE"))
    presentation = data.get("production_presentation_ms")
    if isinstance(presentation, (float, int)) and presentation > 0:
        candidates.append((float(presentation), "Producao: apresentacao"))
    if candidates:
        lines.extend(["", "MAIORES TEMPOS OBSERVADOS (INDICATIVOS)"])
        for duration, label in sorted(candidates, reverse=True)[:5]:
            lines.append("- {}: {} ms".format(label, round(duration, 2)))
    lines.extend([
        "",
        "NOTAS DE MEDICAO",
        "Etapas OCR/extracao, geometria e processamento incluem subetapas.",
        "Comando->frame inclui espera XP, transporte e estabilidade.",
        "Tempos sobrepostos nao devem ser somados.",
        "O XP anterior ao primeiro frame SIDE nao e mensuravel aqui.",
    ])
    return lines


def _show(value):
    return "-" if value is None else str(value)
