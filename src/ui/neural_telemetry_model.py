"""Telemetria pura de CNN FALTANDO v2 e KNN: observabilidade, sem decisão."""
from __future__ import annotations

import math

LIGHTS = ("SIDE", "TOP", "MID")


def _map(value):
    return value if isinstance(value, dict) else {}


def _finite_score(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


def _route(detail: dict) -> str:
    return str(detail.get("recognition_route", "") or "")


def cnn_lighting_telemetry(analyses: dict) -> dict:
    """Snapshot dos 3 especialistas. Ausente não equivale a 0%."""
    out = {}
    for light in LIGHTS:
        frame = _map(_map(analyses).get(light))
        d = _map(frame.get("detail"))
        cnn = bool(d.get("cnn_v2_active") or "faltando_cnn_v2.py" in (frame.get("active_engines") or []))
        out[light] = {
            "route": _route(d) or ("NEW_CNN" if cnn else ""),
            "engine": "faltando_cnn_v2" if cnn else str(d.get("dominant_engine", "") or ""),
            "cnn_active": cnn,
            "cnn_status": str(d.get("cnn_v2_status", "") or ""),
            "checkpoint_verified": bool(d.get("cnn_v2_checkpoint_verified", False)) if cnn else None,
            "checkpoint_sha256": str(d.get("cnn_v2_checkpoint_sha256", "") or "") if cnn else "",
            "ng_score_uncalibrated": _finite_score(d.get("cnn_v2_ng_score_uncalibrated")) if cnn else None,
            "verdict": str(frame.get("verdict", "") or ""),
            "category": str(d.get("cnn_v2_aoi_category", "") or ""),
            "human_memory_label": str(d.get("recognition_known_label", "") or ""),
        }
    return out


def neural_summary(analysis: dict | None) -> dict:
    frame = _map(analysis)
    d = _map(frame.get("detail"))
    trace = _map(d.get("decision_trace"))
    route = _route(d)
    per_light = _map(d.get("cnn_v2_light_diagnostics"))
    has_cnn = bool(
        d.get("cnn_v2_active")
        or "faltando_cnn_v2.py" in (frame.get("active_engines") or [])
        or route == "NEW_CNN"
        or any(_map(x).get("cnn_active") for x in per_light.values())
    )
    local_ng = _finite_score(d.get("cnn_v2_ng_score_uncalibrated"))
    if local_ng is None:
        local_ng = _finite_score(trace.get("cnn_ng_score_uncalibrated"))
    lights = per_light or {}
    routes = _map(d.get("recognition_light_routes"))
    memory_known = route == "KNOWN_KNN" or any(
        value == "KNOWN_KNN" for value in routes.values()
    )
    memory_new = route == "NEW_CNN" or any(
        value == "NEW_CNN" for value in routes.values()
    )
    memory_label = str(d.get("recognition_known_label", "") or "")
    return {
        "active": bool(frame),
        "cnn_active": has_cnn,
        "cnn_route": route,
        "cnn_ng_score": local_ng,
        "cnn_ok_score": (1.0 - local_ng) if local_ng is not None else None,
        "cnn_status": str(d.get("cnn_v2_status", "") or ""),
        "cnn_checkpoint_verified": d.get("cnn_v2_checkpoint_verified") is True,
        "cnn_checkpoint_sha256": str(d.get("cnn_v2_checkpoint_sha256", "") or ""),
        "cnn_best_epoch": d.get("cnn_v2_checkpoint_best_epoch"),
        "cnn_category": str(d.get("cnn_v2_aoi_category") or d.get("multilight_category") or ""),
        "cnn_lighting_mode": str(d.get("cnn_v2_lighting_mode") or frame.get("lighting_mode") or ""),
        "cnn_votes": _map(d.get("cnn_v2_light_votes")),
        "cnn_per_light": lights,
        "cnn_consensus": str(d.get("cnn_v2_consensus", "") or ""),
        "cnn_consensus_reason": str(d.get("cnn_v2_consensus_reason", "") or ""),
        "cnn_auto_eligible": d.get("cnn_v2_supervised_auto_eligible") is True,
        "memory_route": route,
        "memory_known": memory_known,
        "memory_new_cnn": memory_new,
        "memory_label": memory_label,
        "memory_source": str(d.get("recognition_memory_path", "") or ""),
        "memory_reason": str(d.get("recognition_reason", "") or ""),
        "memory_routes": routes,
        "memory_similarity": _finite_score(d.get("recognition_best_similarity")),
        "verdict": str(frame.get("verdict", "") or ""),
    }


def percent(value, *, precision=2):
    if value is None:
        return "N/D"
    return f"{value * 100:.{precision}f}%"


def cnn_panel_text(analysis: dict | None) -> dict:
    m = neural_summary(analysis)
    if not m["cnn_active"]:
        return {"header": "CNN FALTANDO v2 • não executada",
                "lines": ("O julgamento pertence a outro motor.",)}
    label = m["cnn_lighting_mode"] or "MULTILIGHT"
    sha = m["cnn_checkpoint_sha256"]
    ck = "VERIFICADO" if m["cnn_checkpoint_verified"] else "NÃO VERIFICADO"
    lines = [
        f"Categoria AOI: {m['cnn_category'] or 'não informada'}",
        f"Motor: CNN FALTANDO v2 • {label}",
        f"Rota: {m['cnn_route'] or 'MULTILIGHT'}",
        f"Estado: {m['cnn_status'] or 'consulte as três luzes'}",
    ]
    if not m["cnn_per_light"]:
        lines.extend((
            f"Score NG (não calibrado): {percent(m['cnn_ng_score'], precision=4)}",
            f"Score OK complementar: {percent(m['cnn_ok_score'], precision=4)}",
        ))
    elif m["cnn_per_light"]:
        for light in LIGHTS:
            item = _map(m["cnn_per_light"].get(light))
            score = _finite_score(item.get("ng_score_uncalibrated"))
            status = item.get("verdict", "não disponível") or "não disponível"
            lines.append(f"{light}: {status} • NG {percent(score, precision=4)}")
        lines.append(f"Consenso CNN: {m['cnn_consensus'] or 'REVISÃO'}")
        lines.append(
            "Decisão automática supervisionada: "
            + ("ELEGÍVEL" if m["cnn_auto_eligible"] else "NÃO ELEGÍVEL")
        )
    lines.append(f"Checkpoint: {ck}" + (f" • SHA {sha[:12]}…" if sha else ""))
    lines.append("Score NG não calibrado: não representa probabilidade de defeito.")
    return {"header": "CNN FALTANDO v2 • REDE NEURAL", "lines": tuple(lines)}



def memory_seen_state(analysis: dict | None) -> dict:
    """Estado visual binário de registro humano, nunca 'meio termo'.

    JA_VI = pelo menos uma iluminação recuperada via KNOWN_KNN, com
    par exato verificado; NUNCA_VI = consulta válida sem match exato.
    Sem resultado de consulta, devolve None: não inventa 'nunca'.
    Os rótulos tratam histórico de PARES, não tipos físicos de defeito.
    """
    m = neural_summary(analysis)
    route = m["memory_route"]
    light_routes = {
        light: str(m["memory_routes"].get(light, "") or "")
        for light in LIGHTS
    }
    # O roteador expõe status verificado em cada luz. Na fusão o campo
    # recognition_match global normalmente é NOT_FOUND se uma luz for nova:
    # não descartar os KNOWN_KNN individuais.
    recognized = [
        light for light in LIGHTS if light_routes[light] == "KNOWN_KNN"
    ]
    unrecognized = [
        light for light in LIGHTS
        if light_routes[light] in {"NEW_CNN", "NEW_EXPERTS"}
    ]
    unreported = [
        light for light in LIGHTS
        if light_routes[light] not in {"KNOWN_KNN", "NEW_CNN", "NEW_EXPERTS"}
    ]
    if route == "KNOWN_KNN" and not recognized:
        recognized = ["INSPEÇÃO"]
    if route in {"NEW_CNN", "NEW_EXPERTS"} and not any(light_routes.values()):
        unrecognized = ["INSPEÇÃO"]
    known = bool(recognized)
    queried = known or bool(unrecognized)
    status = "JA_VI" if known else "NUNCA_VI" if queried else None
    return {
        "status": status,
        "label": "JÁ VI" if known else "NUNCA VI" if queried else "",
        "recognized_lights": recognized,
        "new_lights": unrecognized,
        "unreported_lights": unreported if any(light_routes.values()) else [],
        "routes_by_light": light_routes,
        "human_label": m["memory_label"],
        "has_multilight": any(light_routes.values()),
        "consultation_valid": queried,
        "memory_conflict": route == "MEMORY_CONFLICT",
    }


def memory_panel_text(analysis: dict | None) -> tuple[str, str]:
    memory = memory_seen_state(analysis)
    result = memory["status"]
    lights = memory["recognized_lights"]
    new = memory["new_lights"]
    if result == "JA_VI":
        if memory["has_multilight"]:
            number = len(lights)
            seen = ", ".join(lights)
            new_text = ", ".join(new)
            extra = f" • {new_text}: CNN/sem match exato" if new_text else ""
            return (
                f"JÁ VI • KNN EXATO • {number}/3 LUZES",
                f"{seen}: pares já registrados e confirmados por humano{extra}. "
                "A identificação é por par exato, não por categoria física de defeito.",
            )
        return (
            "JÁ VI • KNN EXATO",
            f"Par humano registrado • rótulo {memory['human_label'] or '?'}. "
            "Correspondência exata gabarito/teste.",
        )
    if result == "NUNCA_VI":
        if memory["has_multilight"]:
            return (
                "NUNCA VI • KNN SEM MATCH EXATO",
                "Nenhuma das iluminações consultadas tem par humano exato. "
                "Imagens analisadas pela CNN não equivalem a memória conhecida.",
            )
        return (
            "NUNCA VI • KNN SEM MATCH EXATO",
            "Par humano exato não encontrado. O ODIN pode reconhecer a "
            "categoria visual, mas não recuperou esta imagem da memória.",
        )
    if memory["memory_conflict"]:
        return (
            "CONSULTA INCONCLUSIVA • MEMÓRIA KNN",
            "Registros humanos conflitantes; conferir os rótulos antes de concluir.",
        )
    return (
        "CONSULTA KNN NÃO DISPONÍVEL",
        "Nenhuma pesquisa de correspondência exata foi confirmada nesta inspeção.",
    )
