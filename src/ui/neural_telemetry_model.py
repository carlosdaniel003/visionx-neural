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
    """Snapshot auditável: V3 é rotulada V3, sem inventar estado V2."""
    out = {}
    for light in LIGHTS:
        frame = _map(_map(analyses).get(light))
        d = _map(frame.get("detail"))
        v3 = d.get("cnn_v3_primary") is True
        prefix = "cnn_v3" if v3 else "cnn_v2"
        cnn = bool(
            v3 or d.get("cnn_v2_active")
            or "faltando_cnn_v2.py" in (frame.get("active_engines") or [])
        )
        out[light] = {
            "route": _route(d) or ("NEW_CNN" if cnn else ""),
            "engine": ("aoi_quality_cnn_v3" if v3 else
                       "faltando_cnn_v2" if cnn else
                       str(d.get("dominant_engine", "") or "")),
            "cnn_active": cnn,
            "cnn_status": str(d.get(f"{prefix}_status", "") or ""),
            "checkpoint_verified": (
                d.get(f"{prefix}_checkpoint_verified") is True if cnn else None
            ),
            "checkpoint_sha256": (
                str(d.get(f"{prefix}_checkpoint_sha256", "") or "") if cnn else ""
            ),
            "ng_score_uncalibrated": (
                _finite_score(d.get(f"{prefix}_ng_score_uncalibrated"))
                if cnn else None
            ),
            "verdict": str(frame.get("verdict", "") or ""),
            "category": str(
                d.get("cnn_v2_aoi_category") or
                d.get("multilight_category") or ""
            ),
            "human_memory_label": str(d.get("recognition_known_label", "") or ""),
        }
    return out

def neural_summary(analysis: dict | None) -> dict:
    frame = _map(analysis)
    d = _map(frame.get("detail"))
    trace = _map(d.get("decision_trace"))
    route = _route(d)
    per_light = _map(
        d.get("cnn_v3_light_diagnostics")
        or d.get("cnn_v2_light_diagnostics")
    )
    v3 = bool(
        d.get("cnn_v3_primary") is True
        or any(_map(x).get("engine") == "aoi_quality_cnn_v3"
               for x in per_light.values())
    )
    prefix = "cnn_v3" if v3 else "cnn_v2"
    has_cnn = bool(
        v3 or d.get("cnn_v2_active")
        or "faltando_cnn_v2.py" in (frame.get("active_engines") or [])
        or route == "NEW_CNN"
        or any(_map(x).get("cnn_active") for x in per_light.values())
    )
    local_ng = _finite_score(d.get(f"{prefix}_ng_score_uncalibrated"))
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
        "cnn_model_name": "CNN AOI V3" if v3 else "CNN FALTANDO v2",
        "cnn_route": route,
        "cnn_ng_score": local_ng,
        "cnn_ok_score": (1.0 - local_ng) if local_ng is not None else None,
        "cnn_status": str(d.get(f"{prefix}_status", "") or ""),
        "cnn_checkpoint_verified": d.get(f"{prefix}_checkpoint_verified") is True,
        "cnn_checkpoint_sha256": str(d.get(f"{prefix}_checkpoint_sha256", "") or ""),
        "cnn_best_epoch": 3 if v3 else d.get("cnn_v2_checkpoint_best_epoch"),
        "cnn_category": str(d.get("cnn_v2_aoi_category") or d.get("multilight_category") or ""),
        "cnn_lighting_mode": str(d.get(f"{prefix}_lighting_mode") or frame.get("lighting_mode") or ""),
        "cnn_votes": (
            {light: _map(row).get("predicted", "") for light, row in
             _map(_map(d.get("aoi_v3_multilight")).get("lights")).items()}
            if v3 else _map(d.get("cnn_v2_light_votes"))
        ),
        "cnn_per_light": lights,
        "cnn_consensus": (
            str(_map(d.get("aoi_v3_multilight")).get("status", "") or "")
            if v3 else str(d.get("cnn_v2_consensus", "") or "")
        ),
        "cnn_consensus_reason": (
            "V3 experimental; voto por iluminação, sem auto-OK isolado"
            if v3 else str(d.get("cnn_v2_consensus_reason", "") or "")
        ),
        "cnn_auto_eligible": (
            False if v3 else d.get("cnn_v2_supervised_auto_eligible") is True
        ),
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
        return {"header": "CNN • não executada",
                "lines": ("O julgamento pertence a outro motor.",)}
    label = m["cnn_lighting_mode"] or "MULTILIGHT"
    sha = m["cnn_checkpoint_sha256"]
    ck = "VERIFICADO" if m["cnn_checkpoint_verified"] else "NÃO VERIFICADO"
    lines = [
        f"Categoria AOI: {m['cnn_category'] or 'não informada'}",
        f"Motor: {m['cnn_model_name']} • {label}",
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
    return {"header": m["cnn_model_name"] + " • REDE NEURAL", "lines": tuple(lines)}



def memory_seen_state(analysis: dict | None) -> dict:
    """Distingue memória KNN exata, visual semelhante e sem match.

    JÁ VI exige um par humano exato; VI SEMELHANTE exige corroboração
    KNN visual. Não confundir nenhuma dessas memórias com pesos da CNN.
    Sem consulta verificada, não declarar que a peça nunca foi vista.
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
    detail = ((analysis or {}).get("detail") or {}) if isinstance(analysis, dict) else {}
    detail = detail if isinstance(detail, dict) else {}
    visual = detail.get("knn_visual_corroboration") or {}
    visual = visual if isinstance(visual, dict) else {}
    rows = visual.get("lights") or {}
    rows = rows if isinstance(rows, dict) else {}
    status_visual = visual.get("status")
    # Não confiar apenas no texto 'NG_CORROBORATED': o resumo precisa
    # indicar as luzes concretas que efetivamente recuperaram vizinhos.
    strong = visual.get("strong_ng_lights")
    trend = visual.get("trend_ng_lights")
    strong = strong if isinstance(strong, list) else []
    trend = trend if isinstance(trend, list) else []
    approximate_lights = [
        light for light in LIGHTS
        if (
            (light in strong and status_visual == "NG_CORROBORATED"
             and _map(rows.get(light)).get("status") == "NG_SIMILAR")
            or (light in trend and status_visual == "NG_TREND_THREE_LIGHTS"
                and _map(rows.get(light)).get("status") in {"NG_LEANING", "NG_SIMILAR"})
        )
    ]
    human_visual = detail.get("confirmed_visual_recall")
    human_visual = human_visual if isinstance(human_visual, dict) else {}
    human_visual_status = str(human_visual.get("status", "") or "")
    human_visual_valid = bool(
        human_visual.get("schema") == "visionx.human_visual_recall.v1"
        and human_visual.get("source") == "PERSISTENT_HUMAN_LABELS"
        and human_visual.get("diagnostic_only") is True
        and human_visual.get("auto_ok_authorized") is False
    )
    human_visual_label = (
        "OK" if human_visual_valid
        and human_visual_status == "HUMAN_OK_VISUAL_CANDIDATE" else
        "NG" if human_visual_valid
        and human_visual_status == "HUMAN_NG_VISUAL_CANDIDATE" else ""
    )
    archive = detail.get("side_archive_visual_presence")
    archive = archive if isinstance(archive, dict) else {}
    archive_status = str(archive.get("status", "") or "")
    rendered = _map(archive.get("rendered_archive_match"))
    rendered_seen = bool(
        rendered.get("schema") == "visionx.archive_rendered_source_match.v1"
        and rendered.get("status") == "RENDERED_ARCHIVE_SOURCE"
        and rendered.get("label") in {"OK", "NG"}
        and rendered.get("diagnostic_only") is True
        and rendered.get("decision_authorized") is False
        and rendered.get("exact_aoi_pair") is False
        and rendered.get("frame_pixel_identical") is False
        and isinstance(rendered.get("path"), str)
        and rendered.get("path")
        and isinstance(rendered.get("evidence"), dict)
    )
    # O worker consulta SIDE/TOP/MID separadamente, mas a SIDE pode nao
    # ter encontrado vizinho enquanto TOP ou MID recuperou uma captura.
    # A UI agrega apenas fontes diagnósticas com o contrato verificável.
    by_light = archive.get("by_light")
    multilight_archive = isinstance(by_light, dict)
    if not multilight_archive:
        by_light = {"SIDE": archive} if archive else {}
    archive_rows = {
        light: row for light, row in by_light.items()
        if light in LIGHTS and isinstance(row, dict)
        and row.get("diagnostic_only") is True
        and row.get("decision_authorized") is False
        and (
            not multilight_archive or (
                row.get("schema") == "visionx.side_archive_visual_presence.v1"
                and row.get("source") == "PUBLIC_ARCHIVE_PNG"
            )
        )
    }
    similar_frames = {
        light: row for light, row in archive_rows.items()
        if row.get("status") == "SIMILAR" and row.get("label") in {"OK", "NG"}
    }
    similar_labels = {row["label"] for row in similar_frames.values()}
    # O arquivo PNG guarda a tela inteira, enquanto KNOWN_KNN
    # compara os recortes gabarito/teste. Identidade do frame é uma
    # prova de conteúdo distinta, nunca equivale a par AOI exato.
    exact_frames = {
        light: row for light, row in by_light.items()
        if light in LIGHTS
        and isinstance(row, dict)
        and row.get("schema") == "visionx.side_archive_visual_presence.v1"
        and row.get("source") == "PUBLIC_ARCHIVE_PNG"
        and row.get("status") == "EXACT_FRAME"
        and row.get("label") in {"OK", "NG"}
        and row.get("diagnostic_only") is True
        and row.get("decision_authorized") is False
        and isinstance(row.get("exact_frame_sha256"), str)
        and len(row["exact_frame_sha256"]) == 64
        and row.get("exact_frame_labels") == [row["label"]]
    }
    exact_conflict = bool(
        any(row.get("status") == "EXACT_FRAME_CONFLICT"
            for row in by_light.values() if isinstance(row, dict))
        or len({row["label"] for row in exact_frames.values()}) > 1
    )
    archive_conflict = bool(
        exact_conflict
        or any(row.get("status") == "AMBIGUOUS"
               for row in archive_rows.values())
        or len(similar_labels) > 1
        or len(similar_labels | {
            row["label"] for row in exact_frames.values()
        }) > 1
        or rendered.get("status") == "CONFLICT"
        or (rendered_seen and bool(
            (similar_labels | {row["label"] for row in exact_frames.values()})
            - {rendered["label"]}
        ))
    )
    archive_matched = bool(similar_frames and not archive_conflict)
    archive_label = next(iter(similar_labels)) if archive_matched else ""
    archive_lights = [
        light for light in LIGHTS if light in similar_frames
    ] if archive_matched else []
    archive_exact_seen = bool(exact_frames) and not archive_conflict
    archive_rendered_seen = rendered_seen and not archive_conflict
    exact_human = detail.get("operator_latest_exact_frame")
    exact_human = exact_human if isinstance(exact_human, dict) else {}
    exact_human_seen = bool(
        detail.get("fusion_rule") in {
            "operator_latest_exact_frame_triplet_v1",
            "operator_latest_exact_frame_side_v1",
        }
        and exact_human.get("schema") == "visionx.operator_latest_exact_frame.v1"
        and exact_human.get("label") in {"OK", "NG"}
        and exact_human.get("integrity")
        and isinstance(exact_human.get("matches"), dict)
        and set(exact_human["matches"]) in (set(LIGHTS), {"SIDE"})
    )

    # A memória AUTO é distinta da KNN humana. A prova da rota SIDE
    # já selecionada é checada localmente; não há I/O durante a pintura
    # nem qualquer mudança na autorização de comandos XP.
    auto_side = None
    if detail.get("fusion_rule") == "production_auto_side_exact_recall_v1":
        from src.core.production_auto_side_recall import (
            auto_side_display_provenance,
        )
        auto_side = auto_side_display_provenance(analysis)
    auto_exact_seen = bool(
        auto_side
        and not known
        and not archive_exact_seen
        and not exact_human_seen
        and not archive_conflict
    )

    similar_ng = bool(
        not known and (
            (status_visual == "NG_CORROBORATED" and len(approximate_lights) >= 2)
            or (status_visual == "NG_TREND_THREE_LIGHTS"
                and len(approximate_lights) == 3)
        )
    )
    # The multilight classifier can recover an operator-labelled OK source
    # even when the exact-pixel KNN route remains NEW_CNN. Revalidate the
    # complete evidence through the production policy; a bare marker, filename
    # or KNN similarity alone must never make the UI claim "seen OK".
    historical_ok = False
    physical_veto = _map(detail.get("physical_missing_veto"))
    if (
        _map(detail.get("recognized_top_mid_ok_20261029"))
        and detail.get("fusion_rule")
        == "operator_known_top_mid_ok_review_20261029_v1"
        and _map(analysis).get("verdict") == "FALHA FALSA"
    ):
        from src.ui.production_confidence_gate import production_decision_policy
        recalled_policy = production_decision_policy(analysis)
        historical_ok = bool(
            recalled_policy.get("verdict") == "FALHA FALSA"
            and recalled_policy.get("reason") in {
                "supervised_operator_known_top_mid_ok_20261029_auto_v1",
                "operator_known_top_mid_ok_20261029_requires_operator",
            } or (
                recalled_policy.get("verdict") == "FALHA FALSA"
                and str(recalled_policy.get("reason", "")).startswith(
                    "known_top_mid_ok_physical_veto_"
                )
            )
        )
    status = (
        "JA_VI" if known or archive_exact_seen or exact_human_seen or archive_rendered_seen else
        "ACERVO_AMBIGUO" if archive_conflict else
        "JA_VI" if auto_exact_seen else
        "VI_SEMELHANTE" if similar_ng or archive_matched or historical_ok or human_visual_label else
        "CONSULTANDO_ACERVO" if archive_status == "PENDING" else
        "ACERVO_AMBIGUO" if archive_status == "AMBIGUOUS"
            or (human_visual_valid and human_visual_status == "AMBIGUOUS") else
        "NUNCA_VI" if queried else None
    )
    return {
        "status": status,
        "label": (
            "JÁ VI" if known or archive_exact_seen or exact_human_seen or archive_rendered_seen else
            "ACERVO AMBÍGUO" if archive_conflict else
            "JÁ VI" if auto_exact_seen else
            "VI SEMELHANTE" if similar_ng or archive_matched or historical_ok or human_visual_label else
            "CONSULTANDO ACERVO" if archive_status == "PENDING" else
            "ACERVO AMBÍGUO" if archive_status == "AMBIGUOUS"
                or (human_visual_valid and human_visual_status == "AMBIGUOUS") else
            "SEM PAR EXATO" if queried else ""
        ),
        "human_visual_seen": bool(human_visual_label),
        "human_visual_label": human_visual_label,
        "human_visual_similarity": (
            human_visual.get("best_ok_similarity") if human_visual_label == "OK"
            else human_visual.get("best_ng_similarity")
        ) if human_visual_label else None,
        "human_visual_cross_light": human_visual.get("cross_light_candidate") is True,
        "archive_visual_seen": archive_matched or historical_ok,
        "archive_rendered_source_seen": archive_rendered_seen,
        "archive_rendered_source_label": rendered.get("label") if archive_rendered_seen else "",
        "archive_rendered_source_path": rendered.get("path") if archive_rendered_seen else "",
        "cnn_model_name": m["cnn_model_name"],
        "archive_label": "OK" if historical_ok else archive_label if archive_matched else "",
        "archive_visual_lights": ["TOP", "MID"] if historical_ok else archive_lights,
        "historical_ok_recall": historical_ok,
        "physical_veto_status": str(physical_veto.get("status", "") or "") if historical_ok else "",
        "archive_visual_status": archive_status,
        "approximate_ng_seen": similar_ng,
        "approximate_ng_lights": approximate_lights if similar_ng else [],
        "visual_rows": rows if similar_ng else {},
        "exact_pair_seen": known,
        "auto_side_exact_seen": auto_exact_seen,
        "auto_side_label": auto_side["label"] if auto_exact_seen else "",
        "auto_side_analyzed_lights": (
            auto_side["analyzed_lights"] if auto_exact_seen else ""
        ),
        "exact_human_frame_seen": exact_human_seen,
        "exact_human_frame_label": exact_human.get("label", "") if exact_human_seen else "",
        "archive_exact_frame_seen": archive_exact_seen,
        "archive_exact_frame_lights": sorted(exact_frames),
        "archive_exact_frame_label": (
            next(iter({row["label"] for row in exact_frames.values()}))
            if archive_exact_seen else ""
        ),
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
        if memory["exact_human_frame_seen"] and not memory["exact_pair_seen"]:
            return (
                "JÁ VI • FRAME(S) EXATO(S) • HUMANO",
                f"Última confirmação humana: {memory['exact_human_frame_label']}. "
                "A(s) tela(s) XP consultada(s) são pixel-idênticas "
                "às de uma correção anterior. O par AOI recortado "
                "é uma identidade separada.",
            )
        if memory["archive_exact_frame_seen"] and not memory["exact_pair_seen"]:
            lights = ", ".join(memory["archive_exact_frame_lights"])
            label = memory["archive_exact_frame_label"]
            return (
                f"JÁ VI • FRAME EXATO NO ACERVO {label}",
                f"Tela XP pixel-idêntica encontrada no arquivo {label} "
                f"em {lights}. Isso NÃO prova que os dois recortes AOI "
                "tenham hashes iguais e não autoriza um comando 0/1. "
                "A memória humana exata é uma consulta separada.",
            )
        if memory["archive_rendered_source_seen"] and not memory["exact_pair_seen"]:
            label = memory["archive_rendered_source_label"]
            path = memory["archive_rendered_source_path"]
            return (
                f"JÁ VI • ARQUIVO REEXIBIDO {label}",
                f"Origem visual registrada: {path}. A imagem da captura foi "
                "alinhada ao arquivo mesmo com redimensionamento da janela. "
                "Não é igualdade dos pixels brutos nem do par AOI; o rótulo "
                "da pasta não substitui uma classificação física verificada.",
            )
        if memory["auto_side_exact_seen"]:
            label = memory["auto_side_label"]
            lights_before = memory["auto_side_analyzed_lights"]
            source = (
                f" A ocorrência anterior passou por {lights_before.replace(',', ', ')}."
                if lights_before else ""
            )
            return (
                f"JÁ VI • DECISÃO AUTOMÁTICA {label} • SIDE",
                f"Mesmo par AOI da SIDE, pixel-idêntico, recuperado da "
                f"memória operacional automática como {label}.{source} "
                "TOP/MID dispensadas nesta inspeção. Não é confirmação "
                "humana nem treinamento CNN/KNN; o envio XP permanece "
                "dependente da revalidação da política de Produção.",
            )
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
            f"Par humano registrado • Rótulo {memory['human_label'] or '?'}. "
            "Correspondência exata gabarito/teste.",
        )
    if result == "VI_SEMELHANTE" and memory["historical_ok_recall"]:
        physical = memory["physical_veto_status"]
        suffix = (
            " A captura atual permanece bloqueada por contraprova física "
            + physical + "; não há autorização de envio XP 0/OK."
            if physical in {
                "SUSPECT_ABSENCE", "UNAVAILABLE", "GEOMETRY_REVIEW",
                "NG_CORROBORATED",
            } else ""
        )
        return (
            "VI SEMELHANTE • MEMÓRIA OK • TOP/MID",
            "O julgamento visual já registrado como OK no ok_archive foi "
            "recuperado e revalidado pelos descritores A/B de TOP/MID. "
            "Não houve igualdade pixel a pixel com o par atual."
            + suffix,
        )
    if result == "VI_SEMELHANTE" and memory["human_visual_seen"]:
        label = memory["human_visual_label"]
        score = memory["human_visual_similarity"]
        score_text = (
            f"{float(score) * 100:.2f}%" if isinstance(score, (int, float))
            else "sem score"
        )
        return (
            f"VI SEMELHANTE • JULGAMENTO HUMANO {label}",
            f"A memória JSON recuperou uma assinatura de julgamento humano {label} "
            f"({score_text}), sem par RGB exato. A semelhança é uma hipótese, "
            "não autoriza 0/OK automático; verificação multilight e física "
            "seguem necessárias.",
        )
    if result == "VI_SEMELHANTE" and memory["archive_visual_seen"]:
        label = memory["archive_label"]
        return (
            f"VI SEMELHANTE • ARQUIVO {label} • "
            + ", ".join(memory["archive_visual_lights"]),
            "Uma captura XP visualmente semelhante foi localizada em "
            f"public/{'ng_archive' if label == 'NG' else 'ok_archive'} "
            + "na(s) iluminação(ões) "
            + ", ".join(memory["archive_visual_lights"]) + ". "
            "É um rótulo de arquivo, não confirmação física ou par exato. "
            "Esta consulta não altera a CNN, o julgamento nem o comando XP.",
        )
    if result == "CONSULTANDO_ACERVO":
        return (
            "CONSULTANDO ACERVO • SIDE",
            "A comparação visual com os PNGs OK/NG está em andamento, "
            "em segundo plano; o julgamento não espera pelo resultado.",
        )
    if result == "ACERVO_AMBIGUO":
        return (
            "ACERVO AMBÍGUO • OK E NG",
            "Capturas visualmente semelhantes ou exatas apontam para "
            "OK e NG entre as iluminações, ou são ambíguas na mesma luz. "
            "Não atribuir classe confiável a partir desses rótulos. "
            "O julgamento da CNN permanece independente.",
        )
    if result == "VI_SEMELHANTE":
        lights = memory["approximate_ng_lights"]
        summaries = []
        for light in lights:
            row = _map(memory["visual_rows"].get(light))
            ng = _map(row.get("ng"))
            best = _finite_score(ng.get("best_similarity"))
            count = row.get("ng_support")
            count_str = f" ({count} NG)" if isinstance(count, int) and count > 0 else ""
            score_str = f" {best * 100:.2f}%" if best is not None else ""
            summaries.append(f"{light}{score_str}{count_str}")
        return (
            f"VI SEMELHANTE • KNN VISUAL NG • {len(lights)}/3 LUZES",
            " | ".join(summaries)
            + ". São exemplos visuais, não pares exatos nem prova de "
              "retreinamento da CNN. Não autorizam 0/1 automático.",
        )
    if result == "NUNCA_VI":
        if memory["has_multilight"]:
            return (
                "SEM PAR EXATO • KNN",
                "Nenhuma das iluminações consultadas tem par humano exato nem "
                "vizinhos NG corroborados nesta inspeção. A CNN é um modelo "
                "separado da memória KNN. Isso não significa que o tipo "
                "de defeito nunca foi registrado.",
            )
        return (
            "SEM PAR EXATO • KNN",
            "KNN consultado: par humano exato não encontrado nem vizinhos NG "
            "corroborados. Isso não significa que o tipo de defeito nunca "
            "tenha sido salvo.",
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
