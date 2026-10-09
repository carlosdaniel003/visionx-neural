"""Reanalisa todos os screenshots históricos usando SOMENTE as CNNs.

    python -m src.services.startup_regression.cnn_full_history_replay_cli

Não permite apontar saída ao dataset. Não treina, não ativa gate e não
aceita que a taxa 98% inclua revisão/adesivo sem CNN como acertos.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.services.startup_regression.archive_inventory_report import _atomic_text
from .cnn_full_history_replay import replay_full_cnn_history


def write_full_history_cnn_report(report: dict, output_dir: Path):
    root = Path(report["root"]).resolve()
    out = Path(output_dir).resolve()
    reports_root = (root / "reports").resolve()
    if out == reports_root or reports_root not in out.parents:
        raise ValueError("Saída permitida apenas dentro de reports/ em subpasta")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    json_path = out / f"cnn_full_history_{stamp}.json"
    txt_path = out / f"cnn_full_history_{stamp}.txt"
    _atomic_text(json_path, json.dumps(
        report, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n")
    overall = report["overall"]
    lines = [
        "ODIN — REPLAY CNN DE TODO ACERVO VISUAL OK/NG",
        "CNN SOMENTE | SEM KNN | SEM TREINO | SEM GATE",
        f"Meta histórica (todos os PNGs): {report['expected_retention_rate']:.0%}",
        f"PNGs inventariados: {report['inventory_png_count']}",
        f"Elegíveis à CNN existente: {overall['cnn_scoped']}",
        f"Sem CNN para categoria: {overall['unsupported']}",
        f"Acertos efetivos OK/NG: {overall['passed']}/{overall['total']}",
        f"Retenção de todo o acervo: {overall['historical_full_archive_match_rate']}",
        f"Retenção no escopo da CNN: {overall['historical_supported_match_rate']}",
        f"Acerto binário bruto (não é decisão operacional): "
        f"{overall['raw_binary_correct']}/{overall['total']}",
        f"Revisões obrigatórias: {overall['review']}",
        f"Regressões: {overall['regression']}",
        f"Inválidos: {overall['invalid']}",
        f"NG históricos cobertos pela CNN: {overall['expected_NG_scoped']}",
        f"NG identificados como NG: {overall['NG_as_NG']}",
        f"NG classificados incorretamente OK: {overall['NG_as_OK']}",
        f"NG em revisão: {overall['NG_to_review']}",
        f"OK históricos cobertos pela CNN: {overall['expected_OK_scoped']}",
        f"OK classificados OK: {overall['OK_as_OK']}",
        f"OK classificados incorretamente NG: {overall['OK_as_NG']}",
        f"OK em revisão: {overall['OK_to_review']}",
        f"Meta 98% (COM cobertura completa): {overall['historical_98pct_target_met']}",
        "",
        "POR CATEGORIA:",
    ]
    for category, counts in report["by_category"].items():
        lines.append(
            f"  {category}: PASSOU={counts['passed']}/{counts['total']} "
            f"revisao={counts['review']} regressao={counts['regression']} "
            f"invalido={counts['invalid']} sem_cnn={counts['unsupported']} "
            f"NG_certos={counts['NG_as_NG']}/{counts['expected_NG_scoped']} "
            f"NG_como_OK={counts['NG_as_OK']}"
        )
    lines.append("")
    lines.append("POR ILUMINAÇÃO:")
    for mode, counts in report["by_lighting"].items():
        lines.append(
            f"  {mode}: {counts['passed']}/{counts['total']}, "
            f"revisao={counts['review']}, invalido={counts['invalid']}, "
            f"sem_cnn={counts['unsupported']}"
        )
    groups = report["multilight_explicit_events"]
    lines.extend([
        "",
        "EVENTOS TOP/MID/SIDE COM MANIFESTO EXPLÍCITO:",
        f"  Total: {groups['events_with_explicit_manifest']}",
        f"  Status: {groups['status_counts']}",
        "",
        "CASOS QUE NÃO PASSARAM:",
    ])
    for case in report["cases"]:
        if case["status"] == "PASSOU":
            continue
        lines.append(
            f"- {case['status']} | {case['source_path']} | "
            f"esperado={case['expected_label']} | "
            f"CNN={case['model'] or 'SEM MODELO'} | "
            f"resposta={case['verdict']} | scoreNG={case['ng_score_uncalibrated']} "
            f"| {case['error'] or ''}"
        )
    lines.extend(["", "LIMITAÇÃO:", report["note"]])
    _atomic_text(txt_path, "\n".join(lines) + "\n")
    return json_path, txt_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay somente CNN em todo o histórico disponível, meta 98%"
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    print(
        "Replay completo das CNNs, sem memória KNN, sem treinamento...",
        flush=True,
    )

    def progress(n, total, path):
        if n == 1 or n % 25 == 0 or n == total:
            print(f"[{n}/{total}] {path}", flush=True)

    report = replay_full_cnn_history(root, progress=progress)
    dest = root / "reports" / "startup_regression"
    j, t = write_full_history_cnn_report(report, dest)
    overall = report["overall"]
    print(
        f"Histórico: {overall['passed']}/{overall['total']} OK/NG corretos; "
        f"escopo CNN={overall['cnn_scoped']}; "
        f"sem CNN={overall['unsupported']}; "
        f"revisão={overall['review']}; regressões={overall['regression']}",
        flush=True,
    )
    print(
        f"NG detectados: {overall['NG_as_NG']}/{overall['expected_NG_scoped']} "
        f"| NG->OK indevidos={overall['NG_as_OK']}",
        flush=True,
    )
    print(f"Meta 98% histórica atingida: {overall['historical_98pct_target_met']}", flush=True)
    print(f"JSON: {j}\nTXT: {t}", flush=True)
    print("Nenhuma CNN/KNN/dataset foi alterada; gate não ativado.", flush=True)
    # Regressão não provoca exclusão automática e sempre gera relatório.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
