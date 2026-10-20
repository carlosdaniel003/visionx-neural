"""Worker isolado para avaliar similaridade visual KNN dos arquivos OK/NG.

Diagnóstico read-only, sem comandos XP nem promoção de reconhecimento por
aproximação para a política de Produção.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.services.startup_regression.interactive_training_validation import (
    configure_low_priority,
)
from src.services.startup_regression.knn_archive_visual_replay import (
    replay_knn_archive_visual,
)
from src.services.startup_regression.knn_archive_visual_replay_cli import (
    write_knn_archive_reports,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="KNN similaridade, replay por dia")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--cancel-file", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    cancel = args.cancel_file.expanduser().resolve()
    report_dir = (root / "reports" / "startup_regression").resolve()
    if report_dir not in cancel.parents or cancel.suffix != ".cancel":
        raise ValueError("Arquivo de cancelamento fora dos relatórios")

    configure_low_priority()

    def emit(packet):
        print(json.dumps(packet, ensure_ascii=False, allow_nan=False), flush=True)

    def progress(i, total, path):
        if i == 1:
            emit({"event": "started", "total": total, "images": total})
        emit({
            "event": "progress", "done": i, "total": total,
            "path": path,
        })

    try:
        emit({"event": "phase", "phase": "Preparando replay KNN por similaridade..."})
        report = replay_knn_archive_visual(
            root, k=5, protocol="leave_day_out",
            progress=progress, cancelled=cancel.exists,
        )
        jp, tp = write_knn_archive_reports(report, report_dir)
        emit({
            "event": "finished",
            "complete": report["complete"],
            "canceled": report["canceled"],
            "json_path": str(jp),
            "txt_path": str(tp),
            "results": report["results"],
            "by_lighting": report["by_lighting"],
            "by_capture_cohort": report["by_capture_cohort"],
            "capture_grouping": report["capture_grouping"],
        })
        return 0 if report["complete"] else 2
    except Exception as exc:
        emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
