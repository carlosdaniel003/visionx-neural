"""CLI isolada: replay de eventos com manifesto, nunca envia comandos XP."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from src.services.startup_regression.operational_event_replay import (
    replay, save_report,
)


def _low_priority():
    from src.services.startup_regression.interactive_training_validation import (
        configure_low_priority,
    )
    configure_low_priority()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay operacional read-only (event_id + SHA verificados)"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--cancel-file", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    cancel = args.cancel_file

    def emit(row):
        print(json.dumps(row, ensure_ascii=False), flush=True)

    try:
        _low_priority()
        import cv2
        cv2.setNumThreads(1)
        from src.services.startup_regression.interactive_training_validation import (
            _limit_torch_threads,
        )
        _limit_torch_threads()
    except (ImportError, AttributeError):
        pass

    try:
        report = replay(
            root,
            progress=emit,
            cancelled=lambda: bool(cancel and cancel.is_file()),
        )
        paths = save_report(report, root / "reports" / "startup_regression")
        emit({
            "event": "finished", "complete": report["complete"],
            "canceled": report["canceled"], "counts": report["counts"],
            "coverage": report["coverage"],
            "json_path": str(paths[0]), "txt_path": str(paths[1]),
        })
        return 0 if report["complete"] else 2
    except Exception as exc:
        emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
