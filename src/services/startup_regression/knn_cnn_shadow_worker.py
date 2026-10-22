"""Worker offline KNN x CNN; duas inferências históricas e um comparador."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.services.startup_regression.interactive_training_validation import (
    configure_low_priority, validate, save_report as save_cnn_report,
)
from src.services.startup_regression.knn_archive_visual_replay import (
    replay_knn_archive_visual,
)
from src.services.startup_regression.knn_archive_visual_replay_cli import (
    write_knn_archive_reports,
)
from src.services.startup_regression.knn_cnn_shadow_comparator import (
    compare, save_report as save_comparison,
)
from src.services.startup_regression.archive_inventory import inventory_archives


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="ODIN: comparar CNN com KNN visual leave-one-out (sem XP)"
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--cancel-file", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    cancel = args.cancel_file.expanduser().resolve()
    reports = (root / "reports" / "startup_regression").resolve()
    if reports not in cancel.parents or cancel.suffix != ".cancel":
        raise ValueError("Cancelamento fora de reports")
    # configure_low_priority já limita OpenCV e Torch a uma thread.
    # Não chamar _limit_torch_threads: essa função não existe no validador.
    configure_low_priority()

    def emit(data):
        print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)

    try:
        emit({"event":"phase","phase":"1/3 — inventariando os mesmos PNGs"})
        inv=inventory_archives(root)
        if cancel.exists():
            emit({"event":"canceled", "phase":"inventory"})
            return 2
        emit({"event":"phase","phase":"1/3 — verificando CNN isolada no acervo"})
        def cnn_progress(row):
            if row.get("event") == "progress":
                emit({"event":"progress","phase":"CNN","done":row.get("done"),
                      "total":row.get("total"),"path":row.get("path")})
        cnn=validate(root,inventory=inv,progress=cnn_progress,cancelled=cancel.exists)
        cnn_path=save_cnn_report(cnn,reports)
        if not cnn["complete"] or cancel.exists():
            emit({"event":"canceled","phase":"CNN","cnn_json":str(cnn_path[0])})
            return 2
        emit({"event":"phase","phase":"2/3 — KNN visual leave-one-out; sem consulta própria"})
        def knn_progress(done,total,path):
            emit({"event":"progress","phase":"KNN","done":done,
                  "total":total,"path":path})
        knn=replay_knn_archive_visual(
            root,inventory=inv,protocol="visual_holdout",
            geometry_experiment=False,
            progress=knn_progress,cancelled=cancel.exists,
        )
        knn_path=write_knn_archive_reports(knn,reports)
        if not knn["complete"] or cancel.exists():
            emit({"event":"canceled","phase":"KNN","cnn_json":str(cnn_path[0]),
                  "knn_json":str(knn_path[0])})
            return 2
        emit({"event":"phase","phase":"3/3 — comparando hashes e arbitragem experimental"})
        result=compare(cnn,knn)
        result["source_reports"]={
            "cnn":str(cnn_path[0]),"knn_visual":str(knn_path[0])
        }
        jp,tp=save_comparison(result,root)
        emit({"event":"finished","complete":True,"json_path":str(jp),
              "txt_path":str(tp),"source_reports":result["source_reports"],
              "results":result["results"],"conflicts":result["conflicts"],
              "cnn_eligible_pngs":result["cnn_eligible_pngs"]})
        return 0
    except Exception as exc:
        emit({"event":"error","message":f"{type(exc).__name__}: {exc}"})
        return 1


if __name__=="__main__":
    sys.exit(main())
