"""Anotação interativa opcional da máscara de corpo DESLOCADO no Windows.

Use depois de preparar body_masks_review.json:
  python -m src.scripts.review_deslocado_body_masks_v3 --review "...\body_masks_review.json"

Mostra pares, permite 'a' aprovar, 'e' redesenhar, 's' pular, 'q' sair.
As marcações são de um operador, NÃO inferidas por uma CNN. Necessita
OpenCV com HighGUI; se estiver instalado headless, editar JSON manualmente.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import cv2

from src.services.deslocado_body_masks_v3 import (
    SCHEMA, _canvas, check_box, image_hash,
)
from src.scripts.train_deslocado_cnn import load_ok_events, _read, _safe_png


def _atomic_json(path: Path, row: dict) -> None:
    tmp=path.with_name(path.stem+"."+uuid4().hex+".tmp")
    tmp.write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.replace(path)


def _preview(reference,test,ref_box,test_box,mode):
    left=_canvas(reference,box=ref_box,
                 label=mode+" GABARITO (PROPOSTA)",
                 verified=False)
    right=_canvas(test,box=test_box,
                  label=mode+" TESTE (PROPOSTA)",verified=False)
    h=max(left.shape[0],right.shape[0])
    a=cv2.copyMakeBorder(left,0,h-left.shape[0],0,0,
                         cv2.BORDER_CONSTANT,value=(0,0,0))
    b=cv2.copyMakeBorder(right,0,h-right.shape[0],0,0,
                         cv2.BORDER_CONSTANT,value=(0,0,0))
    return cv2.hconcat([a,b])


def annotate_review(path: Path) -> dict:
    path=Path(path).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Revisão JSON inválida")
    content=json.loads(path.read_text(encoding="utf-8"))
    if content.get("schema")!=SCHEMA:
        raise ValueError("Formato de revisão incompatível")
    manifest=Path(content["source_manifest"]).resolve()
    if image_hash(manifest)!=content["source_manifest_sha256"]:
        raise ValueError("Manifesto alterado; não permitir revisão")
    events,data=load_ok_events(manifest)
    records=data["samples"]
    rows=content.get("rows")
    if (
        not isinstance(rows,list) or len(rows)!=len(records)
        or {r.get("source_path") for r in rows}!=set(records)
    ):
        raise ValueError("Revisão incompleta: fonte ou contagem não corresponde")
    # Somente interação guiada manual. Não atualizar dataset no ODIN.
    try:
        cv2.namedWindow("ODIN DESLOCADO V3 - REVISAO",cv2.WINDOW_NORMAL)
    except cv2.error as error:
        raise RuntimeError(
            "OpenCV sem suporte a janelas. Edite body_masks_review.json "
            "manualmente e valide com --review."
        ) from error
    approved=0
    skipped=0
    try:
        for index,row in enumerate(rows):
            if row.get("approved") is True:
                approved+=1
                continue
            source=row["source_path"]
            record=records[source]
            reference_file=_safe_png(data["run"],record["reference_path"])
            test_file=_safe_png(data["run"],record["test_path"])
            if (
                image_hash(reference_file)!=row["reference_sha256"]
                or image_hash(test_file)!=row["test_sha256"]
            ):
                raise ValueError("Par foi modificado desde a preparação: "+source)
            reference,test=_read(reference_file),_read(test_file)
            while True:
                title=f"ODIN DESLOCADO {index+1}/{len(rows)} [{row['lighting_mode']}]"
                preview=_preview(
                    reference,test,row["body_box_reference_xywh"],
                    row["body_box_test_xywh"],row["lighting_mode"],
                )
                cv2.imshow("ODIN DESLOCADO V3 - REVISAO",preview)
                print(
                    f"\n{title} | {source}\n"
                    "a=APROVAR corpo completo | e=EDITAR ROI GABARITO+TESTE | "
                    "s=PULAR | q=SALVAR E SAIR",
                    flush=True,
                )
                key=cv2.waitKey(0)&0xFF
                if key==ord("q") or key==27:
                    return {"approved":approved,"skipped":skipped,
                            "remaining":sum(r.get("approved") is not True for r in rows)}
                if key==ord("s"):
                    skipped+=1
                    break
                if key==ord("e"):
                    for view,frame,field in (
                        ("GABARITO",reference,"body_box_reference_xywh"),
                        ("TESTE",test,"body_box_test_xywh"),
                    ):
                        selected=cv2.selectROI(
                            "Selecionar corpo inteiro - "+view,
                            frame,showCrosshair=True,fromCenter=False
                        )
                        try:
                            cv2.destroyWindow("Selecionar corpo inteiro - "+view)
                        except cv2.error:
                            # Em algumas versões, selectROI fecha a janela
                            # automaticamente; não perder a anotação por isso.
                            pass
                        proposal=[int(n) for n in selected]
                        if proposal[2]>0 and proposal[3]>0:
                            try:
                                check_box(proposal,frame)
                                row[field]=proposal
                            except ValueError as exc:
                                print("CONTORNO RECUSADO:",exc,flush=True)
                    _atomic_json(path,content)
                    continue
                if key==ord("a"):
                    try:
                        check_box(row["body_box_reference_xywh"],reference)
                        check_box(row["body_box_test_xywh"],test)
                    except ValueError as exc:
                        print("NÃO APROVADO:",exc,flush=True)
                        continue
                    print(
                        "CONFIRMAÇÃO: área engloba o corpo FÍSICO E TERMINAIS,"
                        " sem incluir pads fixos? [s=sim / n=não]",
                        flush=True
                    )
                    confirm=cv2.waitKey(0)&0xFF
                    if confirm!=ord("s"):
                        print("Não aprovado; ajuste o contorno.",flush=True)
                        continue
                    row["approved"]=True
                    row["review_notes"]=(
                        "Inspecionado e confirmado interativamente pelo "
                        "operador: corpo completo + terminais, pads fixos fora."
                    )
                    _atomic_json(path,content)
                    approved+=1
                    break
    finally:
        cv2.destroyAllWindows()
    return {"approved":approved,"skipped":skipped,
            "remaining":sum(r.get("approved") is not True for r in rows)}


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(
        description="Marcar e confirmar ROIs do corpo DESLOCADO visualmente"
    )
    parser.add_argument("--review",type=Path,required=True)
    args=parser.parse_args(argv)
    report=annotate_review(args.review)
    print("Revisão manual parcial/concluída:",report)
    print(
        "Próxima validação: python -m src.scripts.prepare_deslocado_body_masks_v3 "
        '--review "'+str(args.review)+'"'
    )
    return 0


if __name__=="__main__":
    raise SystemExit(main())
