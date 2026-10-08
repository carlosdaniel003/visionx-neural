"""DESLOCADO v3 — anotação auditável do CORPO INTEIRO (não letras).

Fase A: gerar previsualizações de referência/teste e um formulário JSON,
sem treinamento, sem edição dos PNG originais e sem modificar Produção.
Fase B: validar todas as caixas anotadas/aprovadas por humano, gerando
previews finais e catálogo somente de ROIs. Nenhuma promoção automática.

As caixas INCLUEM corpo + terminais móveis; EXCLUEM pads fixos da PCB.
A sugestão geométrica inicial NÃO é detecção: 'approved' começa False.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

import cv2
import numpy as np

from src.scripts.train_deslocado_cnn import load_ok_events, _read, _safe_png

SCHEMA = "visionx.deslocado_body_mask_review.v1"
RESULT_SCHEMA = "visionx.deslocado_body_mask_validation.v1"
LIGHTS = ("SIDE", "TOP", "MID")


def image_hash(file: Path) -> str:
    return sha256(file.read_bytes()).hexdigest()


def propose_box(frame: np.ndarray) -> list[int]:
    """Somente moldura inicial para anotação, JAMAIS segmentação aceita.

    Centralização é hipótese visual grosseira, propositalmente aprovada
    apenas após inspeção humana; nenhuma propriedade cromática usa texto.
    """
    h, w = frame.shape[:2]
    return [
        round(w * .13), round(h * .12),
        round(w * .74), round(h * .76)
    ]


def check_box(value, frame: np.ndarray) -> tuple[int, int, int, int]:
    """Bloqueia caixas pequenas (inscrições) e caixas fora da peça."""
    if (
        not isinstance(value, list) or len(value) != 4
        or any(type(v) is not int for v in value)
    ):
        raise ValueError("body_box_xywh precisa conter quatro inteiros [x,y,w,h]")
    x, y, w, h = value
    height, width = frame.shape[:2]
    if (
        x < 0 or y < 0 or w <= 0 or h <= 0
        or x+w > width or y+h > height
    ):
        raise ValueError("Contorno ultrapassa a imagem ou é vazio")
    # Rejeitar caixas minúsculas de '104' ou outros caracteres centrais.
    if (
        w < max(12, round(width*.32))
        or h < max(12, round(height*.32))
        or w*h < width*height*.14
        or w*h > width*height*.88
    ):
        raise ValueError(
            "Contorno pequeno/grande demais para hipótese de CORPO; "
            "evite marcar apenas a inscrição ou quase toda a PCB"
        )
    cx = (x+w*.5)/width
    cy = (y+h*.5)/height
    if abs(cx-.5) > .22 or abs(cy-.5) > .22:
        raise ValueError("Corpo muito fora do centro do crop AOI; revisão manual")
    return x,y,w,h


def _canvas(frame: np.ndarray, *, box: list[int],
            label: str, verified: bool) -> np.ndarray:
    """Preserva pixels e mostra grade em coordenadas de imagem ORIGINAL."""
    h, w = frame.shape[:2]
    zoom = min(3., 540/max(w, 1), 420/max(h, 1))
    zoom = max(.5, zoom)
    new_w, new_h = max(1, round(w*zoom)), max(1, round(h*zoom))
    base = cv2.resize(frame, (new_w,new_h), interpolation=cv2.INTER_NEAREST)
    color = (20,200,40) if verified else (0,170,255)
    x,y,bw,bh = box
    cv2.rectangle(base, (round(x*zoom),round(y*zoom)),
                  (round((x+bw)*zoom),round((y+bh)*zoom)),color,2)
    # Grade e coordenadas, sem usar os números internos do componente.
    for xx in range(0,w,max(10,round(w/8))):
        X=round(xx*zoom)
        cv2.line(base,(X,0),(X,new_h-1),(80,80,80),1)
        cv2.putText(base,str(xx),(X+2,14),cv2.FONT_HERSHEY_SIMPLEX,
                    .35,(220,220,220),1,cv2.LINE_AA)
    for yy in range(0,h,max(10,round(h/8))):
        Y=round(yy*zoom)
        cv2.line(base,(0,Y),(new_w-1,Y),(80,80,80),1)
        cv2.putText(base,str(yy),(2,max(14,Y-2)),cv2.FONT_HERSHEY_SIMPLEX,
                    .35,(220,220,220),1,cv2.LINE_AA)
    final = np.zeros((new_h+39,new_w,3),np.uint8)
    final[39:,:,:]=base
    cv2.putText(final,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,
                .5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(final, f"box=[{x},{y},{bw},{bh}]",(5,34),
                cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
    return final


def _write_preview(output: Path, reference: np.ndarray, test: np.ndarray,
                   reference_box: list[int], test_box: list[int],
                   title: str, *, verified: bool) -> None:
    first=_canvas(reference,box=reference_box,label=title+" GABARITO",
                  verified=verified)
    second=_canvas(test,box=test_box,label=title+" TESTE",
                   verified=verified)
    height=max(first.shape[0],second.shape[0])
    left=cv2.copyMakeBorder(first,0,height-first.shape[0],0,0,
                            cv2.BORDER_CONSTANT,value=(0,0,0))
    right=cv2.copyMakeBorder(second,0,height-second.shape[0],0,0,
                             cv2.BORDER_CONSTANT,value=(0,0,0))
    canvas=cv2.hconcat([left,right])
    if not cv2.imwrite(str(output),canvas):
        raise OSError("Não foi possível salvar preview do contorno")


def _rows(manifest: Path) -> tuple[list[dict],dict]:
    events,data=load_ok_events(manifest)
    if data["warnings"]:
        raise ValueError(
            "Há agrupamentos multilight incertos: "+repr(data["warnings"])
        )
    rows=[]
    for event in events:
        for light,source in event["observations"].items():
            entry=data["samples"][source]
            a=_safe_png(data["run"],entry["reference_path"])
            b=_safe_png(data["run"],entry["test_path"])
            reference, test=_read(a),_read(b)
            rows.append({
                "source_path":source,
                "event_id":event["id"],
                "lighting_mode":light,
                "reference_path":entry["reference_path"],
                "test_path":entry["test_path"],
                "reference_sha256":image_hash(a),
                "test_sha256":image_hash(b),
                "reference_size_wh":[reference.shape[1],reference.shape[0]],
                "test_size_wh":[test.shape[1],test.shape[0]],
                "body_box_reference_xywh":propose_box(reference),
                "body_box_test_xywh":propose_box(test),
                "approved":False,
                "review_notes":"",
                "status":"REVIEW_REQUIRED",
            })
    if len(rows)!=len(data["samples"]):
        raise ValueError("Não é possível perder imagens durante a anotação")
    return rows,data


def _new_folder(root: Path, label: str) -> Path:
    when=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output=root/"reports"/"deslocado_neural"/"body_masks"/(label+"_"+when)
    output.mkdir(parents=True,exist_ok=False)
    return output


def prepare_body_masks(manifest: Path) -> tuple[dict,Path]:
    manifest=Path(manifest).expanduser().resolve()
    rows,data=_rows(manifest)
    out=_new_folder(data["root"],"review")
    previews=out/"preview_proposals"
    previews.mkdir()
    for index,item in enumerate(rows):
        a=_read(_safe_png(data["run"],item["reference_path"]))
        b=_read(_safe_png(data["run"],item["test_path"]))
        name=f"{index:03d}_"+sha256(item["source_path"].encode()).hexdigest()[:12]+"_"+item["lighting_mode"]+".png"
        _write_preview(
            previews/name,a,b,item["body_box_reference_xywh"],
            item["body_box_test_xywh"],item["lighting_mode"],verified=False
        )
        item["preview"]=("preview_proposals/"+name)
    manifest_sha=image_hash(manifest)
    result={
        "schema":SCHEMA,
        "source_manifest":str(manifest),
        "source_manifest_sha256":manifest_sha,
        "stage":"AWAITING_EXPLICIT_OPERATOR_REVIEW",
        "instructions":[
            "Abra cada preview e ajuste body_box_reference_xywh e body_box_test_xywh.",
            "Use coordenadas [x,y,w,h] na imagem extraída, não no PNG de preview.",
            "Marque corpo físico e terminais móveis; NÃO inscreva pads fixos da PCB.",
            "O contorno deve conter todo o componente, NÃO só a marcação 104 ou 0.",
            "Preencha review_notes e altere approved para true somente após revisar.",
            "As sugestões laranja NÃO são máscaras validadas, mesmo se parecerem corretas.",
            "Nenhuma CNN é treinada ou ativada por esta etapa."
        ],
        "total_images":len(rows),
        "events":len({row["event_id"] for row in rows}),
        "lighting_distribution":dict(Counter(r["lighting_mode"] for r in rows)),
        "rows":rows,
        "training_performed":False,
        "production_modified":False,
    }
    file=out/"body_masks_review.json"
    file.write_text(json.dumps(result,indent=2,ensure_ascii=False),
                    encoding="utf-8")
    summary=out/"summary.txt"
    summary.write_text(
        "DESLOCADO — CORPO COMPLETO, REVISÃO HUMANA V3\n"
        f"{len(rows)} imagens em {result['events']} eventos.\n"
        f"Luzes: {result['lighting_distribution']}\n"
        "TODAS as caixas propostas são rascunhos (approved=false).\n"
        "REVISAO OBRIGATORIA de cada referência e teste.\n"
        "Não houve treino ou mudança de motor.\n",
        encoding="utf-8",
    )
    return result,out


def validate_body_masks(review_file: Path) -> tuple[dict,Path]:
    path=Path(review_file).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("JSON de revisão não encontrado")
    review=json.loads(path.read_text(encoding="utf-8"))
    if review.get("schema")!=SCHEMA:
        raise ValueError("Schema de revisão incompatível")
    manifest=Path(review["source_manifest"]).resolve()
    if image_hash(manifest)!=review["source_manifest_sha256"]:
        raise ValueError("Manifesto foi modificado desde a proposta")
    existing,data=_rows(manifest)
    indexed={r["source_path"]:r for r in existing}
    submitted=review.get("rows")
    if not isinstance(submitted,list) or len(submitted)!=len(existing):
        raise ValueError("Revisão não contém todos os arquivos")
    actual={r.get("source_path") for r in submitted if isinstance(r,dict)}
    if len(actual)!=len(submitted) or actual!=set(indexed):
        raise ValueError("Duplicações ou arquivos omitidos na revisão")
    validated=[]
    for row in submitted:
        original=indexed[row["source_path"]]
        immutable=(
            "event_id","lighting_mode","reference_path","test_path",
            "reference_sha256","test_sha256","reference_size_wh","test_size_wh",
        )
        if any(row.get(k)!=original[k] for k in immutable):
            raise ValueError("Metadados de fonte alterados: "+row["source_path"])
        if row.get("approved") is not True:
            raise ValueError(
                "Falta revisão humana do corpo inteiro: "+row["source_path"]
            )
        if not isinstance(row.get("review_notes"),str) or not row["review_notes"].strip():
            raise ValueError("Inclua observação de revisão de: "+row["source_path"])
        ref_path=_safe_png(data["run"],original["reference_path"])
        test_path=_safe_png(data["run"],original["test_path"])
        if image_hash(ref_path)!=original["reference_sha256"] or (
            image_hash(test_path)!=original["test_sha256"]
        ):
            raise ValueError("Par gabarito/teste alterado: "+row["source_path"])
        reference,test=_read(ref_path),_read(test_path)
        boxes=[]
        for key,frame in (
            ("body_box_reference_xywh",reference),
            ("body_box_test_xywh",test),
        ):
            boxes.append(check_box(row.get(key),frame))
        # Corpo físico do componente deve ocupar proporções semelhantes
        # entre gabarito e teste. Não provar alinhamento, só evitar enganos.
        sizes=[(b[2]/f.shape[1],b[3]/f.shape[0])
               for b,f in zip(boxes,(reference,test))]
        if any(abs(sizes[0][k]-sizes[1][k])>.28 for k in (0,1)):
            raise ValueError(
                "Dimensões relativas muito diferentes entre gabarito/teste: "+
                row["source_path"]
            )
        validated.append({
            **{k:original[k] for k in immutable},
            "source_path":row["source_path"],
            "body_box_reference_xywh":list(boxes[0]),
            "body_box_test_xywh":list(boxes[1]),
            "approved_by_operator":True,
            "review_notes":row["review_notes"].strip(),
        })
    out=_new_folder(data["root"],"validated")
    preview_dir=out/"preview_validated"
    preview_dir.mkdir()
    for ix,row in enumerate(validated):
        a=_read(_safe_png(data["run"],row["reference_path"]))
        b=_read(_safe_png(data["run"],row["test_path"]))
        name=f"{ix:03d}_"+sha256(row["source_path"].encode()).hexdigest()[:12]+"_"+row["lighting_mode"]+".png"
        _write_preview(
            preview_dir/name,a,b,row["body_box_reference_xywh"],
            row["body_box_test_xywh"],row["lighting_mode"],verified=True,
        )
        row["preview"]="preview_validated/"+name
    result={
        "schema":RESULT_SCHEMA,
        "source_manifest":str(manifest),
        "source_manifest_sha256":image_hash(manifest),
        "source_review_sha256":image_hash(path),
        "total_approved":len(validated),
        "per_lighting":dict(Counter(r["lighting_mode"] for r in validated)),
        "rows":validated,
        "operator_reviewed_geometry":True,
        "mask_type":"RECTANGULAR_FULL_BODY_WITH_MOVING_TERMINALS",
        "note":"Revisão manual declarada; não é evidência de NG real",
        "training_performed":False,
        "production_approved":False,
        "production_modified":False,
    }
    (out/"validated_body_masks.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8"
    )
    (out/"summary.txt").write_text(
        f"DESLOCADO V3 — ROIs verificadas: {len(validated)}.\n"
        f"Por luz: {result['per_lighting']}\n"
        "Treino: NÃO. Motor operacional: INALTERADO.\n",
        encoding="utf-8",
    )
    return result,out


__all__=[
    "SCHEMA","RESULT_SCHEMA","propose_box","check_box",
    "prepare_body_masks","validate_body_masks",
]
