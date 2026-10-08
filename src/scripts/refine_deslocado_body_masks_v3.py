"""Editor visual de máscaras binárias DESLOCADO v3 (somente offline).

Exemplos:
  python -m src.scripts.refine_deslocado_body_masks_v3 --from-validated "...\validated_body_masks.json"
  python -m src.scripts.refine_deslocado_body_masks_v3 --review "...\pixel_masks_review.json"
  python -m src.scripts.refine_deslocado_body_masks_v3 --validate "...\pixel_masks_review.json"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from src.services.deslocado_pixel_masks_v3 import (
    REVIEW_SCHEMA, _load_mask, _mask_file, _overlay, _save_mask,
    _source_data, check_pixel_mask, image_hash,
    prepare_pixel_review, validate_pixel_review,
)
from src.scripts.train_deslocado_cnn import _read, _safe_png


def _atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.stem + "." + uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _paint(frame: np.ndarray, base_mask: np.ndarray, title: str):
    """Mouse esquerda preenche pixels; direita apaga; Enter aplica; ESC cancela."""
    window = "ODIN V3 - MASCARA PIXELS - " + title
    mask = base_mask.copy()
    state = {"radius": 8, "edited": False}

    def mouse(event, x, y, flags, userdata):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            left = event == cv2.EVENT_LBUTTONDOWN or (
                event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
            )
            right = event == cv2.EVENT_RBUTTONDOWN or (
                event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON
            )
            image_y = y - 35  # _overlay adiciona cabeçalho fora do crop original.
            if (left or right) and 0 <= x < mask.shape[1] and 0 <= image_y < mask.shape[0]:
                cv2.circle(mask, (x, image_y), state["radius"], 255 if left else 0, -1)
                state["edited"] = True

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            rendered = _overlay(frame, mask, "Esquerda=INCLUIR Direita=APAGAR")
            cv2.putText(
                rendered,
                f"Raio={state['radius']} [/] | ENTER=salvar ESC=cancelar C=limpar",
                (5, 13), cv2.FONT_HERSHEY_SIMPLEX, .37, (255, 255, 255), 1,
            )
            cv2.imshow(window, rendered)
            key = cv2.waitKey(35) & 0xFF
            if key in (13, 10):
                return mask, bool(state["edited"])
            if key == 27:
                return None, False
            if key == ord("c"):
                mask[:] = 0
                state["edited"] = True
            if key in (ord("]"), ord("+"), ord("=")):
                state["radius"] = min(65, state["radius"] + 2)
            if key in (ord("["), ord("-")):
                state["radius"] = max(1, state["radius"] - 2)
    finally:
        cv2.destroyWindow(window)


def _pair_preview(reference, test, ref_mask, test_mask, mode):
    a = _overlay(reference, ref_mask, mode + " GABARITO")
    b = _overlay(test, test_mask, mode + " TESTE")
    h = max(a.shape[0], b.shape[0])
    a = cv2.copyMakeBorder(a, 0, h-a.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    b = cv2.copyMakeBorder(b, 0, h-b.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    return cv2.hconcat([a, b])


def annotate(review_file: Path) -> dict:
    review_file = Path(review_file).expanduser().resolve()
    if not review_file.is_file() or review_file.is_symlink():
        raise ValueError("JSON de revisão de pixels não encontrado")
    data = json.loads(review_file.read_text(encoding="utf-8"))
    if data.get("schema") != REVIEW_SCHEMA:
        raise ValueError("Use o arquivo pixel_masks_review.json")
    legacy_path = Path(data["source_legacy_catalog"]).resolve()
    if image_hash(legacy_path) != data["source_legacy_catalog_sha256"]:
        raise ValueError("Catálogo de origem alterado")
    originals, source_data = _source_data(
        json.loads(legacy_path.read_text(encoding="utf-8"))
    )
    rows = data["rows"]
    if (len(rows) != len(originals)
        or {r.get("source_path") for r in rows} != set(originals)):
        raise ValueError("Revisão incompleta ou duplicada")
    window = "ODIN DESLOCADO V3 - CONFERIR MASCARA BINARIA"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    except cv2.error as error:
        raise RuntimeError("OpenCV sem HighGUI. Instale opencv-python (não headless)") from error
    skipped = 0
    try:
        for index, row in enumerate(rows):
            if row.get("approved") is True or row.get("excluded") is True:
                continue
            expected = originals[row["source_path"]]
            if any(row.get(field) != expected[field] for field in (
                "reference_sha256", "test_sha256", "reference_path", "test_path",
                "lighting_mode", "event_id",
            )):
                raise ValueError("Dados da fonte alterados: " + row["source_path"])
            frames = {
                part: _read(_safe_png(source_data["run"], expected[part + "_path"]))
                for part in ("reference", "test")
            }
            while True:
                masks = {}
                for part in ("reference", "test"):
                    file = _mask_file(review_file.parent, row["mask_" + part + "_path"])
                    if image_hash(file) != row["mask_" + part + "_sha256"]:
                        raise ValueError("Máscara modificada fora do editor: " + row["source_path"])
                    masks[part] = _load_mask(file, frames[part])
                cv2.imshow(window, _pair_preview(
                    frames["reference"], frames["test"],
                    masks["reference"], masks["test"], row["lighting_mode"]
                ))
                print(
                    f"\n{index+1}/{len(rows)} {row['source_path']}\n"
                    "e=EDITAR pixels dos dois | a=APROVAR | x=EXCLUIR | "
                    "s=PULAR | q=SALVAR E SAIR",
                    flush=True,
                )
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("q"), 27):
                    return {"remaining": sum(
                        not (r.get("approved") is True or r.get("excluded") is True)
                        for r in rows
                    )}
                if key == ord("s"):
                    skipped += 1
                    break
                if key == ord("e"):
                    edits = {}
                    for part in ("reference", "test"):
                        edited, changed = _paint(
                            frames[part], masks[part],
                            "GABARITO" if part == "reference" else "TESTE",
                        )
                        if edited is None:
                            edits = {}
                            break
                        edits[part] = (edited, changed)
                    if len(edits) != 2:
                        print("Edição cancelada. Não alterei as máscaras.", flush=True)
                        continue
                    for part, (edited, changed) in edits.items():
                        if changed:
                            file = _mask_file(review_file.parent,
                                              row["mask_" + part + "_path"])
                            _save_mask(file, edited)
                            row["mask_" + part + "_sha256"] = image_hash(file)
                            row["edited_" + part] = True
                    row["approved"] = False
                    row["review_notes"] = ""
                    _atomic_json(review_file, data)
                    continue
                if key == ord("x"):
                    print("Excluir caso: [c] componente cortado ou [u] segmentação incerta?", flush=True)
                    reason_key = cv2.waitKey(0) & 0xFF
                    reason = {ord("c"): "CROPPED_COMPONENT",
                              ord("u"): "UNRELIABLE_SEGMENTATION"}.get(reason_key)
                    if not reason:
                        print("Exclusão cancelada.", flush=True)
                        continue
                    print("Confirmar EXCLUSÃO explícita? [s=SIM]", flush=True)
                    if cv2.waitKey(0) & 0xFF != ord("s"):
                        continue
                    row.update(
                        excluded=True, approved=False, exclusion_reason=reason,
                        status="EXCLUDED", review_notes="Revisado e excluído: " + reason,
                    )
                    _atomic_json(review_file, data)
                    break
                if key == ord("a"):
                    try:
                        boxes = [
                            check_pixel_mask(masks[part], frames[part])
                            for part in ("reference", "test")
                        ]
                        for part in ("reference", "test"):
                            if (row.get("proposal_" + part + "_type")
                                == "RECTANGLE_PLACEHOLDER_EDIT_REQUIRED"
                                and row.get("edited_" + part) is not True):
                                raise ValueError("Proposta retangular requer edição: " + part)
                        norm = [
                            (b[2] / frames[p].shape[1], b[3] / frames[p].shape[0])
                            for b, p in zip(boxes, ("reference", "test"))
                        ]
                        if any(abs(norm[0][k] - norm[1][k]) > .28 for k in (0, 1)):
                            raise ValueError("Dimensões incompatíveis entre gabarito/teste")
                    except ValueError as error:
                        print("NÃO APROVADO:", error, flush=True)
                        continue
                    print(
                        "CONFIRME: máscaras incluem corpo e terminais móveis completos, "
                        "sem inscrição isolada, pads fixos ou parte cortada? [s=SIM / n=NÃO]",
                        flush=True,
                    )
                    if cv2.waitKey(0) & 0xFF != ord("s"):
                        continue
                    row.update(
                        approved=True, excluded=False, exclusion_reason="",
                        status="PIXEL_MASK_APPROVED",
                        review_notes="Operador conferiu pixels: corpo e terminais móveis; "
                                     "pads fixos excluídos; crop completo.",
                    )
                    _atomic_json(review_file, data)
                    break
    finally:
        cv2.destroyAllWindows()
    return {"remaining": sum(
        not (r.get("approved") is True or r.get("excluded") is True) for r in rows
    ), "skipped": skipped}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Refinar máscaras de pixels DESLOCADO v3")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--from-validated", type=Path,
                        help="Catálogo retangular validated_body_masks.json (somente fonte)")
    choice.add_argument("--review", type=Path, help="Abrir editor de pixel_masks_review.json")
    choice.add_argument("--validate", type=Path, help="Validar e exportar máscaras binárias")
    args = parser.parse_args(argv)
    if args.from_validated:
        result, folder = prepare_pixel_review(args.from_validated)
        print("PARES PARA REVISÃO:", result["total_images"])
        print("ARQUIVO:", folder / "pixel_masks_review.json")
        print("PREVIEWS:", folder / "preview_proposals")
    elif args.review:
        print("RESULTADO:", annotate(args.review))
    else:
        result, folder = validate_pixel_review(args.validate)
        print(f"MÁSCARAS REVISADAS: {result['total_reviewed']} | "
              f"APROVADAS: {result['total_approved']} | "
              f"EXCLUÍDAS: {result['total_excluded']}")
        print("CATÁLOGO:", folder / "validated_component_masks.json")
        print("PREVIEWS:", folder / "preview_validated")
    print("Treinamento e motores operacionais inalterados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
