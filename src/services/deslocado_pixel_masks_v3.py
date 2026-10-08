"""DESLOCADO v3 — máscaras binárias reais, isoladas da operação.

Refina o catálogo retangular v1 sem convertê-lo em aprovação de pixels.
As propostas GrabCut são rascunhos; revisão humana de CADA par é obrigatória.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from src.services.deslocado_body_masks_v3 import (
    RESULT_SCHEMA, _new_folder, _rows, check_box, image_hash,
)
from src.scripts.train_deslocado_cnn import _read, _safe_png

REVIEW_SCHEMA = "visionx.deslocado_pixel_mask_review.v1"
VALIDATED_SCHEMA = "visionx.deslocado_pixel_mask_validation.v1"
EXCLUSION_REASONS = {"CROPPED_COMPONENT", "UNRELIABLE_SEGMENTATION"}
IMMUTABLE = (
    "source_path", "event_id", "lighting_mode", "reference_path", "test_path",
    "reference_sha256", "test_sha256", "reference_size_wh", "test_size_wh",
)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise OSError("Não foi possível codificar a máscara")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(encoded.tobytes())
    tmp.replace(path)


def _mask_file(folder: Path, relative: str) -> Path:
    """Aceita somente PNG regular dentro de masks/, sem traversal/symlinks."""
    if not isinstance(relative, str):
        raise ValueError("Caminho da máscara inválido")
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or relative_path.parts[:1] != ("masks",)
        or len(relative_path.parts) != 2
        or relative_path.suffix.lower() != ".png"
        or ".." in relative_path.parts
    ):
        raise ValueError("Caminho externo à revisão: " + relative)
    path = folder / relative_path
    if (not path.is_file() or path.is_symlink() or (folder / "masks").is_symlink()):
        raise ValueError("Máscara ausente ou link simbólico: " + relative)
    return path


def _load_mask(path: Path, frame: np.ndarray) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        mask is None or mask.dtype != np.uint8
        or mask.ndim != 2 or mask.shape != frame.shape[:2]
        or not np.all((mask == 0) | (mask == 255))
    ):
        raise ValueError("Máscara deve ser PNG grayscale binário, com dimensões do crop")
    return mask


def _candidate(frame: np.ndarray, box: list[int]) -> tuple[np.ndarray, str]:
    """GrabCut é hipótese local. Jamais aprova, nem inclui recorte fora do ROI."""
    x, y, w, h = check_box(box, frame)
    H, W = frame.shape[:2]
    # GrabCut precisa de fundo em todas as bordas da imagem.
    rect = (max(1, x), max(1, y),
            min(w, W - max(1, x) - 2), min(h, H - max(1, y) - 2))
    if rect[2] > 4 and rect[3] > 4:
        try:
            labels = np.zeros((H, W), np.uint8)
            cv2.grabCut(frame, labels, rect, None, None, 4, cv2.GC_INIT_WITH_RECT)
            proposal = np.where(
                (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD),
                255, 0,
            ).astype(np.uint8)
            if np.count_nonzero(proposal) >= H * W * .08:
                return proposal, "GRABCUT_UNVERIFIED"
        except cv2.error:
            pass
    # Placeholder visível para correção manual; jamais aprovar sem edição.
    proposal = np.zeros((H, W), np.uint8)
    inset_x, inset_y = max(2, w // 18), max(2, h // 18)
    cv2.rectangle(
        proposal, (x + inset_x, y + inset_y),
        (min(W - 2, x + w - inset_x), min(H - 2, y + h - inset_y)),
        255, -1,
    )
    return proposal, "RECTANGLE_PLACEHOLDER_EDIT_REQUIRED"


def _overlay(frame: np.ndarray, mask: np.ndarray, caption: str,
             *, verified: bool = False) -> np.ndarray:
    output = frame.copy()
    color = np.zeros_like(output)
    if verified:
        color[:, :, 1] = 230  # Verde: somente revisão aprovada.
    else:
        color[:, :, 1] = 165  # Laranja: proposta não validada.
        color[:, :, 2] = 255
    blended = cv2.addWeighted(frame, .68, color, .32, 0)
    output[mask > 0] = blended[mask > 0]
    outline, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, outline, -1, (0, 220, 0) if verified else (0, 165, 255), 2)
    return _caption(output, caption)


def _caption(frame: np.ndarray, label: str) -> np.ndarray:
    output = cv2.copyMakeBorder(frame, 35, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(output, label[:58], (5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _preview(output: Path, reference: np.ndarray, test: np.ndarray,
             ref_mask: np.ndarray, test_mask: np.ndarray, title: str,
             *, verified: bool = False) -> None:
    left = _overlay(reference, ref_mask, title + " GABARITO", verified=verified)
    right = _overlay(test, test_mask, title + " TESTE", verified=verified)
    h = max(left.shape[0], right.shape[0])
    left = cv2.copyMakeBorder(left, 0, h-left.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    right = cv2.copyMakeBorder(right, 0, h-right.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    if not cv2.imwrite(str(output), cv2.hconcat([left, right])):
        raise OSError("Falha ao gravar preview")


def _source_data(catalog: dict) -> tuple[dict, dict]:
    manifest = Path(catalog["source_manifest"]).expanduser().resolve()
    if image_hash(manifest) != catalog["source_manifest_sha256"]:
        raise ValueError("Manifesto de origem alterado")
    rows, data = _rows(manifest)  # Também verifica os PNGs originais no acervo.
    original = {r["source_path"]: r for r in rows}
    submitted = catalog.get("rows")
    if not isinstance(submitted, list) or len(submitted) != len(rows):
        raise ValueError("Catálogo não cobre todos os pares")
    if {r.get("source_path") for r in submitted} != set(original):
        raise ValueError("Fontes ausentes, duplicadas ou inesperadas")
    for row in submitted:
        expected = original[row["source_path"]]
        if any(row.get(k) != expected[k] for k in IMMUTABLE):
            raise ValueError("Metadados ou SHA divergentes: " + row["source_path"])
    return original, data


def prepare_pixel_review(catalog_file: Path) -> tuple[dict, Path]:
    catalog_file = Path(catalog_file).expanduser().resolve()
    if not catalog_file.is_file() or catalog_file.is_symlink():
        raise ValueError("Catálogo retangular inexistente")
    catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
    if catalog.get("schema") != RESULT_SCHEMA:
        raise ValueError("Exigido validated_body_masks.json v1")
    originals, data = _source_data(catalog)
    folder = _new_folder(data["root"], "pixel_review")
    (folder / "masks").mkdir()
    (folder / "preview_proposals").mkdir()
    output_rows = []
    for index, row in enumerate(catalog["rows"]):
        if row.get("approved_by_operator") is not True:
            raise ValueError("Catálogo retangular não foi aprovado")
        expected = originals[row["source_path"]]
        frames = [
            _read(_safe_png(data["run"], expected["reference_path"])),
            _read(_safe_png(data["run"], expected["test_path"])),
        ]
        result = dict(row)
        for suffix, frame, field in (
            ("reference", frames[0], "body_box_reference_xywh"),
            ("test", frames[1], "body_box_test_xywh"),
        ):
            mask, kind = _candidate(frame, row[field])
            name = f"{index:03d}_{suffix}.png"
            target = folder / "masks" / name
            _save_mask(target, mask)
            result[f"mask_{suffix}_path"] = "masks/" + name
            result[f"mask_{suffix}_sha256"] = image_hash(target)
            result[f"proposal_{suffix}_type"] = kind
        result.update(
            approved=False, excluded=False, exclusion_reason="",
            review_notes="", manually_edited=False,
            status="PIXEL_REVIEW_REQUIRED",
        )
        _preview(
            folder / "preview_proposals" / f"{index:03d}.png",
            *frames,
            _load_mask(_mask_file(folder, result["mask_reference_path"]), frames[0]),
            _load_mask(_mask_file(folder, result["mask_test_path"]), frames[1]),
            row["lighting_mode"],
        )
        output_rows.append(result)
    review = {
        "schema": REVIEW_SCHEMA,
        "source_manifest": catalog["source_manifest"],
        "source_manifest_sha256": catalog["source_manifest_sha256"],
        "source_legacy_catalog_sha256": image_hash(catalog_file),
        "source_legacy_catalog": str(catalog_file),
        "total_images": len(output_rows),
        "rows": output_rows,
        "note": "GrabCut é proposta não confiável: 0 aprovações automáticas.",
        "training_performed": False, "production_modified": False,
    }
    (folder / "pixel_masks_review.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (folder / "summary.txt").write_text(
        f"DESLOCADO pixel masks: {len(output_rows)} pares, todos pendentes.\n"
        "Revise cada máscara, não apenas a caixa. Recortes incompletos devem ser excluídos.\n"
        "Sem treinamento nem mudança de produção.\n", encoding="utf-8"
    )
    return review, folder


def check_pixel_mask(mask: np.ndarray, frame: np.ndarray) -> tuple[int, int, int, int]:
    H, W = frame.shape[:2]
    if mask.shape != (H, W) or mask.dtype != np.uint8:
        raise ValueError("Formato ou dimensões da máscara inválidas")
    fg = mask == 255
    area = int(np.count_nonzero(fg))
    if area < H * W * .08 or area > H * W * .78:
        raise ValueError("Área da máscara incompatível com corpo físico (inscrição/fundo)")
    if np.any(fg[:3, :]) or np.any(fg[-3:, :]) or np.any(fg[:, :3]) or np.any(fg[:, -3:]):
        raise ValueError("Máscara toca a borda do recorte: componente possivelmente cortado")
    points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(points)
    if w < max(12, round(W * .32)) or h < max(12, round(H * .32)):
        raise ValueError("Máscara pequena: possível inscrição em vez de componente")
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest = max((int(row[cv2.CC_STAT_AREA]) for row in stats[1:]), default=0)
    if components <= 1 or largest < area * .65:
        raise ValueError("Máscara muito fragmentada: possível fundo/pads independentes")
    return x, y, w, h


def validate_pixel_review(review_file: Path) -> tuple[dict, Path]:
    review_file = Path(review_file).expanduser().resolve()
    if not review_file.is_file() or review_file.is_symlink():
        raise ValueError("Revisão não encontrada")
    review = json.loads(review_file.read_text(encoding="utf-8"))
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError("Schema de revisão de pixels incompatível")
    legacy_file = Path(review["source_legacy_catalog"]).resolve()
    if image_hash(legacy_file) != review["source_legacy_catalog_sha256"]:
        raise ValueError("Catálogo retangular alterado")
    legacy = json.loads(legacy_file.read_text(encoding="utf-8"))
    originals, data = _source_data(legacy)
    rows = review.get("rows")
    if (
        not isinstance(rows, list) or len(rows) != len(originals)
        or len({r.get("source_path") for r in rows}) != len(rows)
        or {r.get("source_path") for r in rows} != set(originals)
    ):
        raise ValueError("Cobertura da revisão incompleta ou duplicada")
    if review["source_manifest_sha256"] != legacy["source_manifest_sha256"]:
        raise ValueError("SHA do manifesto incompatível")
    records = []
    for row in rows:
        expected = originals[row["source_path"]]
        if any(row.get(k) != expected[k] for k in IMMUTABLE):
            raise ValueError("Fonte ou hashes adulterados: " + row["source_path"])
        if row.get("excluded") is True and row.get("approved") is False:
            if row.get("exclusion_reason") not in EXCLUSION_REASONS:
                raise ValueError("Exclusão sem justificativa válida: " + row["source_path"])
            records.append({"source_path": row["source_path"],
                            "lighting_mode": row["lighting_mode"],
                            "status": "EXCLUDED",
                            "reason": row["exclusion_reason"]})
            continue
        if row.get("approved") is not True or row.get("excluded") is not False:
            raise ValueError("Máscara sem revisão: " + row["source_path"])
        if not str(row.get("review_notes", "")).strip():
            raise ValueError("Confirmação humana ausente: " + row["source_path"])
        if any(
            row.get("proposal_" + item + "_type") == "RECTANGLE_PLACEHOLDER_EDIT_REQUIRED"
            and row.get("edited_" + item) is not True
            for item in ("reference", "test")
        ):
            raise ValueError("Placeholder retangular exige edição real: " + row["source_path"])
        frames, masks, paths, boxes = [], [], [], []
        for part in ("reference", "test"):
            frame = _read(_safe_png(data["run"], expected[part + "_path"]))
            filename = _mask_file(review_file.parent, row.get("mask_" + part + "_path"))
            if image_hash(filename) != row.get("mask_" + part + "_sha256"):
                raise ValueError("Hash da máscara alterado: " + row["source_path"])
            mask = _load_mask(filename, frame)
            box = check_pixel_mask(mask, frame)
            frames.append(frame)
            masks.append(mask)
            paths.append(filename)
            boxes.append(box)
        sizes = [(b[2] / f.shape[1], b[3] / f.shape[0])
                 for b, f in zip(boxes, frames)]
        if any(abs(sizes[0][k] - sizes[1][k]) > .28 for k in (0, 1)):
            raise ValueError("Geometria desigual entre gabarito/teste: " + row["source_path"])
        records.append((row, frames, masks, paths, boxes))
    approved = [r for r in records if isinstance(r, tuple)]
    if not approved:
        raise ValueError("Todos excluídos: não há máscaras físicas validadas")
    out = _new_folder(data["root"], "pixel_validated")
    (out / "masks_validated").mkdir()
    (out / "preview_validated").mkdir()
    validated, excluded = [], []
    for index, record in enumerate(records):
        if isinstance(record, dict):
            excluded.append(record)
            continue
        row, frames, masks, paths, boxes = record
        item = {k: row[k] for k in IMMUTABLE}
        item.update(
            status="APPROVED_PIXEL_MASK", review_notes=row["review_notes"],
            component_boxes_xywh=[list(b) for b in boxes]
        )
        for part, source in zip(("reference", "test"), paths):
            relative = f"masks_validated/{index:03d}_{part}.png"
            destination = out / relative
            shutil.copyfile(source, destination)
            item["mask_" + part + "_path"] = relative
            item["mask_" + part + "_sha256"] = image_hash(destination)
        preview = f"preview_validated/{index:03d}.png"
        _preview(out / preview, *frames, *masks, row["lighting_mode"], verified=True)
        item["preview"] = preview
        validated.append(item)
    result = {
        "schema": VALIDATED_SCHEMA,
        "source_manifest": review["source_manifest"],
        "source_manifest_sha256": review["source_manifest_sha256"],
        "source_review_sha256": image_hash(review_file),
        "total_reviewed": len(rows), "total_approved": len(validated),
        "total_excluded": len(excluded), "excluded": excluded, "rows": validated,
        "mask_type": "BINARY_PIXEL_MASK_0_255",
        "training_performed": False, "production_approved": False,
        "production_modified": False,
    }
    (out / "validated_component_masks.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "summary.txt").write_text(
        f"DESLOCADO máscaras de pixels: {len(validated)} aprovadas, "
        f"{len(excluded)} excluídas, {len(rows)} revisadas.\n"
        "Nenhum treino e nenhum motor alterado.\n", encoding="utf-8"
    )
    return result, out
