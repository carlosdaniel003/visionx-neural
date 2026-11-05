"""Reconhecimento da mesma tela arquivada quando MSS a exibe redimensionada.

Identidade dos bytes do frame (SHA) e equivalência geométrica de exibição
são contratos distintos. O algoritmo nunca retorna par AOI/KNN exato nem
altera classificação, checkpoint, memória humana ou comando XP.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.config.settings import BASE_DIR
from src.services.image_archive_dedup import (
    archive_category_from_filename, archive_visual_scope, image_fingerprint,
)

SCHEMA = "visionx.archive_rendered_source_match.v1"
_LONG_EDGE = 900


def _thumbnail(frame):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[-1] != 3
            or min(frame.shape[:2]) < 240):
        return None
    factor = min(1.0, _LONG_EDGE / max(frame.shape[:2]))
    if factor == 1.0:
        return frame.copy()
    return cv2.resize(
        frame, (round(frame.shape[1] * factor), round(frame.shape[0] * factor)),
        interpolation=cv2.INTER_AREA,
    )


def compare_rendered_source(source: np.ndarray, screen: np.ndarray) -> dict | None:
    """Verifica registro de escala/translação + fidelidade local dos pixels.

    O teste por tiles rejeita alterações focais mesmo quando o fundo AOI
    inteiro e os textos permanecem parecidos. Nenhuma nota é interpretada
    como uma probabilidade de peça OK/NG.
    """
    source, screen = _thumbnail(source), _thumbnail(screen)
    if source is None or screen is None:
        return None
    detector = cv2.ORB_create(
        nfeatures=1300, scaleFactor=1.2, edgeThreshold=19, fastThreshold=12,
    )
    gray_src = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    kp_src, desc_src = detector.detectAndCompute(gray_src, None)
    kp_dst, desc_dst = detector.detectAndCompute(gray_dst, None)
    if desc_src is None or desc_dst is None or len(kp_src) < 25:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_src, desc_dst, k=2)
    good = [
        a for pair in pairs if len(pair) == 2
        for a, b in (pair,) if a.distance < .74 * b.distance
    ]
    if len(good) < 24:
        return None
    src = np.float32([kp_src[m.queryIdx].pt for m in good])
    dst = np.float32([kp_dst[m.trainIdx].pt for m in good])
    affine, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=2500, confidence=.995,
    )
    if affine is None or inliers is None:
        return None
    votes = int(inliers.sum())
    inlier_fraction = votes / len(good)
    scale = float(np.hypot(affine[0, 0], affine[1, 0]))
    if (votes < 18 or inlier_fraction < .50
            or not .20 <= scale <= 2.5
            or abs(float(np.arctan2(affine[1, 0], affine[0, 0]))) > .08):
        return None

    out_size = (screen.shape[1], screen.shape[0])
    warped = cv2.warpAffine(source, affine, out_size)
    mask = cv2.warpAffine(
        np.full(source.shape[:2], 255, dtype=np.uint8),
        affine, out_size,
    ) == 255
    if int(mask.sum()) < 12000:
        return None
    # Reamostragem de tela causa diferenças de borda; comparar em tons
    # de cinza e por pequenas regiões espacialmente localizadas.
    source_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    screen_gray = gray_dst.astype(np.float32)
    delta = np.abs(source_gray - screen_gray) / 255.0
    masked = delta[mask]
    mae = float(masked.mean())
    p95 = float(np.percentile(masked, 95))
    if mae > .015 or p95 > .08:
        return None
    tile_mae = []
    side = 30
    for y in range(0, screen.shape[0], side):
        for x in range(0, screen.shape[1], side):
            tile_mask = mask[y:y+side, x:x+side]
            if tile_mask.size and float(tile_mask.mean()) >= .70:
                tile_mae.append(float(
                    delta[y:y+side, x:x+side][tile_mask].mean()
                ))
    if len(tile_mae) < 12 or max(tile_mae) > .045:
        return None

    return {
        "registration": "AFFINE_SCALE_TRANSLATION",
        "inliers": votes,
        "inlier_fraction": round(inlier_fraction, 4),
        "photometric_mae": round(mae, 6),
        "max_local_mae": round(max(tile_mae), 6),
        "source_pixel_sha256": image_fingerprint(source),
        "screen_pixel_identical": False,
    }


def match_rendered_archive(
    screenshot: np.ndarray, category: str, candidate_paths: list[str],
    *, root: Path = BASE_DIR,
) -> dict:
    """Consulta fontes sugeridas pelo índice visual, relendo cada PNG.

    A validade da origem depende de raiz, pasta, luz e conteúdo
    atuais. Rótulos conflitantes nunca são reconciliados por score.
    """
    root = Path(root).resolve()
    folders = {
        "OK": (root / "public" / "ok_archive").resolve(),
        "NG": (root / "public" / "ng_archive").resolve(),
    }
    base = {
        "schema": SCHEMA, "status": "NO_MATCH", "label": "",
        "diagnostic_only": True, "decision_authorized": False,
        "exact_aoi_pair": False, "frame_pixel_identical": False,
        "matches": [],
    }
    if _thumbnail(screenshot) is None:
        return {**base, "status": "UNAVAILABLE"}
    requested_scopes = {
        archive_visual_scope(f"{category}_{light}")
        for light in ("SIDE", "TOP", "MID")
    }
    checked = set()
    matches = []
    for relative in candidate_paths[:24]:
        if not isinstance(relative, str):
            continue
        path = (root / relative).resolve()
        if path in checked or path.suffix.lower() != ".png":
            continue
        checked.add(path)
        label = next(
            (name for name, folder in folders.items()
             if path.parent == folder),
            "",
        )
        category_from_path = archive_category_from_filename(path)
        if not label or not category_from_path or archive_visual_scope(category_from_path) not in requested_scopes:
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            continue
        proof = compare_rendered_source(raw, screenshot)
        if proof is not None:
            matches.append({
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "label": label, **proof,
                "source_pixel_sha256": image_fingerprint(raw),
            })
    base["matches"] = matches
    labels = {m["label"] for m in matches}
    if len(labels) > 1:
        base["status"] = "CONFLICT"
    elif matches:
        best = min(matches, key=lambda m: (
            m["max_local_mae"], m["photometric_mae"],
        ))
        base.update(
            status="RENDERED_ARCHIVE_SOURCE",
            label=best["label"],
            path=best["path"],
            evidence=best,
        )
    return base


__all__ = ["compare_rendered_source", "match_rendered_archive", "SCHEMA"]
