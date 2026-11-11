"""Contrato único para identidade de frame AOI já presente no acervo.

Comparação pixel-idêntica, NÃO similaridade KNN nem comprovação de rótulo
humano. O índice de origem é o inventário SHA validado no warmup da KNN C.
A verificação de arquivo é repetida no gate de Produção antes da tecla XP.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path, PurePosixPath

SCHEMA = "visionx.archive_exact_side_evidence.v1"


def _digest(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def verified_archive_exact_frame(proof: dict | None) -> bool:
    """Revalida fonte PNG sem confiar em caminho, label ou cache obsoletos."""
    if not isinstance(proof, dict) or not (
        proof.get("schema") == SCHEMA
        and proof.get("status") == "EXACT_FRAME"
        and proof.get("lighting_mode") == "SIDE"
        and proof.get("label") in {"OK", "NG"}
        and proof.get("index_status") == "READY"
        and proof.get("diagnostic_only") is True
        and proof.get("decision_authorized") is False
        and proof.get("frame_sha256") == proof.get("source_pixel_sha256")
        and _digest(proof.get("frame_sha256"))
        and _digest(proof.get("source_file_sha256"))
    ):
        return False
    root_name, source_name = proof.get("root"), proof.get("source_path")
    if not (isinstance(root_name, str) and root_name
            and isinstance(source_name, str) and source_name):
        return False
    relative = PurePosixPath(source_name)
    archive_dir = "ok_archive" if proof["label"] == "OK" else "ng_archive"
    if (relative.is_absolute() or len(relative.parts) != 3
            or relative.parts[:2] != ("public", archive_dir)
            or relative.suffix.lower() != ".png"):
        return False
    try:
        root = Path(root_name).resolve(strict=True)
        source = root.joinpath(*relative.parts)
        # Não seguir link simbólico (nem o diretório) para escapar do acervo.
        if (source.is_symlink() or not source.is_file()
                or source.resolve(strict=True).parent
                != (root / "public" / archive_dir).resolve(strict=True)):
            return False
        return sha256(source.read_bytes()).hexdigest() == proof["source_file_sha256"]
    except (OSError, ValueError):
        return False


def corroborated_archive_exact_side(analysis: dict | None) -> bool:
    """Identidade de frame + mesma direção V3/KNN, sem limiar KNN oportunista.

    Exige a mesma referência pixel-idêntica como vizinho 1 de A e B.
    Se o archive é provisório ou há contradição real, não transforma a
    fotografia em julgamento humano; o fluxo normal multilight prossegue.
    """
    if not isinstance(analysis, dict):
        return False
    detail = analysis.get("detail")
    if not isinstance(detail, dict):
        return False
    proof = detail.get("archive_exact_frame")
    if not verified_archive_exact_frame(proof):
        return False
    label = "NG" if analysis.get("is_defect") is True else "OK"
    if (proof["label"] != label
            or analysis.get("verdict") != (
                "DEFEITO REAL" if label == "NG" else "FALHA FALSA")):
        return False
    if label == "OK" and proof.get("physical_probe_status") != "NO_STRONG_EVIDENCE":
        return False
    human = detail.get("confirmed_visual_recall")
    if isinstance(human, dict) and human.get("status") == (
        "HUMAN_NG_VISUAL_CANDIDATE" if label == "OK"
        else "HUMAN_OK_VISUAL_CANDIDATE"
    ):
        return False
    knn = detail.get("knn_c")
    if not isinstance(knn, dict) or not (
        knn.get("schema") == "visionx.knn_c_live_review_gated.v1"
        and knn.get("bank_status") == "READY"
        and knn.get("diagnostic_only") is True
        and knn.get("auto_decision_authorized") is False
        and knn.get("production_approved") is False
        and knn.get("conflicts_with_existing_verdict") is not True
    ):
        return False
    return all(
        isinstance(knn.get(key), list) and bool(knn[key])
        and isinstance(knn[key][0], dict)
        and knn[key][0].get("label") == label
        and knn[key][0].get("source_pixel_sha256") == proof["frame_sha256"]
        and knn[key][0].get("path") == proof["source_path"]
        for key in ("neighbors_A", "neighbors_B")
    )
