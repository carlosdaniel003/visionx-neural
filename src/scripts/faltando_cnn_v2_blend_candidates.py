"""OFFLINE weight-space interpolation of two FALTANDO V2 experimental candidates.

No training or model routing/fusion. Only weights from the same champion,
same archive manifest and same immutable train/holdout split are permitted.
The grid is selected on historical + online + training evidence, NEVER on the
holdout. Outputs are always non-production candidates and diagnostics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2
from src.scripts.train_faltando_cnn_v2_online import _all_online, _assess, _base_model, _historical
from src.scripts.train_faltando_cnn_v2_verified_archive import (
    _operational_light_summary, _selection_score, candidate_acceptance,
    split_verified_events,
)
from src.services.faltando_neural_archive_experiment import (
    archive_labeled_experimental_events,
)


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Arquivo ausente ou symlink")
    return sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, given: Path, subdir: Path) -> Path:
    path = (given if given.is_absolute() else root / given)
    if path.is_symlink() or not path.resolve().is_relative_to(subdir.resolve()):
        raise ValueError("Caminho de experimento fora da pasta autorizada")
    if not path.is_file():
        raise ValueError("Arquivo de experimento não encontrado")
    return path.resolve()


def _verified_candidate(root: Path, report_path: Path) -> tuple[dict, dict]:
    folder = root / "reports/faltando_neural/archive_experiments"
    path = _inside(root, report_path, folder)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or (
        report.get("qualification_mode") != "ARCHIVE_FOLDER_BULK_EXPERIMENT"
        or report.get("candidate_only") is not True
        or report.get("promoted") is not False
        or report.get("production_approval_eligible") is not False
    ):
        raise ValueError("Relatório não é candidato experimental em lote")
    candidate = _inside(root, Path(report["candidate_relative_path"]), folder)
    if candidate.parent != path.parent:
        raise ValueError("Candidato e relatório não pertencem ao mesmo experimento")
    if _sha(candidate) != report.get("candidate_sha256"):
        raise ValueError("SHA-256 do candidato não corresponde ao relatório")
    ckpt = torch.load(candidate, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or (
        ckpt.get("schema") != MODEL_SCHEMA_V2
        or ckpt.get("archive_labels_experimental") is not True
        or ckpt.get("production_approved") is not False
        or ckpt.get("individual_pair_review_complete") is not False
        or ckpt.get("source_champion_sha256") != report.get("base_checkpoint_sha256")
        or ckpt.get("provisional_archive_manifest_sha256")
        != report["provenance"]["manifest_sha256"]
    ):
        raise ValueError("Metadados de checkpoint incompatíveis")
    state = ckpt.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint sem state_dict")
    return report, ckpt


def _mixed_weights(first: dict, second: dict, alpha: float) -> dict:
    """alpha=0: conservative, alpha=1: aggressive; only floating tensors."""
    a, b = first["state_dict"], second["state_dict"]
    if set(a) != set(b) or not 0. <= alpha <= 1.:
        raise ValueError("Estados de CNN incompatíveis")
    state = {}
    for key in a:
        left, right = a[key], b[key]
        if (not isinstance(left, torch.Tensor)
                or not isinstance(right, torch.Tensor)
                or left.shape != right.shape
                or left.dtype != right.dtype
                or not left.is_floating_point()
                or not bool(torch.isfinite(left).all())
                or not bool(torch.isfinite(right).all())):
            raise ValueError("Parâmetro neural incompatível/não finito: " + key)
        state[key] = left * (1. - alpha) + right * alpha
    return state


def blend_offline(
    root: Path, manifest: Path, conservative_report: Path,
    aggressive_report: Path, *, steps: int = 10,
) -> dict:
    root = Path(root).resolve()
    if not 2 <= steps <= 20 or type(steps) is not int:
        raise ValueError("Quantidade de passos deve estar entre 2 e 20")
    if (root / "reports/neural_online/live_active.json").exists():
        raise ValueError("Checkpoint online ativo: auditar antes de experimento")
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    left_report, left = _verified_candidate(root, Path(conservative_report))
    right_report, right = _verified_candidate(root, Path(aggressive_report))
    if left_report["candidate_sha256"] == right_report["candidate_sha256"]:
        raise ValueError("Os dois checkpoints são idênticos")
    for key in ("base_checkpoint_sha256", "train_event_ids", "holdout_event_ids"):
        if left_report.get(key) != right_report.get(key):
            raise ValueError("Candidatos têm diferentes origens/splits: " + key)
    if left_report["provenance"]["manifest_sha256"] != right_report["provenance"]["manifest_sha256"]:
        raise ValueError("Manifests diferentes")
    for key in ("image_size", "focus_fraction", "lights", "schema"):
        if left.get(key) != right.get(key):
            raise ValueError("Pré-processamento/arquitetura incompatíveis: " + key)
    source_path = Path(manifest)
    source_path = source_path if source_path.is_absolute() else root / source_path
    if _sha(source_path) != left_report["provenance"]["manifest_sha256"]:
        raise ValueError("Manifesto original foi alterado")
    base_path, base_checkpoint = _base_model(root)
    if _sha(base_path) != left_report["base_checkpoint_sha256"]:
        raise ValueError("Champion ativo não corresponde à origem dos candidatos")
    if base_checkpoint.get("schema") != MODEL_SCHEMA_V2:
        raise ValueError("Champion de arquitetura diferente")
    events, provenance = archive_labeled_experimental_events(source_path)
    train, hold, split = split_verified_events(
        events, seed=left_report["split"]["seed"],
        holdout=left_report["split"]["holdout_requested"],
    )
    if (
        [e["id"] for e in train] != left_report["train_event_ids"]
        or [e["id"] for e in hold] != left_report["holdout_event_ids"]
        or provenance["manifest_sha256"] != left_report["provenance"]["manifest_sha256"]
    ):
        raise ValueError("Split do acervo mudou; não misturar checkpoints")
    _mixed_weights(left, right, 0.)  # Validate full state before any inference.
    size, focus = int(left["image_size"]), float(left["focus_fraction"])
    historical, online = _historical(root, base_checkpoint), _all_online(root)
    champion = FaltandoCNNV2()
    champion.load_state_dict(base_checkpoint["state_dict"], strict=True)
    old_historical = _assess(champion, historical, size, focus)
    old_online = _assess(champion, online, size, focus)
    old_holdout = _assess(champion, hold, size, focus)
    if not old_historical["all_correct"]:
        raise ValueError("Champion não preserva o histórico; abortar")
    model = FaltandoCNNV2()
    grid = []
    best_key, best_alpha, best_state = None, None, None
    for index in range(steps + 1):
        alpha = index / steps
        state = _mixed_weights(left, right, alpha)
        model.load_state_dict(state, strict=True)
        hist = _assess(model, historical, size, focus)
        online_result = _assess(model, online, size, focus)
        train_result = _assess(model, train, size, focus)
        score = _selection_score(hist, online_result, train_result)
        grid.append({
            "aggressive_weight": round(alpha, 6),
            "historical_correct_images": hist["correct_images"],
            "historical_total_images": hist["images"],
            "historical_all_correct": hist["all_correct"],
            "online_correct_images": online_result["correct_images"],
            "online_ng_false_ok": online_result["ng_false_ok"],
            "train_correct_images": train_result["correct_images"],
            "train_total_images": train_result["images"],
        })
        if best_key is None or score < best_key:
            best_key, best_alpha, best_state = score, alpha, state
        print(
            f"BLEND {index+1}/{steps+1} a={alpha:.2f} "
            f"historical={hist['correct_images']}/{hist['images']} "
            f"online={online_result['correct_images']}/{online_result['images']}",
            flush=True,
        )
    if best_state is None:
        raise ValueError("Nenhuma combinação foi avaliada")
    model.load_state_dict(best_state, strict=True)
    new_historical = _assess(model, historical, size, focus)
    new_online = _assess(model, online, size, focus)
    new_holdout = _assess(model, hold, size, focus)
    acceptance = candidate_acceptance(
        old_historical, new_historical, old_online, new_online,
        old_holdout, new_holdout, minimum_holdout_accuracy=.97,
    )
    when = datetime.now(timezone.utc)
    out = root / "reports/faltando_neural/archive_blends" / (
        when.strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:6]
    )
    out.mkdir(parents=True, exist_ok=False)
    candidate_path = out / "faltando_v2_weight_blend_experimental.pt"
    torch.save({
        **base_checkpoint,
        "state_dict": {key: value.detach().cpu().clone()
                       for key, value in model.state_dict().items()},
        "experimental": True, "production_approved": False,
        "archive_labels_experimental": True,
        "individual_pair_review_complete": False,
        "source_champion_sha256": _sha(base_path),
        "provisional_archive_manifest_sha256": provenance["manifest_sha256"],
        "weight_blend_offline": True,
        "blend_alpha_aggressive": float(best_alpha),
    }, candidate_path)
    result = {
        "schema": "visionx.faltando_v2_weight_blend_experiment.v1",
        "created_at_utc": when.isoformat(),
        "candidate_only": True, "production_approval_eligible": False,
        "promoted": False, "automatic_ok_authorized": False,
        "individual_pair_review_complete": False,
        "holdout_previously_examined_in_prior_experiments": True,
        "independent_validation_claim": False,
        "selection_data": ["historical", "online", "archive_train"],
        "holdout_used_for_selection": False,
        "base_checkpoint_sha256": _sha(base_path),
        "source_candidate_sha256": {
            "conservative": left_report["candidate_sha256"],
            "aggressive": right_report["candidate_sha256"],
        },
        "source_manifest_sha256": provenance["manifest_sha256"],
        "split": split,
        "blend_grid": grid,
        "selected_aggressive_weight": float(best_alpha),
        "candidate_relative_path": candidate_path.relative_to(root).as_posix(),
        "candidate_sha256": _sha(candidate_path),
        "old_historical": old_historical, "new_historical": new_historical,
        "old_online": old_online, "new_online": new_online,
        "old_holdout": old_holdout, "new_holdout": new_holdout,
        "operational_threshold_diagnostics": {
            "new_historical": _operational_light_summary(new_historical),
            "new_holdout": _operational_light_summary(new_holdout),
        },
        "acceptance": acceptance,
        "limitations": [
            "Mesmos 87 holdout do treino anterior; avaliação NÃO é cega.",
            "Blending de checkpoints é ensaio offline, não prova de melhorias.",
            "Sem comprovação humana individual dos rótulos das pastas.",
            "Proibido promover sem validação nova, independente, e regressões de todas categorias.",
        ],
    }
    report_path = out / "blend_report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Melhor combinação (somente dados de desenvolvimento): {best_alpha:.2f}",
          flush=True)
    print(
        f"Holdout final: {new_holdout['correct_images']}/{new_holdout['images']}",
        flush=True,
    )
    print(
        "Gates:", "APROVADOS (EXPERIMENTAIS)" if acceptance["passed"]
        else "REPROVADOS", "— SEM PROMOÇÃO", flush=True,
    )
    print("Relatório:", report_path, flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Combinação offline de dois checkpoints CNN V2")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conservative-report", type=Path, required=True)
    parser.add_argument("--aggressive-report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args(argv)
    blend_offline(
        args.root, args.manifest, args.conservative_report,
        args.aggressive_report, steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
