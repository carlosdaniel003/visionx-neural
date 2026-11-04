"""Treinamento REAL da CNN V3 por pares AOI (somente offline).

Usa o conjunto gabarito/teste extraído do acervo, nunca votos KNN ou memória
como entrada. Pasta OK/NG é rótulo PROVISÓRIO; exige opt-in explícito.
O checkpoint tem schema incompatível com o ODIN produtivo e não pode
atualizar live_active.json, XP, memória ou o acervo.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.core.neural.aoi_quality_cnn_v3 import AOIQualityCNNV3, MODEL_SCHEMA_V3
from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2
from src.services.aoi_v3_binary_quality_metrics import (
    binary_quality_metrics, quality_coverage,
)
from src.services.aoi_v3_verified_candidate import load_verified_v3_candidate
from src.services.aoi_v3_operator_image_reviews import REVIEW_PATH
from src.scripts.train_aoi_universal_quality_offline import (
    _pinned_initialization, _trial_readonly_snapshot,
)
from src.scripts.train_faltando_cnn_v2_online import (
    PairDataset, _assess, _supervised_loss,
)
from src.services.aoi_universal_quality_archive import (
    split_universal_events, universal_archive_events,
)

SCHEMA = "visionx.aoi_quality_cnn_v3_candidate_only.v1"
REPORT_SCHEMA = "visionx.aoi_quality_cnn_v3_training_report.v1"


class PhotometricPairDataset(PairDataset):
    """Perturba luzes sem alterar o rótulo de qualidade do par.

    Gabarito e teste recebem ajustes próprios de iluminação: assim o
    classificador não pode usar diferença de brilho como atalho confiável.
    A rotação é aplicada igualmente aos quatro recortes, preservando a
    relação física gabarito/teste em todas as iluminações.
    """

    def __init__(self, events: list[dict], size: int, fraction: float, *,
                 augment: bool = False):
        super().__init__(events, size, fraction)
        self.augment = augment

    def __getitem__(self, index: int):
        ref, test, focus_ref, focus_test, mask, truth = super().__getitem__(index)
        if not self.augment:
            return ref, test, focus_ref, focus_test, mask, truth
        images = [ref, test, focus_ref, focus_test]
        quarter_turn = int(torch.randint(0, 4, ()).item())
        if quarter_turn:
            images = [torch.rot90(t, quarter_turn, (-2, -1)) for t in images]
        # Mesma transformação dentro de full/focus de cada lado do par.
        ref, test, focus_ref, focus_test = images
        for light in range(3):
            if mask[light] == 0:
                continue
            for side, focus_side in ((ref, focus_ref), (test, focus_test)):
                gain = float(torch.empty(()).uniform_(.88, 1.12))
                bias = float(torch.empty(()).uniform_(-.035, .035))
                side[light] = (side[light] * gain + bias).clamp(0., 1.)
                focus_side[light] = (
                    focus_side[light] * gain + bias
                ).clamp(0., 1.)
        return ref, test, focus_ref, focus_test, mask, truth


def _selection_score(metrics: dict) -> tuple:
    """Escolha baseada SOMENTE na validação, com as duas classes."""
    if metrics["NG_total"] < 1 or metrics["OK_total"] < 1:
        raise ValueError("Validação sem rótulos OK e NG nas duas classes")
    return (
        round(metrics["balanced_accuracy"], 10),
        -metrics["NG_false_OK"],
        -metrics["OK_false_NG"],
    )


def _loss_v3(model, batch, *, identity_weight: float = .10):
    ref, test, focus_ref, focus_test, mask, truth = batch
    logits, lights = model(ref, test, focus_ref, focus_test, mask)
    supervised = _supervised_loss(
        logits, lights, mask, truth, light_weight=.20,
    )
    # Regularizador de consistência de par, não exemplo novo de defeito.
    # A referência AOI comparada consigo mesma não tem divergência.
    identity_logits, identity_lights = model(
        ref, ref, focus_ref, focus_ref, mask,
    )
    identity = _supervised_loss(
        identity_logits, identity_lights, mask,
        torch.zeros_like(truth), light_weight=.20,
    )
    return supervised + identity_weight * identity, {
        "supervised": float(supervised.detach().item()),
        "identity_reference": float(identity.detach().item()),
    }


def train_v3(
    root: Path, manifest: Path, *, epochs: int = 6, seed: int = 42,
    image_size: int = 96, learning_rate: float = .0002,
    validation_fraction: float = .20, test_fraction: float = .20,
    allow_provisional_archive_labels: bool = False,
    allow_verified_trial_pointer_readonly: bool = False,
    compare_verified_pinned_cnn: bool = True,
    initialize_from_v3_report: Path | None = None,
) -> dict:
    if allow_provisional_archive_labels is not True:
        raise ValueError(
            "Rótulos das pastas OK/NG são provisórios; use "
            "--allow-provisional-archive-labels somente para teste OFFLINE"
        )
    if (
        type(epochs) is not int or not 1 <= epochs <= 20
        or type(seed) is not int or isinstance(seed, bool)
        or type(image_size) is not int or image_size not in {64, 96, 128, 160}
        or not 0 < learning_rate <= .001
        or not .10 <= validation_fraction <= .30
        or not .10 <= test_fraction <= .30
        or validation_fraction + test_fraction > .50
    ):
        raise ValueError("Hiperparâmetros CNN V3 inválidos")
    root = Path(root).resolve()
    manifest = Path(manifest)
    manifest = (manifest if manifest.is_absolute() else root/manifest).resolve()
    if (
        not manifest.is_file() or manifest.name != "manifest.json"
        or not manifest.is_relative_to(root/"reports/faltando_neural")
        or manifest.is_symlink()
    ):
        raise ValueError("Manifesto AOI não pertence ao staging offline")
    pointer_before = _trial_readonly_snapshot(
        root, opt_in=allow_verified_trial_pointer_readonly,
    )
    torch.set_num_threads(max(1, min(2, torch.get_num_threads())))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Novo inventário pode crescer sem invalidar nove decisões humanas:
    # o importador revalida SHA-256 de cada PNG e dos dois recortes AOI.
    events, provenance = universal_archive_events(
        manifest, allow_refreshed_validation_labels=True,
        use_v3_image_quality_reviews=True,
    )
    human_train_sources = {
        source for row in events
        if row.get("label_provenance") == "HUMAN_V3_IMAGE_SHA256"
        for source in row["sources"]
    }
    if len(human_train_sources) != 14:
        raise ValueError("CNN V3 exige 14 imagens humanas verificadas por SHA")
    # Remover a categoria AOI dos exemplos usados pela CNN V3. O nome
    # original do PNG é somente a localização para recuperar seus pixels.
    events = [dict(row) for row in events]
    for row in events:
        row.pop("category_hint", None)
    provenance.pop("by_category_and_label", None)
    if any(len(e["views"]) != 1 for e in events):
        raise ValueError("Grupo multilight sem vínculo físico confirmado")
    train, validation, test, partition = split_universal_events(
        events, seed=seed, validation_fraction=validation_fraction,
        test_fraction=test_fraction, balance_by_aoi_category=False,
        pinned_training_sources=human_train_sources,
    )
    if (
        len({src for row in train for src in row["sources"]}
            & human_train_sources) != 14
        or any(
            source in human_train_sources
            for fold in (validation, test)
            for row in fold for source in row["sources"]
        )
    ):
        raise ValueError("Vazamento de correção humana entre treino/teste V3")
    coverage = {
        "all_data": quality_coverage(events),
        "train": quality_coverage(train),
        "validation": quality_coverage(validation),
        "test": quality_coverage(test),
    }
    if any(len(fold) < 2 for fold in (train, validation, test)):
        raise ValueError("Partições insuficientes para treino/verificação")
    if any({r["label"] for r in fold} != {"OK", "NG"}
           for fold in (train, validation, test)):
        raise ValueError("Todas as partições precisam de OK e NG")

    # Checkpoint original verificado por SHA, apenas para comparação NO MESMO
    # teste (não é treino, teacher nem pré-treinamento da V3).
    base_model = None
    baseline_sha = None
    baseline_preprocessing = None
    if compare_verified_pinned_cnn:
        _, pinned, baseline_sha = _pinned_initialization(root)
        base_size = pinned.get("image_size")
        base_focus = pinned.get("focus_fraction")
        if (
            type(base_size) is not int or not 64 <= base_size <= 512
            or not isinstance(base_focus, (int, float))
            or isinstance(base_focus, bool) or not .5 <= base_focus <= .90
        ):
            raise ValueError("Pré-processamento do checkpoint V2 não verificado")
        baseline_preprocessing = {
            "image_size": base_size, "focus_fraction": float(base_focus),
        }
        base_model = FaltandoCNNV2()
        base_model.load_state_dict(pinned["state_dict"], strict=True)
    v3_source = None
    if initialize_from_v3_report is not None:
        model, v3_source = load_verified_v3_candidate(
            root, initialize_from_v3_report, provenance["manifest_sha256"],
        )
        if image_size != v3_source["image_size"]:
            raise ValueError("A CNN V3 anterior exige a mesma resolução de treino")
    else:
        model = AOIQualityCNNV3()
    sample_counts = Counter(row["label"] for row in train)
    # Balancear por qualidade física/provisória, nunca por categoria AOI.
    # Amostras conferidas pelo operador têm reforço moderado, sem dominar
    # centenas de exemplos históricos do mesmo conjunto.
    sample_weights = [
        (2.0 if row.get("label_provenance") == "HUMAN_V3_IMAGE_SHA256" else 1.0)
        / sample_counts[row["label"]]
        for row in train
    ]
    loader = DataLoader(
        PhotometricPairDataset(train, image_size, .70, augment=True),
        batch_size=4,
        sampler=WeightedRandomSampler(
            torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(train), replacement=True,
            generator=torch.Generator().manual_seed(seed),
        ), num_workers=0,
    )
    # A validação permanece congelada; augmentation só no treino.
    initial = binary_quality_metrics(
        validation, _assess(model, validation, image_size, .70),
    )
    best_key = _selection_score(initial)
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    trajectory = [{"epoch": 0, "validation": initial, "selected": True}]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=.01,
    )

    for epoch in range(1, epochs+1):
        model.train()
        accumulated = 0.0
        for batch in loader:
            loss, components = _loss_v3(model, batch)
            if not bool(torch.isfinite(loss).item()):
                raise ValueError("Loss CNN V3 não finita")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            accumulated += float(loss.detach().item())
        validation_scores = binary_quality_metrics(
            validation, _assess(model, validation, image_size, .70),
        )
        rank = _selection_score(validation_scores)
        # Se houver base anterior, não aceitar uma melhora de média que
        # aumente um dos dois tipos de erro nesta validação.
        not_worse_than_start = all(
            validation_scores[key] <= initial[key]
            for key in ("NG_false_OK", "OK_false_NG")
        )
        selected = rank > best_key and (
            v3_source is None or not_worse_than_start
        )
        if selected:
            best_key = rank
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        trajectory.append({
            "epoch": epoch,
            "training_loss": round(accumulated / len(loader), 7),
            "validation": validation_scores,
            "selected": selected,
        })
        print(
            f"CNN V3 OFFLINE {epoch}/{epochs} "
            f"val_OK_falsos_NG={validation_scores['OK_false_NG']} "
            f"val_NG_falsos_OK={validation_scores['NG_false_OK']} "
            f"melhor_epoca={best_epoch}",
            flush=True,
        )

    model.load_state_dict(best_state)
    final_validation = binary_quality_metrics(
        validation, _assess(model, validation, image_size, .70),
    )
    # TESTE somente agora; não escolhe época, nem corrige limiares.
    final_test_rows = _assess(model, test, image_size, .70)
    final_test = binary_quality_metrics(test, final_test_rows)
    baseline_v3_test = None
    if v3_source is not None:
        old_v3, _ = load_verified_v3_candidate(
            root, initialize_from_v3_report, provenance["manifest_sha256"],
        )
        baseline_v3_test = binary_quality_metrics(
            test, _assess(old_v3, test, image_size, .70),
        )
    baseline_test = None
    if base_model is not None:
        baseline_test = binary_quality_metrics(
            test,
            _assess(
                base_model, test,
                baseline_preprocessing["image_size"],
                baseline_preprocessing["focus_fraction"],
            ),
        )
    if _trial_readonly_snapshot(
        root, opt_in=allow_verified_trial_pointer_readonly,
    ) != pointer_before:
        raise ValueError("Ponteiro ativo alterado durante experimento")
    if manifest.is_symlink() or sha256(manifest.read_bytes()).hexdigest() != provenance["manifest_sha256"]:
        raise ValueError("Manifesto modificado durante treino")
    if compare_verified_pinned_cnn:
        _, _, digest = _pinned_initialization(root)
        if digest != baseline_sha:
            raise ValueError("Modelo baseline alterado durante experimento")
    if v3_source is not None:
        _, rechecked = load_verified_v3_candidate(
            root, initialize_from_v3_report, provenance["manifest_sha256"],
        )
        if rechecked != v3_source:
            raise ValueError("Candidato V3 inicial alterado durante experimento")

    stamp = datetime.now(timezone.utc)
    folder = root/"reports/aoi_quality_cnn_v3/candidates"/(
        stamp.strftime("%Y%m%dT%H%M%S_%fZ")+"_"+uuid4().hex[:6]
    )
    folder.mkdir(parents=True, exist_ok=False)
    candidate_path = folder/"aoi_quality_cnn_v3_candidate.pt"
    torch.save({
        "schema": SCHEMA,
        "architecture_schema": MODEL_SCHEMA_V3,
        "state_dict": {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()},
        "image_size": image_size,
        "focus_fraction": .70,
        "lights": ("SIDE", "TOP", "MID"),
        "source_manifest_sha256": provenance["manifest_sha256"],
        "initial_v3_candidate_sha256": (
            v3_source["candidate_sha256"] if v3_source else None
        ),
        "aoi_category_used_for_training": False,
        "operator_verified_training_sources": 14,
        "baseline_v2_sha256": baseline_sha,
        "trained_with_provisional_archive_labels": True,
        "aoi_category_is_not_physical_defect": True,
        "source_training": "REFERENCE_TEST_PAIRS_IMAGE_ONLY_QUALITY_BINARY",
        "identity_reference_regularization": .10,
        "candidate_only": True,
        "production_approved": False,
        "automatic_xp_authorized": False,
        "selected_epoch": best_epoch,
    }, candidate_path)
    v3_hash = sha256(candidate_path.read_bytes()).hexdigest()
    report = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": stamp.isoformat(),
        "event": "OFFLINE_AOI_PAIR_QUALITY_CNN_V3",
        "candidate_only": True,
        "production_approved": False,
        "automatic_xp_authorized": False,
        "live_active_modified": False,
        "archive_modified": False,
        "memory_or_knn_used_for_inference": False,
        "aoi_category_used_for_training_selection_evaluation": False,
        "operator_image_review_file": REVIEW_PATH.as_posix(),
        "operator_image_review_verified": 14,
        "operator_image_review_ignored": 1,
        "operator_images_reserved_exclusively_for_training": True,
        "training_performed": True,
        "training_uses_operator_quality_when_verified": True,
        "training_uses_provisional_archive_labels": True,
        "provenance": provenance,
        "coverage": coverage,
        "partition": partition,
        "training": {
            "epochs": epochs, "seed": seed, "learning_rate": learning_rate,
            "image_size": image_size, "focus_fraction": .70,
            "identity_loss_weight": .10,
            "independent_reference_test_photometric_augmentation": True,
            "common_pair_quarter_turn_augmentation": True,
            "selected_epoch_using_validation_only": best_epoch,
            "test_not_used_for_selection": True,
            "quality_only_balancing": True,
            "per_quality_inverse_frequency_sampling": True,
            "operator_verified_weight_multiplier": 2.0,
            "v3_baseline_initialized_from_verified_report": bool(v3_source),
            "selected_epoch_cannot_worsen_either_validation_error_against_start":
                bool(v3_source),
        },
        "epoch_history": trajectory,
        "validation": final_validation,
        "test": final_test,
        "test_cases": final_test_rows["cases"],
        "verified_pinned_v2_baseline": baseline_test,
        "verified_previous_v3_baseline": baseline_v3_test,
        "initial_v3_candidate": v3_source,
        "baseline_v2_sha256": baseline_sha,
        "baseline_v2_native_preprocessing": baseline_preprocessing,
        "comparison_on_same_test": {
            "v3_ng_false_ok": final_test["NG_false_OK"],
            "v3_ok_false_ng": final_test["OK_false_NG"],
            "v2_ng_false_ok": (
                baseline_test["NG_false_OK"] if baseline_test else None
            ),
            "v2_ok_false_ng": (
                baseline_test["OK_false_NG"] if baseline_test else None
            ),
            "no_new_ng_escapes": bool(
                baseline_test and final_test["NG_false_OK"] <= baseline_test["NG_false_OK"]
            ),
            "reduced_false_ng": bool(
                baseline_test and final_test["OK_false_NG"] < baseline_test["OK_false_NG"]
            ),
            "candidate_not_authorized_even_if_both_pass": True,
        },
        "safety": {
            "all_quality_labels_operator_verified": coverage["all_data"]["fully_operator_verified"],
            "each_available_light_has_both_labels": coverage["test"]["each_available_light_has_both_labels"],
            "operator_verified_14_excluded_from_test": True,
            "former_v3_test_no_longer_blind": True,
            "physical_event_independence_proven": False,
            "blind_holdout": False,
            "production_eligible": False,
        },
        "candidate_path": candidate_path.relative_to(root).as_posix(),
        "candidate_sha256": v3_hash,
        "limitations": [
            "Rótulo da pasta não significa inspeção física confirmada.",
            "Algumas iluminações podem não conter ambas as qualidades reais.",
            "A categoria AOI não é utilizada para treinar, balancear ou selecionar a V3.",
            "As 14 correções humanas são treino, jamais pontuação de teste.",
            "Testes de arquivo histórico já vistos NÃO são avaliação cega.",
            "Não utilizar modelo nem limiar V3 na estação sem aprovação separada.",
            "Não enviar 0/1 ao XP nem atualizar memória com previsão da CNN.",
        ],
    }
    out = folder/"training_report.json"
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+"\n",
        encoding="utf-8",
    )
    print("CANDIDATO CNN V3 gerado, SEM instalação: " + str(candidate_path))
    print("RELATÓRIO: " + str(out))
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CNN V3 AOI — treinamento offline isolado")
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=.0002)
    p.add_argument("--allow-provisional-archive-labels", action="store_true")
    p.add_argument("--allow-verified-trial-pointer-readonly", action="store_true")
    p.add_argument("--skip-verified-pinned-cnn-comparison", action="store_true")
    p.add_argument("--initialize-from-v3-report", type=Path)
    a = p.parse_args(argv)
    result = train_v3(
        a.root, a.manifest, epochs=a.epochs, seed=a.seed,
        image_size=a.image_size, learning_rate=a.lr,
        allow_provisional_archive_labels=a.allow_provisional_archive_labels,
        allow_verified_trial_pointer_readonly=a.allow_verified_trial_pointer_readonly,
        compare_verified_pinned_cnn=not a.skip_verified_pinned_cnn_comparison,
        initialize_from_v3_report=a.initialize_from_v3_report,
    )
    print(json.dumps({
        "selected_epoch": result["training"]["selected_epoch_using_validation_only"],
        "v3_false_ng": result["test"]["OK_false_NG"],
        "v3_ng_escaped": result["test"]["NG_false_OK"],
        "baseline_v2": result["verified_pinned_v2_baseline"],
        "candidate_path": result["candidate_path"],
        "production_approved": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
