"""Qualificação humana de pares AOI FALTANDO — offline, independente do ODIN.

Nunca treina ou altera decisões em produção. O relatório é salvo ao lado do
manifesto da preparação, mantendo PNGs e rótulos originais inalterados.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

PREPARATION_SCHEMA = "visionx.faltando_neural_preparation.v1"
QUALIFICATION_SCHEMA = "visionx.faltando_qualification.v1"
REVIEW_STATES = frozenset(("CONFIRMED_OK", "CONFIRMED_NG", "REJECTED"))
GROUP_STATES = frozenset(("CONFIRMED_VISUAL_ASSOCIATION", "REJECTED"))
HEX64 = re.compile(r"^[a-f0-9]{64}$")
LIGHTS = ("SIDE", "TOP", "MID")


def _inside(parent: Path, candidate: Path) -> bool:
    return candidate == parent or parent in candidate.parents


def _read_image(path: Path) -> np.ndarray:
    try:
        frame = cv2.imdecode(
            np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
        )
    except (OSError, cv2.error, ValueError) as exc:
        raise ValueError(f"Imagem derivada ilegível: {path.name}: {exc}") from exc
    if frame is None or frame.ndim != 3 or frame.size == 0:
        raise ValueError(f"Imagem derivada ilegível: {path.name}")
    return frame


def _dhash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reduced = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    value = 0
    for flag in (reduced[:, 1:] > reduced[:, :-1]).flat:
        value = (value << 1) | int(flag)
    return value


def visual_similarities(samples: list[dict], run_dir: Path, limit: int = 4) -> dict:
    """Sinaliza semelhança perceptual; não deduplica nem decide rótulos.

    A comparação usa **ambos** os recortes (gabarito/teste) e iluminação
    compatível. Distâncias pequenas são suspeitas, NÃO prova de repetição.
    """
    signatures = []
    for sample in samples:
        if sample.get("status") != "EXTRACTED_PENDING_REVIEW":
            continue
        try:
            ref = _read_image(run_dir / sample["reference_path"])
            test = _read_image(run_dir / sample["test_path"])
        except (ValueError, TypeError):
            continue
        signatures.append((
            sample["source_path"], sample["lighting_mode"],
            _dhash(ref), _dhash(test),
        ))
    suggestions: dict[str, list[dict]] = defaultdict(list)
    for i, (one_path, one_light, ref_a, test_a) in enumerate(signatures):
        for two_path, two_light, ref_b, test_b in signatures[i + 1:]:
            if one_light != two_light:
                continue
            dr, dt = (ref_a ^ ref_b).bit_count(), (test_a ^ test_b).bit_count()
            if dr > 9 or dt > 9:
                continue
            for source, target in ((one_path, two_path), (two_path, one_path)):
                suggestions[source].append({
                    "other_path": target,
                    "reference_distance": dr,
                    "test_distance": dt,
                })
    return {
        path: sorted(items, key=lambda x: (
            x["reference_distance"] + x["test_distance"], x["other_path"]
        ))[:limit]
        for path, items in suggestions.items()
    }


class QualificationStore:
    """Controla revisão explícita e vínculos humanos sem afetar fontes."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if self.manifest_path.name != "manifest.json" or not self.manifest_path.is_file():
            raise ValueError("Selecione um manifest.json válido da preparação FALTANDO")
        self.run_dir = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != PREPARATION_SCHEMA:
            raise ValueError("Schema de preparação incompatível")
        if not self.run_dir.name.startswith("run_") or self.run_dir.parent.name != "faltando_neural":
            raise ValueError("Manifesto fora de reports/faltando_neural/run_*")
        if self.run_dir.parent.parent.name != "reports":
            raise ValueError("Manifesto fora da árvore de relatórios")
        self.root = Path(self.manifest["root"]).resolve()
        if not _inside(self.root / "reports", self.run_dir):
            raise ValueError("Manifesto não pertence à raiz indicada")
        self.samples = list(self.manifest.get("samples", []))
        self.by_path: dict[str, dict] = {}
        self.expected_labels: dict[str, str] = {}
        self.files: dict[str, tuple[Path, Path, Path]] = {}
        public = (self.root / "public").resolve()
        for sample in self.samples:
            path = sample["source_path"]
            if path in self.by_path:
                raise ValueError("Entrada duplicada no manifesto: " + path)
            if sample.get("expected_label_from_archive") not in ("OK", "NG"):
                raise ValueError("Rótulo original inválido: " + path)
            if sample.get("lighting_mode") not in LIGHTS:
                raise ValueError("Iluminação inválida: " + path)
            if not HEX64.fullmatch(str(sample.get("source_sha256", ""))):
                raise ValueError("SHA-256 da origem ausente/inválido: " + path)
            origin = (self.root / path).resolve()
            if not _inside(public, origin) or not (
                _inside(public / "ok_archive", origin)
                or _inside(public / "ng_archive", origin)
            ):
                raise ValueError("Origem fora dos arquivos AOI: " + path)
            if origin.is_symlink():
                raise ValueError("Fonte via symlink proibida")
            derived = []
            for field in ("reference_path", "test_path"):
                val = sample.get(field)
                if sample.get("status") == "EXTRACTED_PENDING_REVIEW" and not val:
                    raise ValueError("Par de treino incompleto: " + path)
                if val:
                    asset = (self.run_dir / val).resolve()
                    if not _inside(self.run_dir / "pairs", asset):
                        raise ValueError("Caminho derivado fora de pairs/: " + path)
                    derived.append(asset)
                else:
                    derived.append(None)
            self.by_path[path] = sample
            self.expected_labels[path] = sample["expected_label_from_archive"]
            self.files[path] = (origin, derived[0], derived[1])

        self.groups = {
            record["id"]: record
            for record in self.manifest.get("name_only_triplet_candidates", [])
        }
        if len(self.groups) != len(self.manifest.get("name_only_triplet_candidates", [])):
            raise ValueError("IDs de trinca repetidos")
        for name, record in self.groups.items():
            frames = record.get("paths", {})
            if set(frames) != set(LIGHTS) or len(set(frames.values())) != 3:
                raise ValueError("Trinca incompleta: " + name)
            if any(
                value not in self.by_path
                or self.by_path[value]["lighting_mode"] != mode
                or self.expected_labels[value] != record["label"]
                for mode, value in frames.items()
            ):
                raise ValueError("Trinca inconsistente: " + name)
        self.output = self.run_dir / "qualification.json"
        if self.output.is_symlink():
            raise ValueError("Não gravar em link simbólico")
        self.case_reviews: dict[str, dict] = {}
        self.group_reviews: dict[str, dict] = {}
        if self.output.exists():
            current = json.loads(self.output.read_text(encoding="utf-8"))
            if current.get("schema") != QUALIFICATION_SCHEMA:
                raise ValueError("Schema de qualificação já salva é incompatível")
            for path, review in current.get("case_reviews", {}).items():
                if (
                    path in self.by_path
                    and review.get("source_sha256")
                    == self.by_path[path]["source_sha256"]
                    and review.get("status") in REVIEW_STATES
                ):
                    self.case_reviews[path] = review
            for name, review in current.get("group_reviews", {}).items():
                if name not in self.groups or review.get("status") not in GROUP_STATES:
                    continue
                if review.get("sources_sha256") != self._group_sha(name):
                    continue
                self.group_reviews[name] = review

        self.similar = visual_similarities(self.samples, self.run_dir)

    def _validate_source(self, path: str) -> None:
        item = self.by_path[path]
        if item["status"] != "EXTRACTED_PENDING_REVIEW":
            raise ValueError("Exemplo não foi extraído com sucesso")
        origin, reference, test = self.files[path]
        if not origin.is_file() or origin.is_symlink():
            raise ValueError("Fonte AOI não existe ou é symlink")
        if sha256(origin.read_bytes()).hexdigest() != item["source_sha256"]:
            raise ValueError("Arquivo fonte foi modificado após preparar o dataset")
        for derived in (reference, test):
            if derived is None or not derived.is_file():
                raise ValueError("Gabarito/teste derivado indisponível")
            _read_image(derived)

    def _group_sha(self, name: str) -> dict[str, str]:
        return {
            mode: self.by_path[path]["source_sha256"]
            for mode, path in self.groups[name]["paths"].items()
        }

    def case_status(self, path: str) -> str:
        return self.case_reviews.get(path, {}).get("status", "PENDING")

    def group_status(self, name: str) -> str:
        return self.group_reviews.get(name, {}).get("status", "PENDING")

    def save(self) -> None:
        data = {
            "schema": QUALIFICATION_SCHEMA,
            "manifest_name": self.manifest_path.name,
            "manifest_sha256": sha256(self.manifest_path.read_bytes()).hexdigest(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "automated_training_allowed": False,
            "case_reviews": self.case_reviews,
            "group_reviews": self.group_reviews,
        }
        path = self.output.with_suffix(".json.tmp")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        path.replace(self.output)

    def review_case(self, path: str, status: str, notes: str = "") -> None:
        if status not in REVIEW_STATES:
            raise ValueError("Revisão de caso inválida")
        self._validate_source(path)
        self.case_reviews[path] = {
            "status": status,
            "source_sha256": self.by_path[path]["source_sha256"],
            "archive_label": self.expected_labels[path],
            "reviewed_label": (
                "OK" if status == "CONFIRMED_OK"
                else "NG" if status == "CONFIRMED_NG"
                else None
            ),
            "pair_inspected": True,
            "notes": notes.strip()[:500],
            "label_conflict": (
                status in ("CONFIRMED_OK", "CONFIRMED_NG")
                and status.split("_", 1)[1] != self.expected_labels[path]
            ),
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        # Se um caso for revisado de novo, qualquer vínculo anterior com ele
        # precisa ser confirmado novamente, evitando grupo desatualizado.
        for group_id, group in self.groups.items():
            if path in group["paths"].values():
                self.group_reviews.pop(group_id, None)
        self.save()

    def review_group(self, name: str, status: str, notes: str = "") -> None:
        if status not in GROUP_STATES or name not in self.groups:
            raise ValueError("Associação multilight inválida")
        group = self.groups[name]
        if status == "CONFIRMED_VISUAL_ASSOCIATION":
            for mode in LIGHTS:
                path = group["paths"][mode]
                self._validate_source(path)
                review = self.case_reviews.get(path, {})
                if review.get("status") != "CONFIRMED_" + group["label"]:
                    raise ValueError(
                        "Confirme cada par individualmente com o mesmo rótulo "
                        "antes de associar SIDE/TOP/MID"
                    )
        self.group_reviews[name] = {
            "status": status,
            "sources_sha256": self._group_sha(name),
            "human_group_id": name if status == "CONFIRMED_VISUAL_ASSOCIATION" else None,
            # Grupo humano não recupera nem inventa o event_id original.
            "original_event_id": None,
            "notes": notes.strip()[:500],
            "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def summary(self) -> dict[str, int]:
        statuses = [self.case_status(path) for path in self.by_path]
        return {
            "total": len(statuses),
            "confirmed_ok": statuses.count("CONFIRMED_OK"),
            "confirmed_ng": statuses.count("CONFIRMED_NG"),
            "rejected": statuses.count("REJECTED"),
            "pending": statuses.count("PENDING"),
            "confirmed_groups": sum(
                self.group_status(name) == "CONFIRMED_VISUAL_ASSOCIATION"
                for name in self.groups
            ),
        }


def latest_manifest(root: Path) -> Path:
    staging = Path(root).resolve() / "reports" / "faltando_neural"
    paths = sorted(
        staging.glob("run_*/manifest.json"),
        key=lambda p: p.parent.name,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(
            "Nenhum manifesto encontrado. Execute primeiro "
            "python -m src.services.faltando_neural_dataset"
        )
    return paths[0]
