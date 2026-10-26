"""Nine SHA-pinned operator-confirmed quality decisions from offline validation.

These are corrections to CNN predictions, NOT bulk archive relabeling:
six archive OK remain OK, three archive NG remain NG. All sources and both
prepared AOI pair crops must match the exact screenshots reviewed by operator.
Never creates a productive qualification, checkpoint, or XP command.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

SCHEMA = "visionx.operator_confirmed_validation_quality.v1"
EVIDENCE = Path("docs/odin_operator_validation_review_20261026.json")
SHA_RE = re.compile(r"^[a-f0-9]{64}$")


def verified_validation_labels(store) -> dict[str, dict]:
    """Validate all nine operator-verdict paths against local manifest/pair bytes."""
    root = store.root.resolve()
    file = root / EVIDENCE
    if file.is_symlink() or not file.is_file():
        raise ValueError("Revisão humana de validação ausente/symlink")
    doc = json.loads(file.read_text(encoding="utf-8"))
    if (doc.get("schema") != SCHEMA
            or doc.get("reviewed_on") != "2026-10-26"
            or doc.get("approved_for_production") is not False
            or doc.get("physical_event_groups_verified") is not False
            or not SHA_RE.fullmatch(str(doc.get("source_checkpoint_sha256")))
            or doc.get("source_manifest_sha256") !=
                sha256(store.manifest_path.read_bytes()).hexdigest()):
        raise ValueError("Revisão humana não corresponde ao manifesto original")
    entries = doc.get("labels")
    if not isinstance(entries, list) or len(entries) != 9:
        raise ValueError("Esperados exatamente 9 vereditos da validação")
    verdicts = {}
    numbers = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Caso de validação inválido")
        case = entry.get("case_number")
        source = entry.get("source_path")
        label = entry.get("quality_label")
        if (type(case) is not int or case not in range(1, 10)
                or case in numbers or not isinstance(source, str)
                or source in verdicts or label not in ("OK", "NG")
                or entry.get("operator_reviewed_pair") is not True):
            raise ValueError("Veredito humano ausente/repetido")
        numbers.add(case)
        src = Path(source)
        if (src.is_absolute() or len(src.parts) != 3 or ".." in src.parts
                or src.parts[:2] != ("public", label.lower() + "_archive")
                or src.suffix.lower() != ".png"):
            raise ValueError("Caminho do PNG não corresponde ao rótulo confirmado")
        row = store.by_path.get(source)
        if row is None or row.get("status") != "EXTRACTED_PENDING_REVIEW":
            raise ValueError("Caso confirmado não existe no manifesto: " + source)
        if (row["source_sha256"] != entry.get("source_sha256")
                or store.expected_labels[source] != label):
            raise ValueError("PNG de origem ou rótulo diverge do operador")
        store._validate_source(source)
        source_file, ref, test = store.files[source]
        for path, key in (
            (source_file, "source_sha256"),
            (ref, "reference_sha256"),
            (test, "test_sha256"),
        ):
            if (path.is_symlink() or not path.is_file()
                    or not SHA_RE.fullmatch(str(entry.get(key)))
                    or sha256(path.read_bytes()).hexdigest() != entry[key]):
                raise ValueError("Integridade SHA da revisão humana falhou: "
                                 + source + " [" + key + "]")
        verdicts[source] = entry
    if (numbers != set(range(1, 10))
            or Counter(x["quality_label"] for x in verdicts.values())
               != Counter({"OK": 6, "NG": 3})):
        raise ValueError("Vereditos humanos da validação incompletos")
    return verdicts
