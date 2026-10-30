"""Prova funcional: julgamento humano no JSON participa da busca visual.

Este banco não tem fotos do arquivo local e não autoriza comandos XP.
"""
from __future__ import annotations

from hashlib import sha256
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.core.anomaly_signature import (
    build_anomaly_signature, compare_anomaly_signatures,
)
from src.core.archive_persistent_memory import (
    PersistentArchiveKNN, make_operator_record, make_record,
)
from src.core.confirmed_visual_recall import ConfirmedVisualRecall
from src.core.verified_memory_router import (
    VerifiedKNNMemory, install_memory_first_router,
)


def pair():
    reference = np.full((95, 130, 3), 160, dtype=np.uint8)
    test = reference.copy()
    cv2.rectangle(test, (58, 32), (91, 70), (28, 45, 120), -1)
    return reference, test


def signature(ref, test):
    return build_anomaly_signature(
        ref, test, {}, {"category": "FALTANDO"}
    )


def operator(label, light="TOP", *, images=None):
    reference, test = images if images is not None else pair()
    return make_operator_record(
        label=label, source="button", event_id=f"operator_{label}_{light}",
        category="FALTANDO", lighting=light, reference=reference, test=test,
        anomaly_signature=signature(reference, test),
    )


class ConfirmedVisualRecallTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = PersistentArchiveKNN(Path(tmp.name))
        self.memory = ConfirmedVisualRecall(self.store)

    def lookup(self, ref=None, test=None, *, category="FALTANDO", lighting="SIDE"):
        a, b = pair()
        return self.memory.lookup(
            a if ref is None else ref, b if test is None else test,
            {"category": category, "lighting_mode": lighting},
        )

    def test_same_operator_ok_from_top_is_recalled_on_side_with_pixel_change(self):
        self.store.add([operator("OK", "TOP")])
        ref, test = pair()
        altered = test.copy()
        altered[40, 80, 1] ^= 1
        out = self.lookup(ref, altered)
        self.assertEqual(out["status"], "HUMAN_OK_VISUAL_CANDIDATE")
        self.assertTrue(out["cross_light_candidate"])
        self.assertEqual(out["best_ok_source"]["lighting_mode"], "TOP")
        self.assertGreaterEqual(out["best_ok_similarity"], .975)
        self.assertFalse(out["auto_ok_authorized"])

    def test_only_archive_png_is_not_accepted_as_human_confirmation(self):
        ref, tst = pair()
        self.store.add([make_record(
            label="OK", source_path="public/ok_archive/test.png",
            source_sha256=sha256(b"file").hexdigest(),
            category="FALTANDO", lighting="TOP",
            reference=ref, test=tst,
            anomaly_signature=signature(ref, tst),
        )])
        self.assertEqual(self.lookup()["status"], "NO_MATCH")

    def test_conflicting_operator_labels_do_not_promote_ok(self):
        self.store.add([operator("OK", "TOP"), operator("NG", "MID")])
        output = self.lookup()
        self.assertEqual(output["status"], "AMBIGUOUS")
        self.assertEqual(output["best_ok_similarity"], output["best_ng_similarity"])
        self.assertFalse(output["auto_ok_authorized"])

    def test_index_refreshes_after_new_human_judgment(self):
        self.assertEqual(self.lookup()["status"], "NO_MATCH")
        self.store.add([operator("OK", "TOP")])
        self.assertEqual(self.lookup()["status"], "HUMAN_OK_VISUAL_CANDIDATE")
        # Categoria inconsistente não reaproveita o resultado anterior.
        self.assertEqual(self.lookup(category="DESLOCADO")["status"], "NO_MATCH")

    def test_vectorized_scores_match_anomaly_signature_comparator(self):
        ref, tst = pair()
        a = signature(ref, tst)
        shifted = tst.copy()
        shifted[36:40, 60:65] = (44, 58, 141)
        b = signature(ref, shifted)
        query = np.asarray(a["vector"], dtype=np.float32)
        data = np.asarray([a["vector"], b["vector"]], dtype=np.float32)
        vector_scores = self.memory._similarities(query, data)
        for index, item in enumerate((a, b)):
            expected, _ = compare_anomaly_signatures(a, item)
            self.assertAlmostEqual(float(vector_scores[index]), expected, places=6)

    def test_actual_memory_first_pipeline_exposes_operator_recall(self):
        self.store.add([operator("OK", "TOP")])

        class Base:
            def __init__(self):
                self.experts = {"knn": type("KNN", (), {
                    "signatures_ok": [], "signatures_ng": [],
                })()}
            def reload_memory(self):
                pass
            def inspect(self, ref, tst, raw, info, boxes, epicenters):
                return {
                    "verdict": "DEFEITO REAL", "is_defect": True,
                    "confidence": .999, "detail": {"cnn_v2_active": True},
                }

        class Runtime(Base):
            pass

        install_memory_first_router(
            Runtime, memory=VerifiedKNNMemory(archive_store=self.store)
        )
        ref, tst = pair()
        slightly_changed = tst.copy()
        slightly_changed[40, 80, 1] ^= 1
        result = Runtime().inspect(
            ref, slightly_changed, [], {"category": "FALTANDO",
                         "lighting_mode": "SIDE"}, {}, [],
        )
        self.assertEqual(result["detail"]["recognition_route"], "NEW_CNN")
        self.assertEqual(
            result["detail"]["confirmed_visual_recall"]["status"],
            "HUMAN_OK_VISUAL_CANDIDATE",
        )
        self.assertEqual(result["verdict"], "DEFEITO REAL")  # Sem auto-OK.


if __name__ == "__main__":
    unittest.main()
