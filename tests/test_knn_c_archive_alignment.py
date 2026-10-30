"""Mesma métrica de anomalia para PNG indexado e julgamento humano."""
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import cv2
import numpy as np

from src.core.anomaly_signature import build_anomaly_signature
from src.core.archive_persistent_memory import (
    PersistentArchiveKNN, make_record, make_operator_record,
)
from src.core.confirmed_visual_recall import ConfirmedVisualRecall
from src.core.knn_c_archive_alignment import KNNArchiveAlignment
from src.core.side_memory_first import build_side_memory_first, memory_first_side_decision
from src.ui.production_confidence_gate import production_decision_policy
from tests.test_side_memory_first import case, good_probe


def sample():
    reference = np.full((230, 255, 3), 128, dtype=np.uint8)
    test = reference.copy()
    cv2.rectangle(test, (88, 70), (160, 180), (10, 30, 190), -1)
    return reference, test


def persist(store, label, ref, test, *, archive=True, human=True, category="FALTANDO"):
    sig = build_anomaly_signature(ref, test, {}, {"category": category})
    entries = []
    if archive:
        entries.append(make_record(
            label=label,
            source_path="public/" + ("ok_archive" if label == "OK" else "ng_archive") + "/memory_sample.png",
            source_sha256=sha256((label+"png").encode()).hexdigest(),
            category=category, lighting="TOP",
            reference=ref, test=test, anomaly_signature=sig,
        ))
    if human:
        entries.append(make_operator_record(
            label=label, source="button", event_id="human",
            category=category, lighting="TOP",
            reference=ref, test=test, anomaly_signature=sig,
        ))
    store.add(entries)


class KNNArchiveAlignmentTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = PersistentArchiveKNN(Path(tmp.name))
        self.engine = KNNArchiveAlignment(self.store)
        self.human = ConfirmedVisualRecall(self.store)
        self.ref, self.tst = sample()

    def recall(self, category="FALTANDO", test=None):
        info = {"category": category, "lighting_mode": "SIDE"}
        human = self.human.lookup(self.ref, self.tst if test is None else test, info)
        aligned = self.engine.diagnose(self.ref, self.tst if test is None else test, info, human)
        return human, aligned

    def test_link_operator_and_archive_on_same_metric_without_forging_ab_scores(self):
        persist(self.store, "OK", self.ref, self.tst)
        modified = self.tst.copy()
        modified[100, 110, 1] ^= 1
        h, a = self.recall(test=modified)
        self.assertEqual(h["status"], "HUMAN_OK_VISUAL_CANDIDATE")
        self.assertEqual(a["status"], "RECONCILED_OK")
        self.assertGreaterEqual(a["archive_similarity"], .97)
        self.assertGreaterEqual(a["human_archive_similarity"], .97)
        self.assertEqual(a["archive_source"]["label"], "OK")
        self.assertTrue(a["diagnostic_only"])
        self.assertFalse(a["auto_decision_authorized"])
        self.assertEqual(a["source_relation"], "SHARED_ANOMALY_SIGNATURE_NOT_RGB_EXACT")

        # Reprodução do bloqueio de 08:47: A/B antigos preferem NG com 80%.
        data = case()
        data["detail"]["knn_c"]["neighbors_A"][0].update(
            label="NG", similarity=.84487131,
            path="public/ng_archive/old.png",
            source_pixel_sha256="d"*64,
        )
        data["detail"]["knn_c"]["neighbors_B"][0].update(
            label="NG", similarity=.79551278,
            path="public/ng_archive/old.png",
            source_pixel_sha256="d"*64,
        )
        data["detail"]["knn_c"]["vote_B"] = "NG"
        data["detail"]["confirmed_visual_recall"] = h
        data["detail"]["knn_c"]["aligned_archive"] = a
        result = build_side_memory_first(
            data, "FALTANDO", self.ref, modified, probe=lambda *_: good_probe(),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["detail"]["side_memory_first"]["knn_agreement_mode"],
                         "ALIGNED_HUMAN_ARCHIVE_JSON")
        self.assertEqual(memory_first_side_decision(result), "OK")
        self.assertEqual(production_decision_policy(result)["proposed_decision"], "OK")
        # Continua honesto sobre A/B originais; não diz que foram 97%.
        self.assertLess(result["detail"]["side_memory_first"]["knn_a_similarity"], .90)

    def test_no_archive_cannot_approve_a_human_candidate(self):
        persist(self.store, "OK", self.ref, self.tst, archive=False)
        h, a = self.recall()
        self.assertEqual(h["status"], "HUMAN_OK_VISUAL_CANDIDATE")
        self.assertEqual(a["status"], "NO_MATCH")

    def test_no_human_confirmation_cannot_approve_archive_only(self):
        persist(self.store, "OK", self.ref, self.tst, human=False)
        h, a = self.recall()
        self.assertNotEqual(h["status"], "HUMAN_OK_VISUAL_CANDIDATE")
        self.assertEqual(a["status"], "NO_MATCH")

    def test_opposite_label_same_visual_signature_is_conflict(self):
        persist(self.store, "OK", self.ref, self.tst)
        persist(self.store, "NG", self.ref, self.tst)
        h, a = self.recall()
        self.assertEqual(h["status"], "AMBIGUOUS")
        self.assertEqual(a["status"], "NO_MATCH")

    def test_index_refreshes_without_restart_after_new_archive(self):
        persist(self.store, "OK", self.ref, self.tst, archive=False)
        self.assertEqual(self.recall()[1]["status"], "NO_MATCH")
        persist(self.store, "OK", self.ref, self.tst, human=False)
        self.assertEqual(self.recall()[1]["status"], "RECONCILED_OK")

    def test_new_route_fails_closed_on_strong_ng_a_or_b(self):
        persist(self.store, "OK", self.ref, self.tst)
        h, a = self.recall()
        data = case()
        data["detail"]["confirmed_visual_recall"] = h
        data["detail"]["knn_c"]["aligned_archive"] = a
        data["detail"]["knn_c"]["neighbors_B"][0].update(
            label="NG", similarity=.92,
        )
        data["detail"]["knn_c"]["vote_B"] = "NG"
        self.assertIsNone(build_side_memory_first(
            data, "FALTANDO", self.ref, self.tst, probe=lambda *_: good_probe()
        ))

    def test_category_isolation_and_invalid_source(self):
        persist(self.store, "OK", self.ref, self.tst, category="DESLOCADO")
        h, a = self.recall(category="FALTANDO")
        self.assertEqual(a["status"], "NO_MATCH")
        h2, a2 = self.recall(category="DESLOCADO")
        self.assertEqual(a2["status"], "RECONCILED_OK")
        broken = dict(h2)
        broken["best_ok_source"] = {"record_id": "bad", "source": "MODEL", "label": "OK"}
        self.assertEqual(self.engine.diagnose(
            self.ref, self.tst, {"category": "DESLOCADO"}, broken,
        )["status"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
