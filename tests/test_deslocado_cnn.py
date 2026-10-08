"""Preparação/treino isolados CNN DESLOCADO com zero NG reais.

NÃO promete generalização. Testa fontes, máscaras multilight, modelos
candidatos experimentais e ausência de mudança no roteador produtivo.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from src.services.deslocado_neural_dataset import prepare_deslocado
from src.scripts.train_deslocado_cnn import (
    _proxy_displace, load_ok_events, train_deslocado,
)
from src.core.neural.deslocado_cnn import DeslocadoCNN, MODEL_SCHEMA_DESLOCADO


class DeslocadoFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/"public"/"ok_archive").mkdir(parents=True)
        (self.root/"public"/"ng_archive").mkdir(parents=True)
        self.frames = {}
        self.info = {}
        cases = [
            ("2026-10-02_1349_DESLOCADO.png", "SIDE", "LEGACY_DEFAULT", 12, "R12"),
            ("2026-10-02_1350_DESLOCADO.png", "SIDE", "LEGACY_DEFAULT", 23, "R23"),
            ("2026-10-02_1351_DESLOCADO.png", "SIDE", "LEGACY_DEFAULT", 34, "R34"),
            ("2026-10-08_0754_DESLOCADO_SIDE.png", "SIDE", "EXPLICIT_SUFFIX", 45, "R45"),
            ("2026-10-08_0754_DESLOCADO_TOP.png", "TOP", "EXPLICIT_SUFFIX", 46, "R45"),
            ("2026-10-08_0754_DESLOCADO_MID.png", "MID", "EXPLICIT_SUFFIX", 47, "R45"),
        ]
        self.inventory = {"images": []}
        for name, light, light_source, value, part in cases:
            source = self.root/"public"/"ok_archive"/name
            frame = np.full((95, 105, 3), value, dtype=np.uint8)
            frame[20:65, 27:75] = (5, 85, 195)
            valid, buf = cv2.imencode(".png", frame)
            self.assertTrue(valid)
            source.write_bytes(buf.tobytes())
            self.frames[value] = frame
            self.info[value] = {
                "board": "PCB-A", "parts": part,
                "category": "DESLOCADO", "value": "5 <= 20 <= 30 DESLOCADO"
            }
            self.inventory["images"].append({
                "path": source.relative_to(self.root).as_posix(),
                "file_sha256": sha256(source.read_bytes()).hexdigest(),
                "expected_label": "OK", "category_hint": "DESLOCADO",
                "lighting_mode": light, "lighting_source": light_source,
                "status": "VALID_PNG", "event_id": None,
            })
        # Verificar exclusão absoluta de outras categorias.
        self.inventory["images"].append({
            "path": "public/ok_archive/other_FALTANDO.png",
            "expected_label": "OK", "category_hint": "FALTANDO",
        })

    def extractor(self, frame):
        value = int(frame[0, 0, 0])
        return frame.copy(), frame.copy(), dict(self.info[value])

    def prepared(self):
        return prepare_deslocado(self.root, extractor=self.extractor,
                                inventory=self.inventory)


class DeslocadoPreparationTests(DeslocadoFixtures):
    def test_extracts_side_legacy_and_multilight_without_ng(self):
        report, folder = self.prepared()
        self.assertEqual(report["summary"]["total"], 6)
        self.assertEqual(report["summary"]["extracted"], 6)
        self.assertEqual(report["summary"]["real_ng_count"], 0)
        self.assertFalse(report["summary"]["real_ng_validation_ready"])
        self.assertEqual(report["summary"]["name_only_triplets"], 1)
        self.assertEqual(report["summary"]["by_lighting"],
                         {"SIDE": 4, "TOP": 1, "MID": 1})
        self.assertEqual(report["schema"], "visionx.deslocado_neural_preparation.v1")
        self.assertFalse(report["production_approved"])
        self.assertTrue((folder/"manifest.json").exists())
        self.assertFalse((self.root/"public"/"ok_archive"/"other_FALTANDO.png").exists())
        events, data = load_ok_events(folder/"manifest.json")
        self.assertEqual(len(events), 4)
        self.assertEqual(sum(len(x["observations"]) for x in events), 6)
        self.assertTrue(any(len(x["observations"]) == 3 for x in events))

    def test_source_hash_change_aborts_training(self):
        _, folder = self.prepared()
        origin = self.root/"public"/"ok_archive"/"2026-10-02_1349_DESLOCADO.png"
        origin.write_bytes(origin.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError, "original alterado"):
            load_ok_events(folder/"manifest.json")

    def test_archive_ng_real_blocks_synthetic_only_protocol(self):
        report, folder = self.prepared()
        manifest_path = folder/"manifest.json"
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["samples"][0]["expected_label_from_archive"] = "NG"
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "NG DESLOCADO reais"):
            train_deslocado(manifest_path, epochs=1, size=64)

    def test_new_lighting_copy_group_does_not_count_as_independent(self):
        report, folder = self.prepared()
        self.assertEqual(len(report["name_only_triplet_candidates"]), 1)
        tri = report["name_only_triplet_candidates"][0]
        self.assertEqual(set(tri["paths"]), {"SIDE", "TOP", "MID"})

    def test_augment_proxy_modifies_test_not_reference(self):
        ref = np.full((75, 87, 3), 35, dtype=np.uint8)
        ref[20:60, 20:60] = (185, 120, 40)
        original = ref.copy()
        proxy = _proxy_displace(ref, 42)
        self.assertEqual(proxy.shape, ref.shape)
        self.assertTrue(np.array_equal(ref, original))
        self.assertGreater(np.count_nonzero(proxy != ref), 0)


class DeslocadoTrainingTests(DeslocadoFixtures):
    def test_one_epoch_model_is_distinct_and_never_production_approved(self):
        _, location = self.prepared()
        source_before = {
            file: sha256(file.read_bytes()).hexdigest()
            for file in (self.root/"public"/"ok_archive").glob("*.png")
        }
        report, output = train_deslocado(
            location/"manifest.json", epochs=1, size=64, batch_size=2
        )
        self.assertEqual(report["total_real_ok_events"], 4)
        self.assertEqual(report["ng_real_events"], 0)
        self.assertIsNone(
            report["train_results"]["real_ng_recall"]
        )
        self.assertFalse(report["production_approved"])
        self.assertTrue(report["activation_disabled"])
        self.assertTrue((output/"training_report_deslocado.json").exists())
        self.assertTrue((output/"training_summary_deslocado.txt").exists())
        checkpoint = torch.load(
            output/"deslocado_cnn_candidate.pt", weights_only=True
        )
        self.assertEqual(checkpoint["schema"], MODEL_SCHEMA_DESLOCADO)
        self.assertFalse(checkpoint["production_approved"])
        self.assertFalse(checkpoint["allow_automatic_classification"])
        self.assertEqual(checkpoint["real_ng_used"], 0)
        self.assertEqual(
            {f: sha256(f.read_bytes()).hexdigest() for f in source_before},
            source_before
        )

    def test_model_forward_accepts_side_and_three_lights(self):
        model = DeslocadoCNN()
        ref = torch.rand(2, 3, 3, 64, 64)
        test = torch.rand_like(ref)
        mask = torch.tensor([[1.,0.,0.], [1.,1.,1.]])
        out, by_light = model(ref,test,ref,test,mask)
        self.assertEqual(tuple(out.shape), (2,))
        self.assertEqual(tuple(by_light.shape), (2,3))
        self.assertTrue(torch.allclose(out[0], by_light[0,0]))


if __name__ == "__main__":
    unittest.main()
