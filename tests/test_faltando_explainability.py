"""Verifica Grad-CAM REAL, diferença contrafactual e projeção do encoder v2."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import torch

from src.core.neural.faltando_cnn_v2 import FaltandoCNNV2, MODEL_SCHEMA_V2, LIGHTS
from src.core.neural.faltando_live import FaltandoCNNLive
from src.core.neural.faltando_explainability import (
    FaltandoExplainability, generate_explainability_triplet, LAYER_NAME,
)


def pair():
    ref = np.full((135, 166, 3), 52, np.uint8)
    cv2.rectangle(ref, (35, 20), (125, 110), (190, 195, 185), -1)
    test = ref.copy()
    cv2.rectangle(test, (61, 48), (104, 79), (28, 35, 219), -1)
    return ref, test


class NeuralProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(9)
        self.model=FaltandoCNNV2().eval()

    def test_real_nn_maps_from_actual_forward_and_gradients(self):
        ref, test = pair()
        before={k:v.clone() for k,v in self.model.state_dict().items()}
        data=generate_explainability_triplet(
            self.model, ref, test, image_size=96, focus_fraction=.70,
        )
        self.assertEqual(data["schema"], "visionx.cnn_v2_neural_evidence.v1")
        self.assertEqual(data["layer"], LAYER_NAME)
        self.assertEqual(len(data["images"]),3)
        self.assertTrue(data["neural"])
        self.assertFalse(data["reconstruction_decoder"])
        self.assertTrue(data["probe_only"])
        self.assertIn(data["target_class"], {"NG","OK"})
        self.assertTrue(0<=data["probe_ng_score_uncalibrated"]<=1)
        for image in data["images"]:
            self.assertEqual(image.shape, test.shape)
            self.assertEqual(image.dtype,np.uint8)
        # O encoder é realmente sensível a alterações nos nove canais.
        self.assertGreater(data["raw_feature_means"][0], 1e-6)
        # Não ajusta pesos nem gera gradientes permanentes nos parâmetros.
        self.assertTrue(all(
            torch.equal(before[k],self.model.state_dict()[k])
            for k in before
        ))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(not any(
            bool(m._forward_hooks) for m in self.model.modules()
        ))

    def test_counterfactual_equal_pair_has_zero_feature_difference(self):
        ref,_ = pair()
        data=generate_explainability_triplet(
            self.model, ref, ref.copy(), image_size=96, focus_fraction=.70,
        )
        self.assertEqual(data["raw_feature_means"][0],0)
        # COLORMAP_JET no zero == azul, não mapa de diferença fictício.
        self.assertTrue(np.all(data["images"][0] == data["images"][0][0,0]))

    def test_rejects_missing_crop_and_non_eval_model(self):
        ref,test=pair()
        with self.assertRaises(ValueError):
            generate_explainability_triplet(
                self.model,None,test,image_size=96,focus_fraction=.7)
        self.model.train()
        with self.assertRaises(ValueError):
            generate_explainability_triplet(
                self.model,ref,test,image_size=96,focus_fraction=.7)

    def test_model_loader_requires_verified_checkpoint_and_reuses_metadata(self):
        ref,test=pair()
        with TemporaryDirectory() as temp:
            path=Path(temp)/"cnn_v2.pt"
            state={
                "schema":MODEL_SCHEMA_V2,"experimental":True,
                "production_approved":False,"lights":list(LIGHTS),
                "image_size":96,"focus_fraction":.7,
                "best_epoch":18,"state_dict":self.model.state_dict(),
            }
            torch.save(state,path)
            digest=sha256(path.read_bytes()).hexdigest()
            clf=FaltandoCNNLive(checkpoint_path=path,expected_sha256=digest)
            probe=FaltandoExplainability(clf)
            images={
                "large_reference":ref,"large":test,
                "small_reference":ref[27:117,34:132].copy(),
                "small":test[27:117,34:132].copy(),
            }
            result=probe.explain_epicenters(images)
            self.assertEqual(set(result),{"major","minor"})
            for item in result.values():
                self.assertEqual(item["checkpoint_sha256"],digest)
                self.assertTrue(item["neural"])
            self.assertIs(clf._model,probe.predictor._model)
            wrong=FaltandoCNNLive(
                checkpoint_path=path,expected_sha256="0"*64
            )
            with self.assertRaisesRegex(ValueError,"hash"):
                FaltandoExplainability(wrong).explain_epicenters(images)

    def test_no_fake_neural_map_without_real_roi(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/"model.pt"
            torch.save({
                "schema":MODEL_SCHEMA_V2,"experimental":True,
                "production_approved":False,"lights":list(LIGHTS),
                "image_size":96,"focus_fraction":.7,
                "state_dict":self.model.state_dict(),
            },path)
            clf=FaltandoCNNLive(path,sha256(path.read_bytes()).hexdigest())
            maps=FaltandoExplainability(clf).explain_epicenters({})
            self.assertIsNone(maps["major"])
            self.assertIsNone(maps["minor"])


if __name__ == "__main__":
    unittest.main()
