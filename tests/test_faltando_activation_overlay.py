"""Regressões para visualizações CNN sobre o TESTE com GAB em miniatura.

Nenhuma máscara de diferença é calculada a partir dos pixels do teste.
A intensidade sobreposta é EXCLUSIVAMENTE saída da CNN/Grad-CAM.
"""
from __future__ import annotations

import unittest
import cv2
import numpy as np

from src.core.neural.faltando_activation_overlay import (
    compose_neural_overlay, MAX_EDGE, KINDS,
)


def sample():
    h, w = 90, 160
    test = np.full((h,w,3), (48,59,78), dtype=np.uint8)
    ref = np.full((h,w,3), (17,165,221), dtype=np.uint8)
    cv2.rectangle(test,(55,12),(144,69),(186,199,202),-1)
    return test, ref


class NeuralOverlayEvidenceTests(unittest.TestCase):
    def test_test_original_is_visible_where_cnn_has_zero_response(self):
        test,ref=sample()
        activation=np.zeros(test.shape[:2],np.float32)
        for kind in KINDS:
            with self.subTest(kind=kind):
                output=compose_neural_overlay(test,ref,activation,kind=kind)
                h,w=output.shape[:2]
                scaled=cv2.resize(test,(w,h),interpolation=cv2.INTER_LINEAR)
                # Outside top-left thumbnail, zero activity preserves
                # actual test pixels: no fake heatmap/solid blue bands.
                self.assertTrue(np.array_equal(
                    output[int(h*.80), int(w*.85)],
                    scaled[int(h*.80), int(w*.85)],
                ))
                self.assertTrue(np.array_equal(
                    output[int(h*.62), int(w*.45)],
                    scaled[int(h*.62), int(w*.45)],
                ))
                self.assertEqual(output.dtype,np.uint8)
                self.assertLessEqual(max(h,w),MAX_EDGE)

    def test_high_cnn_response_highlights_but_does_not_replace_test(self):
        test,ref=sample()
        energy=np.zeros(test.shape[:2],np.float32)
        energy[36:70,75:120]=1.0
        for kind in KINDS:
            with self.subTest(kind=kind):
                output=compose_neural_overlay(test,ref,energy,kind=kind)
                h,w=output.shape[:2]
                base=cv2.resize(test,(w,h),interpolation=cv2.INTER_LINEAR)
                strong=(int(h*.60),int(w*.55))
                calm=(int(h*.80),int(w*.85))
                self.assertFalse(np.array_equal(output[strong],base[strong]))
                self.assertTrue(np.array_equal(output[calm],base[calm]))
                # Original structure is blended in, never replaced by
                # a pure CNN colormap. Compare distinct backgrounds.
                changed=test.copy()
                changed[36:70,75:120]=(29,100,155)
                changed_result=compose_neural_overlay(changed,ref,energy,kind=kind)
                self.assertFalse(np.array_equal(output[strong],changed_result[strong]))

    def test_reference_thumbnail_is_real_and_yellow_bordered(self):
        test,ref=sample()
        heat=np.zeros(test.shape[:2],np.float32)
        result=compose_neural_overlay(test,ref,heat,kind="latent")
        self.assertEqual(tuple(result[7,7]), (0,215,255))
        # Reference inset is distinctive from dark TEST background.
        self.assertFalse(np.array_equal(result[15,52],test[15,52]))
        # Content below inset still matches original TEST.
        w,h=result.shape[1],result.shape[0]
        self.assertTrue(np.array_equal(result[h-8,w-8],test[-8,-8]))

    def test_partial_response_and_gradcam_zero_are_not_scaled_to_hot(self):
        test,ref=sample()
        low=np.full(test.shape[:2], .01, np.float32)
        result=compose_neural_overlay(test,ref,low,kind="gradcam")
        h,w=result.shape[:2]
        base=cv2.resize(test,(w,h),interpolation=cv2.INTER_LINEAR)
        self.assertTrue(np.array_equal(result[h-5,w-5],base[h-5,w-5]))
        with self.assertRaisesRegex(ValueError,"Tipo"):
            compose_neural_overlay(test,ref,low,kind="fake")

    def test_small_epicenter_preserves_horizontal_aspect_ratio(self):
        test=np.full((25,130,3),113,np.uint8)
        ref=np.full((25,130,3),171,np.uint8)
        signal=np.ones((25,130),np.float32)
        result=compose_neural_overlay(test,ref,signal,kind="activation")
        self.assertGreater(result.shape[1],result.shape[0])
        self.assertLessEqual(max(result.shape[:2]),MAX_EDGE)
        self.assertEqual(result.dtype,np.uint8)

    def test_invalid_map_never_falls_back_to_fake_pixel_filter(self):
        test,ref=sample()
        with self.assertRaisesRegex(ValueError,"incompatíveis"):
            compose_neural_overlay(test,ref,np.zeros((2,2),np.float32),kind="latent")
        bad=np.zeros(test.shape[:2],np.float32)
        bad[0,0]=np.nan
        with self.assertRaisesRegex(ValueError,"incompatíveis"):
            compose_neural_overlay(test,ref,bad,kind="gradcam")


if __name__=="__main__":
    unittest.main()
