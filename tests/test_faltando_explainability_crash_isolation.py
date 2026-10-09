"""Critical regression: fatal native Torch error must not close ODIN Qt process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.core.neural.faltando_explainability_runner import (
    _read_response, _safe_inputs, explain_in_isolated_process,
)


def pair():
    ref = np.full((125, 136, 3), 105, np.uint8)
    test = ref.copy()
    test[25:50, 30:60, 1] = 224
    return {"large_reference":ref, "large":test}


class ProbeCrashIsolationTests(unittest.TestCase):
    def test_rejects_missing_or_malformed_image_before_spawn(self):
        with self.assertRaisesRegex(ValueError,"par completo"):
            _safe_inputs({"large_reference": pair()["large_reference"]})
        with self.assertRaisesRegex(ValueError,"inválido"):
            _safe_inputs({"large_reference":np.zeros((5,5,3),np.uint8),
                          "large":pair()["large"]})

    def test_native_windows_crash_is_error_not_main_process_exit(self):
        with patch("src.core.neural.faltando_explainability_runner.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                returncode=3221225477, stderr=b"access violation in PyTorch"
            )
            with self.assertRaisesRegex(RuntimeError, "3221225477"):
                explain_in_isolated_process(pair())
            self.assertTrue(run.called)
            args, kwargs = run.call_args
            self.assertIn("--child", args[0])
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"],"1")
            self.assertEqual(kwargs["env"]["MKL_NUM_THREADS"],"1")
            self.assertEqual(kwargs["check"], False)

    def test_timeout_is_displayable_error(self):
        with patch("src.core.neural.faltando_explainability_runner.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("cnn", 12)):
            with self.assertRaisesRegex(RuntimeError, "Tempo limite"):
                explain_in_isolated_process(pair(),timeout_seconds=12)

    def test_emergency_disable_prevents_any_subprocess(self):
        with patch.dict(os.environ, {"VISIONX_DISABLE_NEURAL_MAPS":"1"}):
            with patch("src.core.neural.faltando_explainability_runner.subprocess.run") as run:
                with self.assertRaisesRegex(RuntimeError, "desativados"):
                    explain_in_isolated_process(pair())
                run.assert_not_called()

    def test_only_neural_maps_can_be_read(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"result.npz"
            rows={}
            for name in ("major","minor"):
                for index in range(3):
                    rows[f"{name}_{index}"]=np.zeros((32,40,3),dtype=np.uint8)
            np.savez_compressed(output,**rows)
            metadata = {
                name: {
                    "neural":True, "layer":"encoder.4",
                    "target_class":"OK", "dimensions":[40,32],
                    "raw_feature_means":[.01,.02,.03]
                }
                for name in ("major","minor")
            }
            output.with_suffix(".json").write_text(json.dumps(metadata))
            result=_read_response(output)
            self.assertEqual(len(result["major"]["images"]),3)
            self.assertEqual(result["minor"]["layer"],"encoder.4")
            metadata["major"]["neural"]=False
            output.with_suffix(".json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,"não foi derivada"):
                _read_response(output)

    def test_child_failure_leaves_no_native_torch_in_qt_parent(self):
        """Only the disposable --child path is allowed to load torch."""
        path=Path("src/core/neural/faltando_explainability_runner.py")
        script=path.read_text(encoding="utf-8")
        self.assertIn('def _child(',script)
        self.assertIn('    import torch',script)
        self.assertIn("subprocess.run(",script)
        self.assertIn("BoundedSemaphore(1)",script)


class QtProbeResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
        from PyQt6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def test_qt_worker_catches_child_failure_and_emits_error(self):
        from src.ui.widgets.neural_evidence import _NeuralProbeTask
        task=_NeuralProbeTask(17,pair())
        self.assertFalse(task.autoDelete())
        received=[]
        task.signals.done.connect(lambda epoch, rows, error:
                                  received.append((epoch,rows,error)))
        with patch("src.core.neural.faltando_explainability_runner.explain_in_isolated_process",
                   side_effect=RuntimeError("native child stopped")):
            task.run()
        self.assertEqual(received[0][0],17)
        self.assertIsNone(received[0][1])
        self.assertIn("native child stopped",received[0][2])
        self.assertEqual(task.crops,{})

    def test_panel_displays_error_and_does_not_show_old_images(self):
        from src.ui.widgets.neural_evidence import NeuralEvidencePanel
        widget=NeuralEvidencePanel()
        self.addCleanup(widget.deleteLater)
        widget.set_visual_payload(pair())
        with patch("src.ui.widgets.neural_evidence.QThreadPool.globalInstance") as pool:
            widget.update_data({"cnn_v2_active":True},{"verdict":"FALHA FALSA"})
            self.assertTrue(pool.return_value.start.called)
        current=widget._epoch
        widget._on_probe_finished(current,None,"CNN indisponível: native failure")
        self.assertIn("CNN indisponível",widget.footer.text())
        self.assertTrue(all(tile.image._source.isNull() for tile in widget.tiles))
        widget.clear_data()
        widget._on_probe_finished(current,{"major":{"neural":True}},"")
        self.assertTrue(all(tile.image._source.isNull() for tile in widget.tiles))


if __name__ == "__main__":
    unittest.main()
