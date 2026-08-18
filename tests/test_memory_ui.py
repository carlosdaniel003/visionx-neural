import copy
import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

from src.ui.memory_status_model import memory_status_from_detail, memory_summary_text
from src.ui.memory_status_ui import install_memory_status_ui
from src.ui.strict_category_memory_ui import install_strict_category_memory_ui
from src.ui.widgets.knn_spectrum import KNNSpectrumWidget


ROOT = Path(__file__).resolve().parents[1]


def strong_ng_detail():
    return {
        "has_memory": True,
        "memory_available": True,
        "memory_category": "FALTANDO",
        "best_similarity": 0.979,
        "best_ng_similarity": 0.979,
        "best_ok_similarity": 0.912,
        "hypothesis_margin": 0.067,
        "best_match_label": "NG",
        "leading_hypothesis": "NG",
        "memory_score": 1.0,
        "quantity_influence": False,
        "n_neighbors": 5,
        "similarity_breakdown": {
            "dual_scale": True,
            "epicenter_similarity": 0.984,
            "context_similarity": 0.967,
            "similarity": 0.979,
            "scale_weights": {
                "epicenter": 0.70,
                "component_context": 0.30,
            },
        },
        "memory_prototype_stats": {
            "raw_ok_jsons": 23,
            "ok_prototypes": 23,
            "ok_observations": 412,
            "raw_ng_jsons": 7,
            "protected_ng_prototypes": 7,
            "quantity_influence": False,
        },
        "decision_trace": {
            "weights": {"knn": 1.0, "physical": 0.0},
            "memory": {
                "has_memory": True,
                "memory_available": True,
                "best_match_label": "NG",
                "best_similarity": 0.979,
                "best_ng_similarity": 0.979,
                "best_ok_similarity": 0.912,
                "hypothesis_margin": 0.067,
                "memory_score": 1.0,
                "quantity_influence": False,
                "memory_scope": "categoria",
                "role": "HIPÓTESE NG",
            },
        },
    }


def conflict_detail():
    detail = strong_ng_detail()
    detail.update(
        {
            "has_memory": False,
            "memory_conflict": True,
            "operator_review_required": True,
            "best_similarity": 0.975,
            "best_ng_similarity": 0.972,
            "best_ok_similarity": 0.975,
            "hypothesis_margin": 0.003,
            "best_match_label": "OK",
            "leading_hypothesis": "OK",
            "memory_score": 0.5,
            "conflict_margin_threshold": 0.01,
        }
    )
    detail["decision_trace"] = {
        "operator_review_required": True,
        "fusion_rule": "memory_conflict_operator_review",
        "memory": {
            "has_memory": False,
            "memory_available": True,
            "memory_conflict": True,
            "operator_review_required": True,
            "best_match_label": "OK",
            "best_similarity": 0.975,
            "best_ng_similarity": 0.972,
            "best_ok_similarity": 0.975,
            "hypothesis_margin": 0.003,
            "conflict_margin_threshold": 0.01,
            "memory_score": 0.5,
            "quantity_influence": False,
            "memory_scope": "categoria",
            "role": "CONFLITO DE MEMÓRIA",
        },
    }
    return detail


class MemoryStatusModelTests(unittest.TestCase):
    def test_reads_all_three_memory_improvements(self):
        model = memory_status_from_detail(strong_ng_detail())

        self.assertTrue(model["has_memory"])
        self.assertTrue(model["dual_scale"])
        self.assertAlmostEqual(model["epicenter_similarity"], 0.984)
        self.assertAlmostEqual(model["context_similarity"], 0.967)
        self.assertAlmostEqual(model["combined_similarity"], 0.979)
        self.assertAlmostEqual(model["best_ng_similarity"], 0.979)
        self.assertAlmostEqual(model["best_ok_similarity"], 0.912)
        self.assertAlmostEqual(model["hypothesis_margin"], 0.067)
        self.assertEqual(model["leading_hypothesis"], "NG")
        self.assertEqual(model["ok_prototypes"], 23)
        self.assertEqual(model["ok_observations"], 412)
        self.assertEqual(model["protected_ng"], 7)
        self.assertFalse(model["quantity_influence"])

    def test_conflict_remains_visible_even_without_reliable_memory(self):
        model = memory_status_from_detail(conflict_detail())

        self.assertFalse(model["has_memory"])
        self.assertTrue(model["memory_available"])
        self.assertTrue(model["conflict"])
        self.assertTrue(model["review_required"])
        self.assertAlmostEqual(model["best_ng_similarity"], 0.972)
        self.assertAlmostEqual(model["best_ok_similarity"], 0.975)
        self.assertAlmostEqual(model["hypothesis_margin"], 0.003)
        self.assertIn("CONFLITO DE MEMÓRIA", memory_summary_text(conflict_detail()))
        self.assertIn("revisão obrigatória", memory_summary_text(conflict_detail()))

    def test_legacy_epicenter_only_is_presented_without_fake_context(self):
        detail = {
            "has_memory": True,
            "memory_available": True,
            "best_similarity": 0.91,
            "best_match_label": "OK",
            "similarity_breakdown": {
                "policy": "legacy_epicenter_only",
                "dual_scale": False,
                "epicenter_similarity": 0.91,
                "context_similarity": None,
            },
        }
        model = memory_status_from_detail(detail)

        self.assertFalse(model["dual_scale"])
        self.assertAlmostEqual(model["epicenter_similarity"], 0.91)
        self.assertIsNone(model["context_similarity"])
        self.assertEqual(model["context_weight"], 0.0)


class FakeLabel:
    def __init__(self):
        self.text = ""
        self.style = ""

    def setText(self, text):
        self.text = str(text)

    def setStyleSheet(self, style):
        self.style = str(style)


class FakePanel:
    def __init__(self):
        self.lbl_db_info = FakeLabel()
        self.lbl_verdict = FakeLabel()

    def _update_confidence_panel(self, analysis):
        self.lbl_db_info.setText("legacy summary")
        self.lbl_verdict.setText(str(analysis.get("verdict", "-")))


class MemoryStatusUiWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_memory_status_ui(FakePanel)

    def test_conflict_visual_does_not_mutate_analysis(self):
        panel = FakePanel()
        analysis = {
            "verdict": "FALHA FALSA",
            "confidence": 0.5,
            "detail": conflict_detail(),
        }
        original = copy.deepcopy(analysis)

        panel._update_confidence_panel(analysis)

        self.assertEqual(analysis, original)
        self.assertIn("CONFLITO DE MEMÓRIA", panel.lbl_db_info.text)
        self.assertIn("REVISÃO OBRIGATÓRIA", panel.lbl_verdict.text)
        self.assertIn("#ffb454", panel.lbl_verdict.style)

    def test_normal_summary_preserves_original_verdict(self):
        panel = FakePanel()
        analysis = {
            "verdict": "DEFEITO REAL",
            "detail": strong_ng_detail(),
        }
        panel._update_confidence_panel(analysis)

        self.assertEqual(panel.lbl_verdict.text, "DEFEITO REAL")
        self.assertIn("Hipótese NG", panel.lbl_db_info.text)
        self.assertIn("epicentro", panel.lbl_db_info.text)


class MemoryWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        install_strict_category_memory_ui(KNNSpectrumWidget)

    def test_widget_exposes_hierarchical_memory_values(self):
        widget = KNNSpectrumWidget()
        widget.update_data(strong_ng_detail())

        self.assertTrue(widget.is_active)
        self.assertTrue(widget.has_memory)
        self.assertFalse(widget.memory_conflict)
        self.assertEqual(widget.best_label, "NG")
        self.assertAlmostEqual(widget.best_sim, 0.979)
        self.assertEqual(widget.model["ok_observations"], 412)

    def test_conflict_is_not_hidden_by_strict_category_ui(self):
        widget = KNNSpectrumWidget()
        data = conflict_detail()
        data["memory_filter_strict"] = True
        data["memory_candidate_count"] = 30
        widget.update_data(data)

        self.assertFalse(widget.has_memory)
        self.assertTrue(widget.memory_available)
        self.assertTrue(widget.memory_conflict)

    def test_responsive_widget_renders_wide_and_narrow(self):
        widget = KNNSpectrumWidget()
        widget.update_data(conflict_detail())

        self.assertLess(widget.heightForWidth(700), widget.heightForWidth(420))

        widget.resize(700, widget.heightForWidth(700))
        wide = QPixmap(widget.size())
        widget.render(wide)
        self.assertFalse(wide.isNull())

        widget.resize(420, widget.heightForWidth(420))
        narrow = QPixmap(widget.size())
        widget.render(narrow)
        self.assertFalse(narrow.isNull())


class UiOnlyScopeTests(unittest.TestCase):
    def test_status_layer_is_explicitly_visual_only(self):
        source = (ROOT / "src/ui/memory_status_ui.py").read_text(encoding="utf-8")
        self.assertNotIn("src.core", source)
        self.assertNotIn("DatasetManager", source)
        self.assertNotIn("save_sample", source)

    def test_main_installs_visual_layer_before_panel_creation(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("install_memory_status_ui(ControlPanel)"),
            source.index("panel = ControlPanel()"),
        )


if __name__ == "__main__":
    unittest.main()
