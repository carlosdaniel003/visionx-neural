import sys
import types
import unittest


class FakeSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in list(self.callbacks):
            callback(*args)


class FakeLayout:
    def __init__(self):
        self.items = []

    def addStretch(self):
        self.items.append("stretch")

    def addWidget(self, widget):
        self.items.append(widget)

    def addLayout(self, layout):
        self.items.append(layout)


class FakeButton:
    def __init__(self, text):
        self.text = text
        self.visible = True
        self.clicked = FakeSignal()

    def setToolTip(self, _):
        pass

    def setStyleSheet(self, _):
        pass

    def setVisible(self, visible):
        self.visible = visible


class FakeMessageBox:
    class StandardButton:
        Yes = 1
        No = 2

    answer = StandardButton.Yes
    information_calls = []
    warning_calls = []

    @classmethod
    def question(cls, *_args):
        return cls.answer

    @classmethod
    def information(cls, *args):
        cls.information_calls.append(args)

    @classmethod
    def warning(cls, *args):
        cls.warning_calls.append(args)


qtwidgets = types.ModuleType("PyQt6.QtWidgets")
qtwidgets.QHBoxLayout = FakeLayout
qtwidgets.QMessageBox = FakeMessageBox
qtwidgets.QPushButton = FakeButton
pyqt6 = types.ModuleType("PyQt6")
pyqt6.QtWidgets = qtwidgets
sys.modules.setdefault("PyQt6", pyqt6)
sys.modules.setdefault("PyQt6.QtWidgets", qtwidgets)

from src.ui import test_mode_dataset_controls as controls


class FakeCombo:
    def __init__(self):
        self.value = "Modo Teste"
        self.currentTextChanged = FakeSignal()

    def currentText(self):
        return self.value

    def setCurrentText(self, value):
        self.value = value
        self.currentTextChanged.emit(value)


class FakeActionWidget:
    def __init__(self):
        self._layout = FakeLayout()

    def layout(self):
        return self._layout


class FakeLabel:
    def __init__(self):
        self.text = ""

    def setText(self, value):
        self.text = value


class FakeOrchestrator:
    def __init__(self):
        self.reload_count = 0

    def reload_memory(self):
        self.reload_count += 1


class FakeWindow:
    def __init__(self):
        self.combo_mode = FakeCombo()
        self.action_widget = FakeActionWidget()
        self.orchestrator = FakeOrchestrator()
        self.lbl_db_info = FakeLabel()
        self.ui_builder = types.SimpleNamespace(lbl_status_history=FakeLabel())
        self.skip_count = 0
        self.brain_status = None

    def skip_image(self):
        self.skip_count += 1

    def update_brain_status(self, message, active):
        self.brain_status = (message, active)


class TestModeDatasetControlsTests(unittest.TestCase):
    def setUp(self):
        FakeMessageBox.answer = FakeMessageBox.StandardButton.Yes
        FakeMessageBox.information_calls = []
        FakeMessageBox.warning_calls = []

    def test_button_is_visible_only_in_test_mode(self):
        window = FakeWindow()
        controls.install_test_mode_dataset_controls(window)

        self.assertTrue(window.btn_clear_dataset.visible)

        window.combo_mode.setCurrentText("Modo Produção")
        self.assertFalse(window.btn_clear_dataset.visible)

        window.combo_mode.setCurrentText("Modo Sombra")
        self.assertFalse(window.btn_clear_dataset.visible)

        window.combo_mode.setCurrentText("Modo Teste")
        self.assertTrue(window.btn_clear_dataset.visible)

    def test_confirmed_clear_reloads_memory_and_resets_panel(self):
        window = FakeWindow()
        original_clear = controls.clear_local_dataset
        controls.clear_local_dataset = lambda: {
            "success": True,
            "deleted_files": 4,
            "deleted_directories": 2,
            "errors": [],
        }
        try:
            controls.install_test_mode_dataset_controls(window)
            window.btn_clear_dataset.clicked.emit()
        finally:
            controls.clear_local_dataset = original_clear

        self.assertEqual(window.orchestrator.reload_count, 1)
        self.assertEqual(window.skip_count, 1)
        self.assertEqual(window.lbl_db_info.text, "Dataset local vazio.")
        self.assertIn("Memória KNN zerada", window.brain_status[0])
        self.assertEqual(len(FakeMessageBox.information_calls), 1)


if __name__ == "__main__":
    unittest.main()
