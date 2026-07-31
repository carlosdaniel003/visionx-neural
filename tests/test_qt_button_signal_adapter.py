import unittest

from src.ui.qt_button_signal_adapter import install_qt_button_signal_adapter


class FakeControlPanel:
    def __init__(self):
        self.start_calls = 0
        self.skip_calls = 0
        self.minimized = False
        self.mss_started = False

    def start_monitoring(self):
        self.start_calls += 1
        self.minimized = True
        self.mss_started = True
        return True

    def skip_image(self):
        self.skip_calls += 1
        return True


install_qt_button_signal_adapter(FakeControlPanel)


class QtButtonSignalAdapterTests(unittest.TestCase):
    def test_clicked_false_does_not_reach_start_monitoring(self):
        panel = FakeControlPanel()

        # Reproduz QPushButton.clicked(False).
        result = panel.start_monitoring(False)

        self.assertTrue(result)
        self.assertEqual(panel.start_calls, 1)
        self.assertTrue(panel.minimized)
        self.assertTrue(panel.mss_started)

    def test_clicked_true_does_not_reach_start_monitoring(self):
        panel = FakeControlPanel()

        result = panel.start_monitoring(True)

        self.assertTrue(result)
        self.assertEqual(panel.start_calls, 1)
        self.assertTrue(panel.minimized)
        self.assertTrue(panel.mss_started)

    def test_clicked_boolean_does_not_reach_skip_image(self):
        panel = FakeControlPanel()

        result = panel.skip_image(False)

        self.assertTrue(result)
        self.assertEqual(panel.skip_calls, 1)

    def test_programmatic_call_without_argument_still_works(self):
        panel = FakeControlPanel()

        self.assertTrue(panel.start_monitoring())
        self.assertTrue(panel.skip_image())
        self.assertEqual(panel.start_calls, 1)
        self.assertEqual(panel.skip_calls, 1)


if __name__ == "__main__":
    unittest.main()
