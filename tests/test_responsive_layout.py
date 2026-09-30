import unittest

from src.ui.responsive_layout import profile_for_width


class ResponsiveLayoutProfileTests(unittest.TestCase):
    def test_compact_profile_for_small_notebook(self):
        profile = profile_for_width(1024)
        self.assertEqual(profile.name, "compact")
        self.assertTrue(profile.splitter_vertical)
        self.assertEqual(profile.info_columns, 2)
        self.assertEqual(profile.action_columns, 2)

    def test_common_notebook_widths_remain_compact(self):
        for width in (1280, 1366, 1440, 1499):
            with self.subTest(width=width):
                profile = profile_for_width(width)
                self.assertEqual(profile.name, "compact")
                self.assertTrue(profile.splitter_vertical)
                self.assertEqual(profile.footer_columns, 1)

    def test_standard_profile_starts_after_notebook_range(self):
        profile = profile_for_width(1500)
        self.assertEqual(profile.name, "standard")
        self.assertFalse(profile.splitter_vertical)
        self.assertEqual(profile.info_columns, 3)
        self.assertEqual(profile.footer_columns, 2)

    def test_wide_profile_for_desktop_monitor(self):
        profile = profile_for_width(1920)
        self.assertEqual(profile.name, "wide")
        self.assertEqual(profile.info_columns, 5)
        self.assertEqual(profile.footer_columns, 3)
        self.assertGreater(profile.debugger_min_width, 450)

    def test_breakpoints_are_stable(self):
        self.assertEqual(profile_for_width(1499).name, "compact")
        self.assertEqual(profile_for_width(1500).name, "standard")
        self.assertEqual(profile_for_width(1799).name, "standard")
        self.assertEqual(profile_for_width(1800).name, "wide")


if __name__ == "__main__":
    unittest.main()
