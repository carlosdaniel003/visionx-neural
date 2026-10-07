import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.services.production_daily_session_store import (
    ProductionDailySessionStore,
    normalize_metrics,
)


class _MutableDate:
    def __init__(self, value: date):
        self.value = value

    def __call__(self):
        return self.value


class ProductionDailySessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.clock = _MutableDate(date(2026, 10, 7))
        self.store = ProductionDailySessionStore(
            root_dir=self.temp_dir.name,
            today_provider=self.clock,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_reload_same_day(self):
        self.store.save_today(
            {
                "auto_ok": 98,
                "auto_ng": 0,
                "manual_judgments": 2,
                "analysis_count": 100,
                "analysis_time_total": 142.0,
                "analysis_time_count": 100,
            }
        )

        restored = self.store.load_today()
        metrics = restored["metrics"]

        self.assertEqual(restored["date"], "2026-10-07")
        self.assertEqual(metrics["auto_ok"], 98)
        self.assertEqual(metrics["manual_judgments"], 2)
        self.assertEqual(metrics["analysis_count"], 100)
        self.assertAlmostEqual(metrics["accuracy_percent"], 98.0)
        self.assertAlmostEqual(
            metrics["average_analysis_time_seconds"],
            1.42,
        )

    def test_next_day_starts_zero_without_erasing_previous_file(self):
        self.store.save_today(
            {
                "auto_ok": 3,
                "manual_judgments": 1,
                "analysis_count": 4,
                "analysis_time_total": 4.0,
                "analysis_time_count": 4,
            }
        )
        previous_path = self.store.today_path()
        self.assertTrue(previous_path.exists())

        self.clock.value = date(2026, 10, 8)
        new_day = self.store.load_today()

        self.assertEqual(new_day["date"], "2026-10-08")
        self.assertEqual(new_day["metrics"]["auto_ok"], 0)
        self.assertEqual(new_day["metrics"]["manual_judgments"], 0)
        self.assertTrue(previous_path.exists())
        self.assertFalse(self.store.today_path().exists())

    def test_each_day_uses_its_own_file(self):
        self.store.save_today({"auto_ok": 1})
        first = self.store.today_path()

        self.clock.value = date(2026, 10, 8)
        self.store.save_today({"auto_ok": 2})
        second = self.store.today_path()

        self.assertNotEqual(first, second)
        self.assertEqual(first.name, "2026-10-07.json")
        self.assertEqual(second.name, "2026-10-08.json")
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_write_is_valid_json_with_schema_and_derived_metrics(self):
        path = self.store.save_today(
            {
                "auto_ok": 2,
                "manual_judgments": 1,
                "analysis_count": 3,
                "analysis_time_total": 6.0,
                "analysis_time_count": 3,
            }
        )
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload["schema"],
            "visionx.production_daily_session.v1",
        )
        self.assertEqual(payload["date"], "2026-10-07")
        self.assertAlmostEqual(
            payload["metrics"]["accuracy_percent"],
            (2.0 / 3.0) * 100.0,
        )
        self.assertAlmostEqual(
            payload["metrics"]["average_analysis_time_seconds"],
            2.0,
        )

    def test_normalization_rejects_negative_and_invalid_values(self):
        metrics = normalize_metrics(
            {
                "auto_ok": -9,
                "auto_ng": "bad",
                "manual_judgments": 2,
                "analysis_count": -1,
                "analysis_time_total": float("nan"),
                "analysis_time_count": 0,
            }
        )

        self.assertEqual(metrics["auto_ok"], 0)
        self.assertEqual(metrics["auto_ng"], 0)
        self.assertEqual(metrics["manual_judgments"], 2)
        self.assertEqual(metrics["analysis_count"], 0)
        self.assertEqual(metrics["analysis_time_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
