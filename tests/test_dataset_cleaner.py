import tempfile
import unittest
from pathlib import Path

from src.config.settings import settings
from src.services.dataset_cleaner import clear_local_dataset


class DatasetCleanerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.original_dataset_dir = settings.DATASET_DIR
        self.original_anomaly_dir = settings.ANOMALY_DIR
        self.original_normal_dir = settings.NORMAL_DIR

        settings.DATASET_DIR = self.root / "dataset"
        settings.ANOMALY_DIR = settings.DATASET_DIR / "anomalia"
        settings.NORMAL_DIR = settings.DATASET_DIR / "nao_anomalia"
        settings.ANOMALY_DIR.mkdir(parents=True)
        settings.NORMAL_DIR.mkdir(parents=True)

    def tearDown(self):
        settings.DATASET_DIR = self.original_dataset_dir
        settings.ANOMALY_DIR = self.original_anomaly_dir
        settings.NORMAL_DIR = self.original_normal_dir
        self.temp_dir.cleanup()

    def test_clear_removes_files_and_subfolders_but_keeps_roots(self):
        nested_anomaly = settings.ANOMALY_DIR / "Shifted"
        nested_normal = settings.NORMAL_DIR / "Bridge"
        nested_anomaly.mkdir()
        nested_normal.mkdir()

        (nested_anomaly / "sample_ng.png").write_bytes(b"ng")
        (nested_anomaly / "sample_ng.json").write_text("{}", encoding="utf-8")
        (nested_normal / "sample_ok.json").write_text("{}", encoding="utf-8")

        result = clear_local_dataset()

        self.assertTrue(result["success"])
        self.assertEqual(result["deleted_files"], 3)
        self.assertEqual(result["deleted_directories"], 2)
        self.assertTrue(settings.ANOMALY_DIR.is_dir())
        self.assertTrue(settings.NORMAL_DIR.is_dir())
        self.assertEqual(list(settings.ANOMALY_DIR.iterdir()), [])
        self.assertEqual(list(settings.NORMAL_DIR.iterdir()), [])

    def test_clear_refuses_target_outside_dataset_root(self):
        outside = self.root / "outside"
        outside.mkdir()
        protected_file = outside / "do_not_delete.json"
        protected_file.write_text("{}", encoding="utf-8")
        settings.ANOMALY_DIR = outside

        result = clear_local_dataset()

        self.assertFalse(result["success"])
        self.assertTrue(protected_file.exists())
        self.assertTrue(any("recusado por segurança" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
