"""Workers de preparo não podem inferir, interagir com Qt ou bloquear startup."""
from __future__ import annotations

from threading import Event
import unittest

from src.services.inspection_warmup import start_inspection_warmup


class InspectionWarmupTests(unittest.TestCase):
    def test_startup_launches_independent_background_jobs(self):
        release = Event()
        knn_started, cnn_started = Event(), Event()

        class FakeOrchestrator:
            def __init__(self):
                self.index_created = False
                self.calls = []

            def _get_verified_memory_index(self):
                self.index_created = True
                return self

            def prepare_verified_memory_index(self):
                self.calls.append("KNN")
                knn_started.set()
                self.assert_on_worker()
                release.wait(5)

            def prepare_faltando_cnn(self):
                self.calls.append("CNN")
                cnn_started.set()
                self.assert_on_worker()
                release.wait(5)

            def assert_on_worker(self):
                from threading import current_thread
                assert current_thread().name.startswith("ODINWarmup")
                assert self.index_created

        system = FakeOrchestrator()
        threads = start_inspection_warmup(system)
        try:
            self.assertEqual(len(threads), 2)
            self.assertTrue(all(t.daemon for t in threads))
            self.assertTrue(knn_started.wait(5))
            self.assertTrue(cnn_started.wait(5))
            self.assertCountEqual(system.calls, ["KNN", "CNN"])
        finally:
            release.set()
            for worker in threads:
                worker.join(5)
        self.assertFalse(any(worker.is_alive() for worker in threads))

    def test_absent_optional_warmup_hooks_are_ignored(self):
        class Fake:
            pass
        self.assertEqual(start_inspection_warmup(Fake()), ())


if __name__ == "__main__":
    unittest.main()
