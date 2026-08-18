from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MicroburstExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_module(
            "microburst_runner",
            ROOT / "experiments/1p3d_microburst_20260817/run_microburst_experiment.py",
        )
        cls.instrumentation = load_module(
            "microburst_instrumentation",
            ROOT / "experiments/1p3d_microburst_20260817/vllm_preemption_instrumentation.py",
        )

    def test_burst_fits_two_second_window(self) -> None:
        offsets = self.runner.burst_offsets(21)
        self.assertEqual(len(offsets), 63)
        self.assertLess(max(offsets), 2.0)
        self.assertEqual(offsets[:3], [0.0, 0.0005, 0.001])

    def test_background_uses_four_triplets_during_burst(self) -> None:
        offsets = self.runner.background_offsets(20)
        self.assertEqual(len(offsets), 20)
        self.assertEqual(offsets[:3], [0.24, 0.2405, 0.241])
        self.assertGreater(offsets[12], 2.0)

    def test_instrumentation_restores_original(self) -> None:
        fixture = "before\n" + self.instrumentation.NEEDLE + "after\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scheduler.py"
            path.write_text(fixture, encoding="utf-8")
            state = self.instrumentation.instrument(path)
            self.assertIn(self.instrumentation.MARKER, path.read_text(encoding="utf-8"))
            state.restore()
            self.assertEqual(path.read_text(encoding="utf-8"), fixture)


if __name__ == "__main__":
    unittest.main()
