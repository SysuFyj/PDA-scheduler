from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vllm_pd_router.config import load_config


class ConfigTest(unittest.TestCase):
    def test_load_minimal_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                """
model: test-model
prefill:
  - id: P0
    url: http://127.0.0.1:8101
decode:
  - id: D0
    url: http://127.0.0.1:8201
router:
  strategy: least_active
""",
                encoding="utf-8",
            )
            config = load_config(path)
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.prefill[0].worker_id, "P0")
        self.assertEqual(config.decode[0].worker_id, "D0")
        self.assertEqual(config.router.strategy, "least_active")

    def test_unknown_strategy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                """
model: test-model
prefill: [{id: P0, url: http://127.0.0.1:8101}]
decode: [{id: D0, url: http://127.0.0.1:8201}]
router: {strategy: unknown}
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsupported strategy"):
                load_config(path)

    def test_least_active_strategy_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                """
model: test-model
prefill: [{id: P0, url: http://127.0.0.1:8101}]
decode: [{id: D0, url: http://127.0.0.1:8201}]
router: {strategy: least_active}
""",
                encoding="utf-8",
            )
            self.assertEqual(load_config(path).router.strategy, "least_active")

    def test_public_strategies_are_accepted(self) -> None:
        for strategy in ("token_balance", "default_fcfs"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.yaml"
                path.write_text(
                    f"""
model: test-model
prefill: [{{id: P0, url: http://127.0.0.1:8101}}]
decode: [{{id: D0, url: http://127.0.0.1:8201}}]
router: {{strategy: {strategy}}}
""",
                    encoding="utf-8",
                )
                self.assertEqual(load_config(path).router.strategy, strategy)


if __name__ == "__main__":
    unittest.main()
