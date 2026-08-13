from __future__ import annotations

import unittest

from vllm_pd_router.protocol import (
    build_decode_payload,
    build_prefill_payload,
    extract_kv_transfer_params,
)


class ProtocolTest(unittest.TestCase):
    def test_prefill_payload_forces_one_nonstream_token(self) -> None:
        original = {
            "model": "m",
            "prompt": "hello",
            "max_completion_tokens": 128,
            "stream": True,
            "stream_options": {"include_usage": True},
            "metadata": {"request_id": "r0", "oracle_output_tokens": 128},
        }
        payload = build_prefill_payload(original)
        self.assertEqual(payload["max_tokens"], 1)
        self.assertFalse(payload["stream"])
        self.assertNotIn("max_completion_tokens", payload)
        self.assertNotIn("stream_options", payload)
        self.assertNotIn("metadata", payload)
        self.assertTrue(payload["kv_transfer_params"]["do_remote_decode"])
        self.assertTrue(original["stream"])

    def test_extract_and_attach_transfer_params(self) -> None:
        response = {
            "kv_transfer_params": {
                "remote_engine_id": "producer",
                "remote_port": 5601,
                "remote_block_ids": [1, 2],
            }
        }
        params = extract_kv_transfer_params(response, "http://10.0.0.1:8101")
        self.assertEqual(params["remote_host"], "10.0.0.1")
        decode = build_decode_payload(
            {"prompt": "hello", "max_tokens": 10, "metadata": {"request_id": "r0"}},
            params,
        )
        self.assertEqual(decode["kv_transfer_params"]["remote_engine_id"], "producer")
        self.assertNotIn("metadata", decode)

    def test_missing_transfer_params_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "kv_transfer_params"):
            extract_kv_transfer_params({}, "http://127.0.0.1:8101")


if __name__ == "__main__":
    unittest.main()
