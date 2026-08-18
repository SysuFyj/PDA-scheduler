from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


MARKER = "PD_REQUEST_PREEMPT request_id=%s computed_tokens=%d preemption_count=%d"
NEEDLE = """        self.kv_cache_manager.free(request)\n        self.encoder_cache_manager.free(request)\n"""
REPLACEMENT = """        logger.info(\n            \"PD_REQUEST_PREEMPT request_id=%s computed_tokens=%d preemption_count=%d\",\n            request.request_id,\n            request.num_computed_tokens,\n            request.num_preemptions + 1,\n        )\n        self.kv_cache_manager.free(request)\n        self.encoder_cache_manager.free(request)\n"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class InstrumentedFile:
    path: Path
    original: str
    original_sha256: str
    instrumented_sha256: str

    def restore(self) -> None:
        self.path.write_text(self.original, encoding="utf-8")
        restored = self.path.read_text(encoding="utf-8")
        if sha256_text(restored) != self.original_sha256:
            raise RuntimeError(f"Failed to restore {self.path}")


def instrument(path: Path) -> InstrumentedFile:
    original = path.read_text(encoding="utf-8")
    if MARKER in original:
        raise RuntimeError(f"Instrumentation already present in {path}")
    if original.count(NEEDLE) != 1:
        raise RuntimeError(f"Expected one instrumentation point in {path}")
    modified = original.replace(NEEDLE, REPLACEMENT)
    path.write_text(modified, encoding="utf-8")
    return InstrumentedFile(
        path=path,
        original=original,
        original_sha256=sha256_text(original),
        instrumented_sha256=sha256_text(modified),
    )
