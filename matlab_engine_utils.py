"""Shared, bounded startup retry for transient MATLAB license shutdowns."""

from __future__ import annotations

import time


def start_matlab_with_retry(engine_module, attempts: int = 6, delay_seconds: float = 10.0):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return engine_module.start_matlab()
        except engine_module.EngineError:
            if attempt == attempts:
                raise
            wait_seconds = delay_seconds * attempt
            print(
                f"MATLAB Engine startup failed ({attempt}/{attempts}); "
                f"retrying in {wait_seconds:.0f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")
