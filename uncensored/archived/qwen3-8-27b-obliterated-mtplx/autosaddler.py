"""Minimal stubs so the Kilo proxy can run without AutoSaddler EvoDAG.

The production stack under this folder does not run the optimizer daemon.
The autosaddler experiment lives in ../qwen3-8-27b-obliterated-mtplx-autosaddler/.
"""

from __future__ import annotations

from typing import Any

AS_LOOP_START = "[AS_LOOP]"
AS_LOOP_END = "[/AS_LOOP]"


def load_active() -> dict[str, Any]:
    return {}


def append_live_event(*_args: Any, **_kwargs: Any) -> None:
    return None
