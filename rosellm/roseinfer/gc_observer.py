from __future__ import annotations

import gc
import time
from typing import Callable, Sequence


def install_gc_observer(
    *,
    warn_ms: float = 0.0,
    log_all: bool = False,
    prefix: str = "",
    nvtx: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> Callable[[], None]:
    """Install a low-overhead GC pause observer.

    - When `warn_ms > 0`, it only logs pauses >= warn_ms.
    - When `log_all` is True, it logs every GC cycle.
    - Returns an `uninstall()` callback.
    """

    warn_ms = float(warn_ms)
    if warn_ms <= 0.0 and not bool(log_all):
        return lambda: None

    if not hasattr(gc, "callbacks"):
        return lambda: None

    start_stacks: dict[int, list[float]] = {}
    prefix_str = (str(prefix).strip() + " ") if str(prefix).strip() else ""
    warn_s = warn_ms / 1000.0

    if log_fn is None:
        log_fn = lambda msg: print(msg, flush=True)

    nvtx_push: Callable[[str], None] | None = None
    nvtx_pop: Callable[[], None] | None = None
    if bool(nvtx):
        try:
            import torch

            if torch.cuda.is_available():
                nvtx_push = torch.cuda.nvtx.range_push
                nvtx_pop = torch.cuda.nvtx.range_pop
        except Exception:
            nvtx_push = None
            nvtx_pop = None

    def _cb(phase: str, info: dict) -> None:  # type: ignore[no-untyped-def]
        gen_raw = info.get("generation", -1)
        try:
            gen = int(gen_raw)
        except Exception:
            gen = -1

        if phase == "start":
            start_stacks.setdefault(gen, []).append(time.perf_counter())
            if nvtx_push is not None:
                nvtx_push(f"{prefix_str}gc gen{gen}")
            return

        if phase != "stop":
            return

        stack = start_stacks.get(gen)
        if not stack:
            return
        start_s = stack.pop()
        dur_s = time.perf_counter() - start_s
        if nvtx_pop is not None:
            try:
                nvtx_pop()
            except Exception:
                pass

        if (not log_all) and dur_s < warn_s:
            return

        collected = info.get("collected")
        uncollectable = info.get("uncollectable")
        counts: Sequence[int] | None
        try:
            counts = tuple(int(x) for x in gc.get_count())
        except Exception:
            counts = None

        msg = (
            f"[gc] {prefix_str}gen={gen} dur_ms={dur_s*1000.0:.3f}"
            f" collected={collected} uncollectable={uncollectable}"
        )
        if counts is not None:
            msg += f" count={counts}"
        log_fn(msg)

    gc.callbacks.append(_cb)  # type: ignore[attr-defined]

    def uninstall() -> None:
        try:
            gc.callbacks.remove(_cb)  # type: ignore[attr-defined]
        except Exception:
            pass

    return uninstall
