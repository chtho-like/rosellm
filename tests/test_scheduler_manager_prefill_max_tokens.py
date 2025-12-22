import queue
from collections import deque

import pytest

from rosellm.roseinfer.server import _PendingRequest, _take_pending_for_prefill


def _req(rid: int, n: int) -> _PendingRequest:
    return _PendingRequest(
        request_id=int(rid),
        prompt="",
        prompt_token_ids=[1] * int(n),
        max_new_tokens=1,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        stop_on_eos=False,
        do_sample=False,
    )


def test_take_pending_for_prefill_respects_max_tokens_fifo() -> None:
    buf: deque[_PendingRequest] = deque()
    q: "queue.Queue[_PendingRequest]" = queue.Queue()
    q.put(_req(0, 2))
    q.put(_req(1, 2))
    q.put(_req(2, 2))

    out = _take_pending_for_prefill(
        buf,
        q,
        max_reqs=8,
        max_tokens=4,
        max_context=1024,
        admission_policy="fifo",
        lookahead=64,
        force_fifo=False,
    )
    assert [r.request_id for r in out] == [0, 1]
    assert list(buf)[0].request_id == 2


def test_take_pending_for_prefill_allows_single_oversize_request() -> None:
    buf: deque[_PendingRequest] = deque()
    q: "queue.Queue[_PendingRequest]" = queue.Queue()
    q.put(_req(0, 100))
    q.put(_req(1, 1))

    out = _take_pending_for_prefill(
        buf,
        q,
        max_reqs=8,
        max_tokens=4,
        max_context=1024,
        admission_policy="fifo",
        lookahead=64,
        force_fifo=False,
    )
    assert [r.request_id for r in out] == [0]
    assert q.get_nowait().request_id == 1


def test_take_pending_for_prefill_uses_max_context_for_cost() -> None:
    buf: deque[_PendingRequest] = deque()
    q: "queue.Queue[_PendingRequest]" = queue.Queue()
    q.put(_req(0, 100))
    q.put(_req(1, 100))

    out = _take_pending_for_prefill(
        buf,
        q,
        max_reqs=8,
        max_tokens=16,
        max_context=8,
        admission_policy="fifo",
        lookahead=64,
        force_fifo=False,
    )
    assert [r.request_id for r in out] == [0, 1]


def test_take_pending_for_prefill_validates_args() -> None:
    buf: deque[_PendingRequest] = deque()
    q: "queue.Queue[_PendingRequest]" = queue.Queue()
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        _take_pending_for_prefill(
            buf,
            q,
            max_reqs=1,
            max_tokens=0,
            max_context=1,
            admission_policy="fifo",
            lookahead=1,
            force_fifo=False,
        )
