import queue
from collections import deque

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


def test_take_pending_for_prefill_pack_skips_oversize_head() -> None:
    buf: deque[_PendingRequest] = deque()
    q: "queue.Queue[_PendingRequest]" = queue.Queue()
    q.put(_req(0, 100))
    q.put(_req(1, 2))
    q.put(_req(2, 2))

    out = _take_pending_for_prefill(
        buf,
        q,
        max_reqs=8,
        max_tokens=4,
        max_context=1024,
        admission_policy="pack",
        lookahead=16,
        force_fifo=False,
    )
    assert [r.request_id for r in out] == [1, 2]
    assert list(buf)[0].request_id == 0
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    else:
        raise AssertionError("expected queue to be empty")


def test_take_pending_for_prefill_pack_progresses_when_all_oversize() -> None:
    buf: deque[_PendingRequest] = deque()
    q: "queue.Queue[_PendingRequest]" = queue.Queue()
    q.put(_req(0, 100))
    q.put(_req(1, 100))

    out = _take_pending_for_prefill(
        buf,
        q,
        max_reqs=8,
        max_tokens=4,
        max_context=1024,
        admission_policy="pack",
        lookahead=16,
        force_fifo=False,
    )
    assert [r.request_id for r in out] == [0]
    assert list(buf)[0].request_id == 1
