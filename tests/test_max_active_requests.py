import pytest

from rosellm.roseinfer.server import _cap_prefill_max_reqs


def test_cap_prefill_max_reqs_unlimited() -> None:
    assert (
        _cap_prefill_max_reqs(
            8,
            max_active_requests=None,
            active_unfinished=123,
        )
        == 8
    )


def test_cap_prefill_max_reqs_clamps_to_slots() -> None:
    assert (
        _cap_prefill_max_reqs(
            8,
            max_active_requests=16,
            active_unfinished=15,
        )
        == 1
    )


def test_cap_prefill_max_reqs_returns_zero_when_full() -> None:
    assert (
        _cap_prefill_max_reqs(
            8,
            max_active_requests=16,
            active_unfinished=16,
        )
        == 0
    )


def test_cap_prefill_max_reqs_validates_args() -> None:
    with pytest.raises(ValueError, match="max_reqs must be positive"):
        _cap_prefill_max_reqs(0, max_active_requests=None, active_unfinished=0)
    with pytest.raises(ValueError, match="active_unfinished must be non-negative"):
        _cap_prefill_max_reqs(1, max_active_requests=None, active_unfinished=-1)
    with pytest.raises(ValueError, match="max_active_requests must be positive"):
        _cap_prefill_max_reqs(1, max_active_requests=0, active_unfinished=0)
