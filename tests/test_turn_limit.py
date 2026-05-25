"""Tests for agent-turn-limit."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from agent_turn_limit import (
    TurnCounter,
    TurnLimitExceeded,
    TurnWarning,
    make_turn_counter,
)


# ---------------------------------------------------------------------------
# Basic tick behavior
# ---------------------------------------------------------------------------

def test_initial_turn_is_zero():
    c = TurnCounter(hard_limit=5)
    assert c.turn == 0


def test_tick_increments_turn():
    c = TurnCounter(hard_limit=5)
    c.tick()
    assert c.turn == 1


def test_tick_returns_true_below_limit():
    c = TurnCounter(hard_limit=5)
    assert c.tick() is True


def test_tick_allows_up_to_hard_limit():
    c = TurnCounter(hard_limit=3)
    for _ in range(3):
        assert c.tick() is True
    assert c.turn == 3


def test_tick_raises_at_limit_plus_one():
    c = TurnCounter(hard_limit=3)
    for _ in range(3):
        c.tick()
    with pytest.raises(TurnLimitExceeded):
        c.tick()


def test_tick_error_message_contains_label():
    c = TurnCounter(hard_limit=2, label="my-agent")
    c.tick(); c.tick()
    with pytest.raises(TurnLimitExceeded, match="my-agent"):
        c.tick()


def test_remaining_decreases():
    c = TurnCounter(hard_limit=5)
    assert c.remaining == 5
    c.tick()
    assert c.remaining == 4
    c.tick()
    assert c.remaining == 3


def test_remaining_zero_at_limit():
    c = TurnCounter(hard_limit=3)
    c.tick(); c.tick(); c.tick()
    assert c.remaining == 0


def test_is_exhausted_false_below_limit():
    c = TurnCounter(hard_limit=5)
    c.tick()
    assert c.is_exhausted is False


def test_is_exhausted_true_at_limit():
    c = TurnCounter(hard_limit=2)
    c.tick(); c.tick()
    assert c.is_exhausted is True


# ---------------------------------------------------------------------------
# stop_on_limit=False
# ---------------------------------------------------------------------------

def test_stop_on_limit_false_returns_false():
    c = TurnCounter(hard_limit=2, stop_on_limit=False)
    c.tick(); c.tick()
    result = c.tick()
    assert result is False


def test_stop_on_limit_false_does_not_raise():
    c = TurnCounter(hard_limit=2, stop_on_limit=False)
    for _ in range(5):
        c.tick()  # no exception


# ---------------------------------------------------------------------------
# Warning thresholds — integer
# ---------------------------------------------------------------------------

def test_warn_at_int_triggers():
    warnings = []
    c = TurnCounter(hard_limit=10, warn_at=[5], on_warn=warnings.append)
    for _ in range(5):
        c.tick()
    assert len(warnings) == 1
    assert warnings[0].turn == 5


def test_warn_at_multiple():
    warnings = []
    c = TurnCounter(hard_limit=10, warn_at=[3, 7], on_warn=warnings.append)
    for _ in range(7):
        c.tick()
    assert [w.turn for w in warnings] == [3, 7]


def test_warn_not_triggered_before_threshold():
    warnings = []
    c = TurnCounter(hard_limit=10, warn_at=[5], on_warn=warnings.append)
    for _ in range(4):
        c.tick()
    assert warnings == []


def test_warn_once_per_threshold():
    count = [0]
    c = TurnCounter(hard_limit=20, warn_at=[5], on_warn=lambda _: count.__setitem__(0, count[0]+1))
    for _ in range(10):
        c.tick()
    assert count[0] == 1


# ---------------------------------------------------------------------------
# Warning thresholds — float fraction
# ---------------------------------------------------------------------------

def test_warn_at_float_fraction():
    warnings = []
    c = TurnCounter(hard_limit=10, warn_at=[0.5], on_warn=warnings.append)
    for _ in range(5):
        c.tick()
    assert warnings[0].turn == 5


def test_warn_at_float_80_percent():
    warnings = []
    c = TurnCounter(hard_limit=10, warn_at=[0.8], on_warn=warnings.append)
    for _ in range(8):
        c.tick()
    assert warnings[0].turn == 8


def test_warn_warning_message_contains_turns():
    warnings = []
    c = TurnCounter(hard_limit=10, warn_at=[5], on_warn=warnings.append)
    for _ in range(5):
        c.tick()
    assert "5/10" in warnings[0].message


def test_warn_warning_remaining():
    warnings = []
    c = TurnCounter(hard_limit=10, warn_at=[5], on_warn=warnings.append)
    for _ in range(5):
        c.tick()
    assert warnings[0].remaining == 5


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_clears_turn():
    c = TurnCounter(hard_limit=5)
    c.tick(); c.tick()
    c.reset()
    assert c.turn == 0


def test_reset_clears_warnings():
    warnings = []
    c = TurnCounter(hard_limit=10, warn_at=[3], on_warn=warnings.append)
    for _ in range(3):
        c.tick()
    c.reset()
    warnings.clear()
    for _ in range(3):
        c.tick()
    assert len(warnings) == 1  # warning fires again after reset


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

def test_context_manager_works():
    with TurnCounter(hard_limit=5) as c:
        c.tick()
    assert c.turn == 1


def test_context_manager_raises_inside():
    with pytest.raises(TurnLimitExceeded):
        with TurnCounter(hard_limit=2) as c:
            c.tick(); c.tick(); c.tick()


# ---------------------------------------------------------------------------
# make_turn_counter factory
# ---------------------------------------------------------------------------

def test_make_turn_counter_defaults():
    c = make_turn_counter(10)
    assert c.hard_limit == 10
    assert c.warn_at == [5, 8]  # 0.5*10=5, 0.8*10=8


def test_make_turn_counter_custom_warn():
    c = make_turn_counter(20, warn_at=[5, 15])
    assert c.warn_at == [5, 15]


def test_make_turn_counter_label():
    c = make_turn_counter(5, label="triage")
    assert c.label == "triage"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_hard_limit_zero_raises():
    with pytest.raises(ValueError):
        TurnCounter(hard_limit=0)


def test_hard_limit_negative_raises():
    with pytest.raises(ValueError):
        TurnCounter(hard_limit=-1)
