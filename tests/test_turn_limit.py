"""Tests for agent-turn-limit.

These tests use the Python standard-library ``unittest`` framework only, so
they run with::

    python3 -m unittest discover -s tests

without any third-party dependencies.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from agent_turn_limit import (  # noqa: E402
    TurnCounter,
    TurnLimitError,
    TurnLimitExceeded,
    TurnWarning,
    make_turn_counter,
)


class TickBehaviorTests(unittest.TestCase):
    """Core tick / counter behavior."""

    def test_initial_turn_is_zero(self):
        c = TurnCounter(hard_limit=5)
        self.assertEqual(c.turn, 0)

    def test_tick_increments_turn(self):
        c = TurnCounter(hard_limit=5)
        c.tick()
        self.assertEqual(c.turn, 1)

    def test_tick_returns_true_below_limit(self):
        c = TurnCounter(hard_limit=5)
        self.assertIs(c.tick(), True)

    def test_tick_allows_up_to_hard_limit(self):
        c = TurnCounter(hard_limit=3)
        for _ in range(3):
            self.assertIs(c.tick(), True)
        self.assertEqual(c.turn, 3)

    def test_tick_raises_at_limit_plus_one(self):
        c = TurnCounter(hard_limit=3)
        for _ in range(3):
            c.tick()
        with self.assertRaises(TurnLimitExceeded):
            c.tick()

    def test_tick_error_message_contains_label(self):
        c = TurnCounter(hard_limit=2, label="my-agent")
        c.tick()
        c.tick()
        with self.assertRaisesRegex(TurnLimitExceeded, "my-agent"):
            c.tick()

    def test_tick_error_message_contains_turn_number(self):
        c = TurnCounter(hard_limit=2)
        c.tick()
        c.tick()
        with self.assertRaisesRegex(TurnLimitExceeded, r"turn 3"):
            c.tick()

    def test_remaining_decreases(self):
        c = TurnCounter(hard_limit=5)
        self.assertEqual(c.remaining, 5)
        c.tick()
        self.assertEqual(c.remaining, 4)
        c.tick()
        self.assertEqual(c.remaining, 3)

    def test_remaining_zero_at_limit(self):
        c = TurnCounter(hard_limit=3)
        c.tick()
        c.tick()
        c.tick()
        self.assertEqual(c.remaining, 0)

    def test_remaining_never_negative_past_limit(self):
        c = TurnCounter(hard_limit=2, stop_on_limit=False)
        for _ in range(5):
            c.tick()
        self.assertEqual(c.remaining, 0)

    def test_is_exhausted_false_below_limit(self):
        c = TurnCounter(hard_limit=5)
        c.tick()
        self.assertIs(c.is_exhausted, False)

    def test_is_exhausted_true_at_limit(self):
        c = TurnCounter(hard_limit=2)
        c.tick()
        c.tick()
        self.assertIs(c.is_exhausted, True)


class StopOnLimitFalseTests(unittest.TestCase):
    """Behavior when ``stop_on_limit=False``."""

    def test_returns_false_past_limit(self):
        c = TurnCounter(hard_limit=2, stop_on_limit=False)
        c.tick()
        c.tick()
        self.assertIs(c.tick(), False)

    def test_does_not_raise(self):
        c = TurnCounter(hard_limit=2, stop_on_limit=False)
        for _ in range(5):
            c.tick()  # must not raise

    def test_returns_true_up_to_limit(self):
        c = TurnCounter(hard_limit=2, stop_on_limit=False)
        self.assertIs(c.tick(), True)
        self.assertIs(c.tick(), True)


class IntegerWarningTests(unittest.TestCase):
    """Integer warning thresholds."""

    def test_warn_at_int_triggers(self):
        warnings = []
        c = TurnCounter(hard_limit=10, warn_at=[5], on_warn=warnings.append)
        for _ in range(5):
            c.tick()
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].turn, 5)

    def test_warn_at_multiple(self):
        warnings = []
        c = TurnCounter(hard_limit=10, warn_at=[3, 7], on_warn=warnings.append)
        for _ in range(7):
            c.tick()
        self.assertEqual([w.turn for w in warnings], [3, 7])

    def test_warn_not_triggered_before_threshold(self):
        warnings = []
        c = TurnCounter(hard_limit=10, warn_at=[5], on_warn=warnings.append)
        for _ in range(4):
            c.tick()
        self.assertEqual(warnings, [])

    def test_warn_once_per_threshold(self):
        count = [0]
        c = TurnCounter(
            hard_limit=20,
            warn_at=[5],
            on_warn=lambda _w: count.__setitem__(0, count[0] + 1),
        )
        for _ in range(10):
            c.tick()
        self.assertEqual(count[0], 1)

    def test_no_callback_does_not_error(self):
        # warn_at set but no on_warn callback: must not raise.
        c = TurnCounter(hard_limit=10, warn_at=[5])
        for _ in range(6):
            c.tick()

    def test_duplicate_thresholds_deduplicated(self):
        c = TurnCounter(hard_limit=10, warn_at=[5, 5, 5])
        self.assertEqual(c.warn_at, [5])


class FloatWarningTests(unittest.TestCase):
    """Fractional warning thresholds."""

    def test_warn_at_float_fraction(self):
        warnings = []
        c = TurnCounter(hard_limit=10, warn_at=[0.5], on_warn=warnings.append)
        for _ in range(5):
            c.tick()
        self.assertEqual(warnings[0].turn, 5)

    def test_warn_at_float_80_percent(self):
        warnings = []
        c = TurnCounter(hard_limit=10, warn_at=[0.8], on_warn=warnings.append)
        for _ in range(8):
            c.tick()
        self.assertEqual(warnings[0].turn, 8)

    def test_warn_message_contains_turns(self):
        warnings = []
        c = TurnCounter(hard_limit=10, warn_at=[5], on_warn=warnings.append)
        for _ in range(5):
            c.tick()
        self.assertIn("5/10", warnings[0].message)

    def test_warn_warning_remaining(self):
        warnings = []
        c = TurnCounter(hard_limit=10, warn_at=[5], on_warn=warnings.append)
        for _ in range(5):
            c.tick()
        self.assertEqual(warnings[0].remaining, 5)

    def test_warn_warning_is_turnwarning_instance(self):
        warnings = []
        c = TurnCounter(hard_limit=10, warn_at=[5], on_warn=warnings.append)
        for _ in range(5):
            c.tick()
        self.assertIsInstance(warnings[0], TurnWarning)

    def test_small_fraction_floors_to_at_least_one(self):
        # 0.1 * 5 == 0.5 -> int() == 0, but floor is clamped to >= 1.
        c = TurnCounter(hard_limit=5, warn_at=[0.1])
        self.assertEqual(c.warn_at, [1])


class ResetTests(unittest.TestCase):
    """Reset behavior."""

    def test_reset_clears_turn(self):
        c = TurnCounter(hard_limit=5)
        c.tick()
        c.tick()
        c.reset()
        self.assertEqual(c.turn, 0)

    def test_reset_clears_warnings(self):
        warnings = []
        c = TurnCounter(hard_limit=10, warn_at=[3], on_warn=warnings.append)
        for _ in range(3):
            c.tick()
        c.reset()
        warnings.clear()
        for _ in range(3):
            c.tick()
        self.assertEqual(len(warnings), 1)  # warning fires again after reset

    def test_reset_allows_ticking_again_after_limit(self):
        c = TurnCounter(hard_limit=2)
        c.tick()
        c.tick()
        with self.assertRaises(TurnLimitExceeded):
            c.tick()
        c.reset()
        self.assertIs(c.tick(), True)
        self.assertEqual(c.turn, 1)


class ContextManagerTests(unittest.TestCase):
    """Context-manager usage."""

    def test_context_manager_returns_counter(self):
        with TurnCounter(hard_limit=5) as c:
            self.assertIsInstance(c, TurnCounter)
            c.tick()
        self.assertEqual(c.turn, 1)

    def test_context_manager_raises_inside(self):
        with self.assertRaises(TurnLimitExceeded):
            with TurnCounter(hard_limit=2) as c:
                c.tick()
                c.tick()
                c.tick()


class FactoryTests(unittest.TestCase):
    """``make_turn_counter`` factory."""

    def test_defaults(self):
        c = make_turn_counter(10)
        self.assertEqual(c.hard_limit, 10)
        self.assertEqual(c.warn_at, [5, 8])  # 0.5*10=5, 0.8*10=8

    def test_custom_warn(self):
        c = make_turn_counter(20, warn_at=[5, 15])
        self.assertEqual(c.warn_at, [5, 15])

    def test_label(self):
        c = make_turn_counter(5, label="triage")
        self.assertEqual(c.label, "triage")

    def test_returns_turncounter(self):
        self.assertIsInstance(make_turn_counter(5), TurnCounter)

    def test_stop_on_limit_passthrough(self):
        c = make_turn_counter(2, stop_on_limit=False)
        c.tick()
        c.tick()
        self.assertIs(c.tick(), False)


class ValidationTests(unittest.TestCase):
    """Constructor validation."""

    def test_hard_limit_zero_raises(self):
        with self.assertRaises(ValueError):
            TurnCounter(hard_limit=0)

    def test_hard_limit_negative_raises(self):
        with self.assertRaises(ValueError):
            TurnCounter(hard_limit=-1)

    def test_hard_limit_one_is_valid(self):
        c = TurnCounter(hard_limit=1)
        self.assertIs(c.tick(), True)
        with self.assertRaises(TurnLimitExceeded):
            c.tick()


class ExceptionHierarchyTests(unittest.TestCase):
    """Exception class relationships."""

    def test_exceeded_is_subclass_of_limit_error(self):
        self.assertTrue(issubclass(TurnLimitExceeded, TurnLimitError))

    def test_limit_error_is_exception(self):
        self.assertTrue(issubclass(TurnLimitError, Exception))

    def test_can_catch_as_limit_error(self):
        c = TurnCounter(hard_limit=1)
        c.tick()
        with self.assertRaises(TurnLimitError):
            c.tick()


if __name__ == "__main__":
    unittest.main()
