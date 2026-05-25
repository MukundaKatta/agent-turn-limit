"""Hard turn limit and threshold warnings for agent loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


class TurnLimitError(Exception):
    """Raised when the hard turn limit is exceeded."""


class TurnLimitExceeded(TurnLimitError):
    """Raised when hard_limit is reached and stop_on_limit=True."""


@dataclass
class TurnWarning:
    """Emitted when a warning threshold is crossed."""

    turn: int
    hard_limit: int
    remaining: int
    message: str


@dataclass
class TurnCounter:
    """Track agent loop turns, emit warnings, enforce a hard cap.

    Args:
        hard_limit: maximum number of turns allowed (inclusive).
        warn_at: iterable of turn numbers (or fractions of hard_limit if float)
            at which a warning callback is invoked.
        on_warn: callable invoked with a TurnWarning when a threshold is hit.
        stop_on_limit: if True (default), raise TurnLimitExceeded when the
            hard limit is reached. If False, just return False from tick().
        label: optional label for error messages.

    Usage::

        counter = TurnCounter(hard_limit=20, warn_at=[10, 15])
        while True:
            counter.tick()          # raises TurnLimitExceeded at turn 20
            ...run agent turn...

    Or use as a context manager::

        with TurnCounter(hard_limit=20) as c:
            while True:
                c.tick()
                ...
    """

    hard_limit: int
    warn_at: list[int] = field(default_factory=list)
    on_warn: Callable[[TurnWarning], None] | None = None
    stop_on_limit: bool = True
    label: str = "agent"
    _turn: int = field(default=0, init=False, repr=False)
    _warned: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.hard_limit < 1:
            raise ValueError(f"hard_limit must be >= 1, got {self.hard_limit}")
        # Normalise warn_at: floats become fractions of hard_limit
        resolved: list[int] = []
        for w in self.warn_at:
            if isinstance(w, float):
                resolved.append(max(1, int(w * self.hard_limit)))
            else:
                resolved.append(int(w))
        self.warn_at = sorted(set(resolved))

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def turn(self) -> int:
        """Current turn number (0 before any tick)."""
        return self._turn

    @property
    def remaining(self) -> int:
        """Turns remaining before the hard limit (0 means limit reached)."""
        return max(0, self.hard_limit - self._turn)

    @property
    def is_exhausted(self) -> bool:
        """True if the turn count has reached or exceeded hard_limit."""
        return self._turn >= self.hard_limit

    def tick(self) -> bool:
        """Advance the turn counter by one.

        Returns:
            True if the turn is allowed (below hard_limit).

        Raises:
            TurnLimitExceeded: if stop_on_limit=True and hard_limit is reached.
        """
        self._turn += 1
        self._check_warnings()
        if self._turn > self.hard_limit:
            if self.stop_on_limit:
                raise TurnLimitExceeded(
                    f"{self.label}: hard turn limit of {self.hard_limit} exceeded "
                    f"(turn {self._turn})"
                )
            return False
        return True

    def reset(self) -> None:
        """Reset the turn counter to 0."""
        self._turn = 0
        self._warned.clear()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "TurnCounter":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_warnings(self) -> None:
        for threshold in self.warn_at:
            if self._turn == threshold and threshold not in self._warned:
                self._warned.add(threshold)
                warning = TurnWarning(
                    turn=self._turn,
                    hard_limit=self.hard_limit,
                    remaining=self.remaining,
                    message=(
                        f"{self.label}: turn {self._turn}/{self.hard_limit} "
                        f"({self.remaining} remaining)"
                    ),
                )
                if self.on_warn:
                    self.on_warn(warning)


# ---------------------------------------------------------------------------
# Functional convenience
# ---------------------------------------------------------------------------

def make_turn_counter(
    hard_limit: int,
    *,
    warn_at: list[int | float] | None = None,
    on_warn: Callable[[TurnWarning], None] | None = None,
    stop_on_limit: bool = True,
    label: str = "agent",
) -> TurnCounter:
    """Factory function for TurnCounter.

    Args:
        hard_limit: maximum turns allowed.
        warn_at: turn numbers (int) or fractions of hard_limit (float 0.0-1.0)
            at which to trigger the warning callback. Default: [0.5, 0.8].
        on_warn: callback invoked with a TurnWarning at each threshold.
        stop_on_limit: raise TurnLimitExceeded when limit hit. Default True.
        label: label for error messages.
    """
    if warn_at is None:
        warn_at = [0.5, 0.8]
    return TurnCounter(
        hard_limit=hard_limit,
        warn_at=warn_at,  # type: ignore[arg-type]
        on_warn=on_warn,
        stop_on_limit=stop_on_limit,
        label=label,
    )
