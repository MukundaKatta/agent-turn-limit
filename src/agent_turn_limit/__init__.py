"""agent-turn-limit: hard turn cap and threshold warnings for agent loops.

Public API:
    TurnCounter(hard_limit, warn_at, on_warn, stop_on_limit, label)
    TurnWarning   — emitted at warning thresholds
    TurnLimitExceeded — raised when hard limit hit
    make_turn_counter(...) -> TurnCounter
"""

from .core import (
    TurnCounter,
    TurnLimitError,
    TurnLimitExceeded,
    TurnWarning,
    make_turn_counter,
)

__all__ = [
    "TurnCounter",
    "TurnWarning",
    "TurnLimitError",
    "TurnLimitExceeded",
    "make_turn_counter",
]
__version__ = "0.1.0"
