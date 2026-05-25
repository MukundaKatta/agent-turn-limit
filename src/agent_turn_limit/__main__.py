"""CLI: demo the turn counter (not a typical CLI tool but useful for docs)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="agent-turn-limit",
        description="Simulate an agent loop with a hard turn limit.",
    )
    parser.add_argument("hard_limit", type=int, help="Hard turn limit")
    parser.add_argument("--turns", type=int, default=None, help="Simulate this many turns (default: hard_limit + 1)")
    args = parser.parse_args(argv)

    from . import TurnLimitExceeded, TurnWarning, make_turn_counter

    warnings: list[str] = []

    def _warn(w: TurnWarning) -> None:
        warnings.append(w.message)
        print(f"  WARNING: {w.message}")

    counter = make_turn_counter(args.hard_limit, on_warn=_warn)
    turns = args.turns if args.turns is not None else args.hard_limit + 1

    for i in range(turns):
        try:
            counter.tick()
            print(f"Turn {counter.turn}: ok ({counter.remaining} remaining)")
        except TurnLimitExceeded as e:
            print(f"Turn {counter.turn}: LIMIT HIT — {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
