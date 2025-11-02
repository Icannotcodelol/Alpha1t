"""Simple benchmarking script."""

from __future__ import annotations

import argparse
import time

from ..engine.game import TysiacGame


def main(argv: list[str] | None = None) -> None:
    """Run benchmark games and report throughput."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    args = parser.parse_args(argv)
    game = TysiacGame.new()
    start = time.perf_counter()
    for _ in range(args.games):
        game.run_hand()
    duration = time.perf_counter() - start
    print(f"Played {args.games} games in {duration:.2f}s")


if __name__ == "__main__":
    main()
