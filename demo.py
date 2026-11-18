"""Demo: a Hopfield network of 25 neurons (a 5x5 grid) recovers learned patterns.

Run with ``python demo.py`` from this directory.
"""

from __future__ import annotations

import random

from hopfield import AsynchronousUpdate, HopfieldNetwork, Pattern, SynchronousUpdate, UpdateStrategy

GRID_WIDTH = 5

SQUARE = Pattern.from_grid("""
    # # # # #
    # . . . #
    # . . . #
    # . . . #
    # # # # #
""")

BAR = Pattern.from_grid("""
    . . . . .
    # # # # #
    # # # # #
    # # # # #
    . . . . .
""")

# Each test case: (description, input presented to the network, pattern that should be recovered).
TEST_CASES: list[tuple[str, Pattern, Pattern]] = [
    ("Square with half of the border missing", Pattern.from_grid("""
        # # # . .
        # . . . .
        # . . . .
        # . . . .
        # # # # #
    """), SQUARE),
    ("Square with three flipped neurons", Pattern.from_grid("""
        # # # # #
        # . # . #
        # . . . .
        # . . . #
        # # # . #
    """), SQUARE),
    ("Bar with its bottom row missing", Pattern.from_grid("""
        . . . . .
        # # # # #
        # # # # #
        . . . . .
        . . . . .
    """), BAR),
    ("Bar with holes and noise", Pattern.from_grid("""
        . # . . .
        # . # # #
        # # . # #
        # # # . .
        . . . . .
    """), BAR),
]


def side_by_side(*blocks: str, gap: str = "    ") -> str:
    """Join multi-line text blocks horizontally."""
    split_blocks = [block.splitlines() for block in blocks]
    return "\n".join(gap.join(lines) for lines in zip(*split_blocks))


def run_demo(strategy_name: str, strategy: UpdateStrategy) -> None:
    """Train a fresh network on both patterns and try to recover them from damaged input."""
    network = HopfieldNetwork(size=GRID_WIDTH ** 2, alpha=0.2, theta=1.75, update_strategy=strategy)
    network.train(SQUARE, BAR)

    print(f"=== {strategy_name} update (alpha={network.alpha}, theta={network.theta}) ===\n")
    for description, damaged, expected in TEST_CASES:
        recalled = network.recall(damaged)
        verdict = "recovered" if recalled == expected else "NOT recovered"
        print(f"{description}: {verdict}")
        print("input        output       expected")
        print(side_by_side(
            damaged.to_grid(GRID_WIDTH),
            recalled.to_grid(GRID_WIDTH),
            expected.to_grid(GRID_WIDTH),
        ))
        print()


def main() -> None:
    """Show recall with both the synchronous and the asynchronous update rule."""
    print("Learned patterns:\n")
    print(side_by_side(SQUARE.to_grid(GRID_WIDTH), BAR.to_grid(GRID_WIDTH)))
    print()
    run_demo("Synchronous", SynchronousUpdate())
    run_demo("Asynchronous", AsynchronousUpdate(random.Random(42)))


if __name__ == "__main__":
    main()
