"""Demo data for the Hopfield network: two 5x5 patterns (25 neurons) and damaged versions of them.

Run it with ``python3 hopfield.py`` (this demo is the default) or ``python3 hopfield.py demo``.

Any other demo module can be used instead, as long as it defines the same names:
``ALPHA``, ``THETA``, ``GRID_WIDTH``, ``PATTERNS`` and ``TEST_CASES``.
"""

ALPHA = 0.2
THETA = 1.75
GRID_WIDTH = 5


def from_grid(grid: str) -> list[int]:
    """Turn a text grid of '#' (on) and '.' (off) into a pattern, ignoring whitespace."""
    return [1 if symbol == "#" else 0 for symbol in grid if symbol in "#."]


SQUARE = from_grid("""
    # # # # #
    # . . . #
    # . . . #
    # . . . #
    # # # # #
""")

BAR = from_grid("""
    . . . . .
    # # # # #
    # # # # #
    # # # # #
    . . . . .
""")

# The patterns the network learns.
PATTERNS = [SQUARE, BAR]

# (description, damaged input, pattern that should come out)
TEST_CASES = [
    ("Square with half of the border missing", from_grid("""
        # # # . .
        # . . . .
        # . . . .
        # . . . .
        # # # # #
    """), SQUARE),
    ("Square with three flipped neurons", from_grid("""
        # # # # #
        # . # . #
        # . . . .
        # . . . #
        # # # . #
    """), SQUARE),
    ("Bar with its bottom row missing", from_grid("""
        . . . . .
        # # # # #
        # # # # #
        . . . . .
        . . . . .
    """), BAR),
    ("Bar with holes and noise", from_grid("""
        . # . . .
        # . # # #
        # # . # #
        # # # . .
        . . . . .
    """), BAR),
]
