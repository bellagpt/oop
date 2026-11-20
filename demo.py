"""
Demo for the Hopfield network: two patterns on a 5x5 grid (25 neurons) and damaged versions of them.

A pattern assigns the value 1 (on) or 0 (off) to every neuron. The lists below are
written as 5 rows of 5 values only to make the shapes easy to see.

Run it with ``python3 hopfield.py``. Another demo case can be used instead with
``python3 hopfield.py --demo other_demo``, as long as it defines the same three names:
GRID_WIDTH, PATTERNS and TEST_CASES.
"""

GRID_WIDTH = 5  # Only used to print the patterns as a grid

SQUARE = [
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    1, 1, 1, 1, 1,
]

BAR = [
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
]

# The patterns the network learns
PATTERNS = [SQUARE, BAR]

# (damaged input, pattern that should come out)
TEST_CASES = [
    ([
        1, 1, 1, 0, 0,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
    ], SQUARE),
    ([
        1, 1, 1, 1, 1,
        1, 0, 1, 0, 1,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 1,
        1, 1, 1, 0, 1,
    ], SQUARE),
    ([
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ], BAR),
    ([
        0, 1, 0, 0, 0,
        1, 0, 1, 1, 1,
        1, 1, 0, 1, 1,
        1, 1, 1, 0, 0,
        0, 0, 0, 0, 0,
    ], BAR),
]
