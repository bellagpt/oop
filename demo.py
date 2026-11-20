
"""
File: demo.py
Author: Isabella Ko (S6379338), Kaja Sledzinska (S6350992)
Description: Demo for the Hopfield network. Has two test patterns and partial versions of those patterns. 
"""

GRID_WIDTH = 5  

# The test patterns 

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

PATTERNS = [SQUARE, BAR]


# partial patterns 

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
