"""A Hopfield network in an object oriented way.

Run ``python3 hopfield.py`` to run the demo in demo.py, or ``python3 hopfield.py other_demo``
to run another demo module (see run_demo for what such a module must define).

A pattern is a list of 0/1 values, one per neuron (1 = on, 0 = off).
"""

from __future__ import annotations

import importlib
import math
import sys
from types import ModuleType


class Neuron:
    """A neuron that tracks its connections to other neurons, learns from patterns and updates."""

    def __init__(self) -> None:
        """Create a neuron that is off and not yet connected to anything."""
        self._active = False
        self._weights: dict[Neuron, float] = {}  # connected neuron -> weight w_ij

    @property
    def is_active(self) -> bool:
        """Whether this neuron is currently on."""
        return self._active

    def connect_to(self, other: Neuron) -> None:
        """Connect this neuron to ``other`` with weight 0. A neuron cannot connect to itself."""
        if other is self:
            raise ValueError("A neuron cannot be connected to itself.")
        self._weights[other] = 0.0

    def weight_to(self, other: Neuron) -> float:
        """Return the weight of the connection to ``other`` (0 for itself, since w_ii = 0)."""
        return 0.0 if other is self else self._weights[other]

    def set_state(self, active: bool) -> None:
        """Turn this neuron on or off, e.g. when the network presents a pattern."""
        self._active = active

    def learn(self, alpha: float) -> None:
        """Update every connection weight according to the current states of both neurons.

        Both on or both off: w := alpha + (1 - alpha) * w, otherwise: w := (1 - alpha) * w.
        """
        for other, weight in self._weights.items():
            same_state = other.is_active == self._active
            self._weights[other] = (alpha if same_state else 0.0) + (1 - alpha) * weight

    def next_state(self, theta: float) -> bool:
        """Return whether the summed weight from active neurons reaches the threshold ``theta``."""
        activation = sum(weight for other, weight in self._weights.items() if other.is_active)
        return activation >= theta


class HopfieldNetwork:
    """A fully connected network that learns patterns and recalls them from partial input.

    The network presents patterns to its neurons, lets them update, and stops
    as soon as a stable state has been reached.
    """

    def __init__(self, size: int, alpha: float = 0.2, theta: float = 1.75, max_steps: int = 100) -> None:
        """Create ``size`` neurons and connect every neuron to every other neuron."""
        self.alpha = alpha  # Goes through the property setters below, so the values are checked.
        self.theta = theta
        self._max_steps = max_steps
        self._neurons = tuple(Neuron() for _ in range(size))
        for neuron in self._neurons:
            for other in self._neurons:
                if other is not neuron:
                    neuron.connect_to(other)

    @property
    def alpha(self) -> float:
        """The learning parameter, a number between 0 and 1."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError(f"alpha must be between 0 and 1, got {value}.")
        self._alpha = float(value)

    @property
    def theta(self) -> float:
        """The threshold a neuron's activation must reach for it to turn on."""
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"theta must be a finite number, got {value}.")
        self._theta = float(value)

    @property
    def state(self) -> list[int]:
        """The current state of all neurons as a new list of 0/1 values."""
        return [int(neuron.is_active) for neuron in self._neurons]

    def weight(self, i: int, j: int) -> float:
        """Return the weight w_ij between neuron ``i`` and neuron ``j``."""
        return self._neurons[i].weight_to(self._neurons[j])

    def train(self, *patterns: list[int]) -> None:
        """Present the patterns one by one and let every neuron learn from each of them."""
        for pattern in patterns:
            self._present(pattern)
            for neuron in self._neurons:
                neuron.learn(self._alpha)

    def recall(self, pattern: list[int]) -> list[int]:
        """Present a (partial) pattern and update synchronously until the state is stable."""
        self._present(pattern)
        for _ in range(self._max_steps):
            # Synchronous: compute all new states first, then apply them all at once.
            new_states = [neuron.next_state(self._theta) for neuron in self._neurons]
            if new_states == [neuron.is_active for neuron in self._neurons]:
                return self.state  # S_n == S_n+1: stable state reached.
            for neuron, new_state in zip(self._neurons, new_states):
                neuron.set_state(new_state)
        raise RuntimeError(f"No stable state reached within {self._max_steps} steps.")

    def _present(self, pattern: list[int]) -> None:
        """Set every neuron to the matching value of ``pattern`` (after checking the pattern)."""
        if len(pattern) != len(self._neurons) or any(value not in (0, 1) for value in pattern):
            raise ValueError(f"A pattern must be a list of {len(self._neurons)} values that are 0 or 1.")
        for neuron, value in zip(self._neurons, pattern):
            neuron.set_state(value == 1)


# --- Running a demo -----------------------------------------------------------------

def to_grid(pattern: list[int], width: int) -> list[str]:
    """Turn a pattern into rows of '#' (on) and '.' (off)."""
    symbols = ["#" if value else "." for value in pattern]
    return [" ".join(symbols[i:i + width]) for i in range(0, len(symbols), width)]


def run_demo(demo: ModuleType) -> None:
    """Train a network on the demo's patterns and try to recover them from its damaged inputs.

    The demo module must define ALPHA, THETA, GRID_WIDTH, PATTERNS and TEST_CASES,
    where TEST_CASES is a list of (description, damaged input, expected pattern).
    """
    network = HopfieldNetwork(size=len(demo.PATTERNS[0]), alpha=demo.ALPHA, theta=demo.THETA)
    network.train(*demo.PATTERNS)
    print(f"Trained on {len(demo.PATTERNS)} patterns (alpha={network.alpha}, theta={network.theta})\n")

    for description, damaged, expected in demo.TEST_CASES:
        recalled = network.recall(damaged)
        print(f"{description}: {'recovered' if recalled == expected else 'NOT recovered'}")
        print("input".ljust(demo.GRID_WIDTH * 2 + 3) + "output".ljust(demo.GRID_WIDTH * 2 + 3) + "expected")
        grids = (to_grid(pattern, demo.GRID_WIDTH) for pattern in (damaged, recalled, expected))
        for row in zip(*grids):
            print("    ".join(row))
        print()


if __name__ == "__main__":
    # Usage: python3 hopfield.py [demo_module]   (default: demo, i.e. demo.py)
    demo_name = sys.argv[1] if len(sys.argv) > 1 else "demo"
    run_demo(importlib.import_module(demo_name))
