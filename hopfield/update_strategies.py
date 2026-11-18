"""Strategies that determine the order in which neurons update their state."""

from __future__ import annotations

import random
from collections.abc import Sequence

from .protocols import Updatable


class SynchronousUpdate:
    """Every neuron updates at the same time, based on the previous state of the network.

    This implements ``S_{n+1} = { i | sum_{j in S_n} w_ij >= theta }``.
    """

    def step(self, neurons: Sequence[Updatable], theta: float) -> bool:
        """Update all neurons simultaneously and return whether any state changed."""
        # First compute all new states, so that no neuron sees a partially updated network.
        new_states = [neuron.next_state(theta) for neuron in neurons]
        changed = False
        for neuron, new_state in zip(neurons, new_states):
            changed |= neuron.is_active != new_state
            neuron.present(new_state)
        return changed

    def __repr__(self) -> str:
        return "SynchronousUpdate()"


class AsynchronousUpdate:
    """Neurons update one at a time, in a random order, each seeing the latest states.

    One step is a sweep in which every neuron updates exactly once. A sweep
    without any change means that no single neuron would change anymore,
    i.e. a stable state has been reached.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        """Create the strategy; pass a seeded ``rng`` for reproducible results."""
        self._rng = rng if rng is not None else random.Random()

    def step(self, neurons: Sequence[Updatable], theta: float) -> bool:
        """Update every neuron once, one by one in random order; return whether any state changed."""
        # Shuffle a copy: the caller's sequence is passed by reference and must not be modified.
        order = list(neurons)
        self._rng.shuffle(order)
        changed = False
        for neuron in order:
            changed |= neuron.update(theta)
        return changed

    def __repr__(self) -> str:
        return "AsynchronousUpdate()"
