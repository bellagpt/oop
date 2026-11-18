"""Protocols that describe what the parts of a Hopfield network expect from each other.

The protocols are deliberately small (interface segregation): a class that
only needs to update neurons does not have to know how neurons learn, and
vice versa. Classes depend on these protocols instead of on concrete
classes (dependency inversion).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class HasState(Protocol):
    """Something that is either on (active) or off."""

    @property
    def is_active(self) -> bool:
        """Whether this unit is currently on."""
        ...


class Presentable(HasState, Protocol):
    """A unit whose state can be set from the outside, e.g. by presenting a pattern."""

    def present(self, active: bool) -> None:
        """Set the state of the unit to ``active``."""
        ...


class Learner(Presentable, Protocol):
    """A unit that adapts its connections to the currently presented pattern."""

    def learn(self, alpha: float) -> None:
        """Adapt the connection weights using learning rate ``alpha``."""
        ...


class Updatable(Presentable, Protocol):
    """A unit that can determine its next state from the states of its connections."""

    def next_state(self, theta: float) -> bool:
        """Return the state the unit would take given threshold ``theta``, without changing it."""
        ...

    def update(self, theta: float) -> bool:
        """Move the unit to its next state and return whether its state changed."""
        ...


class UpdateStrategy(Protocol):
    """A rule that determines in what order neurons update their state."""

    def step(self, neurons: Sequence[Updatable], theta: float) -> bool:
        """Perform one update step on ``neurons`` and return whether any state changed.

        A step in which no state changes means that a stable state has been reached.
        """
        ...
