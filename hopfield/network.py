"""The Hopfield network, which coordinates training and updating of its neurons."""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import combinations

from .neuron import Neuron
from .pattern import Pattern
from .protocols import UpdateStrategy
from .update_strategies import SynchronousUpdate


class NotStableError(RuntimeError):
    """Raised when the network does not reach a stable state within the allowed number of steps."""


class HopfieldNetwork:
    """A fully connected network of neurons that learns patterns and recalls them from partial input.

    The network presents patterns to its neurons, initiates their updating
    and stops updating once a stable state has been reached. The neurons
    themselves keep track of their connections, learn and update.

    Example::

        network = HopfieldNetwork(size=25, alpha=0.5, theta=2.0)
        network.train(pattern_a, pattern_b)
        recalled = network.recall(partial_pattern_a)
    """

    def __init__(
        self,
        size: int,
        alpha: float = 0.5,
        theta: float = 1.0,
        update_strategy: UpdateStrategy | None = None,
        max_steps: int = 100,
    ) -> None:
        """Create a network of ``size`` fully connected neurons with all weights 0.

        :param size: the number of neurons.
        :param alpha: the learning parameter, in the interval [0, 1].
        :param theta: the activation threshold used while updating.
        :param update_strategy: how neurons update; synchronous by default.
        :param max_steps: the maximum number of update steps before giving up.
        :raises ValueError: if any of the parameters is invalid.
        """
        if isinstance(size, bool) or not isinstance(size, int) or size < 2:
            raise ValueError(f"A network needs an integer size of at least 2, got {size!r}.")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError(f"max_steps must be a positive integer, got {max_steps!r}.")

        self.alpha = alpha  # Uses the property setters, so values are validated.
        self.theta = theta
        self._update_strategy: UpdateStrategy = (
            update_strategy if update_strategy is not None else SynchronousUpdate()
        )
        self._max_steps = max_steps
        # A tuple, so the collection of neurons cannot be changed after creation.
        self._neurons: tuple[Neuron, ...] = tuple(Neuron() for _ in range(size))
        for first, second in combinations(self._neurons, 2):
            first.connect_to(second)
            second.connect_to(first)

    @property
    def size(self) -> int:
        """The number of neurons in the network."""
        return len(self._neurons)

    @property
    def alpha(self) -> float:
        """The learning parameter α, a number in the interval [0, 1]."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        number = self._as_finite_number(value, "alpha")
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"alpha must lie in the interval [0, 1], got {value!r}.")
        self._alpha = number

    @property
    def theta(self) -> float:
        """The activation threshold θ a neuron must reach to be on."""
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        self._theta = self._as_finite_number(value, "theta")

    @property
    def state(self) -> Pattern:
        """The current state of the network as a (read-only) pattern."""
        return Pattern(neuron.is_active for neuron in self._neurons)

    def weight(self, i: int, j: int) -> float:
        """Return the weight w_ij of the connection between neuron ``i`` and neuron ``j``."""
        return self._neurons[i].weight_to(self._neurons[j])

    def train(self, *patterns: Pattern | Iterable[int]) -> None:
        """Present the given patterns to the network one by one and let every neuron learn from them.

        Each pattern may be a :class:`Pattern` or any iterable of 0/1 values.

        :raises ValueError: if a pattern does not match the size of the network.
        """
        # Validate everything first, so an invalid pattern never leaves the network half-trained.
        validated = [self._validate(pattern) for pattern in patterns]
        for pattern in validated:
            self._present(pattern)
            for neuron in self._neurons:
                neuron.learn(self._alpha)

    def recall(self, partial: Pattern | Iterable[int]) -> Pattern:
        """Present a (partial) pattern and update the network until a stable state is reached.

        The weights do not change during recall.

        :returns: the stable state of the network.
        :raises ValueError: if the pattern does not match the size of the network.
        :raises NotStableError: if no stable state is reached within ``max_steps`` steps.
        """
        self._present(self._validate(partial))
        for _ in range(self._max_steps):
            if not self._update_strategy.step(self._neurons, self._theta):
                return self.state
        raise NotStableError(f"No stable state reached within {self._max_steps} update steps.")

    def _validate(self, pattern: Pattern | Iterable[int]) -> Pattern:
        """Convert ``pattern`` to a :class:`Pattern` and check that it fits this network."""
        converted = pattern if isinstance(pattern, Pattern) else Pattern(pattern)
        if len(converted) != self.size:
            raise ValueError(f"Expected a pattern of {self.size} values, got {len(converted)}.")
        return converted

    def _present(self, pattern: Pattern) -> None:
        """Set the state of every neuron to the corresponding value of ``pattern``."""
        for neuron, value in zip(self._neurons, pattern):
            neuron.present(bool(value))

    @staticmethod
    def _as_finite_number(value: float, name: str) -> float:
        """Return ``value`` as a float, rejecting non-numbers, booleans, NaN and infinity."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}.")
        return float(value)

    def __repr__(self) -> str:
        return (
            f"HopfieldNetwork(size={self.size}, alpha={self._alpha}, theta={self._theta}, "
            f"update_strategy={self._update_strategy!r})"
        )
