"""An object oriented implementation of a Hopfield network, including a demo.

Run ``python hopfield.py`` to see a network of 25 neurons (a 5x5 grid)
recover learned patterns from damaged input.

Contents:

- ``Pattern``: an immutable 0/1 pattern.
- Protocols: small interfaces describing what the parts expect from each other.
- ``Neuron``: tracks its connections, learns from patterns and updates its state.
- ``SynchronousUpdate`` / ``AsynchronousUpdate``: the order in which neurons update.
- ``HopfieldNetwork``: presents patterns, starts updating and stops once stable.
- The demo.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Iterator, Sequence
from itertools import combinations
from typing import Final, Protocol

ON: Final[int] = 1
OFF: Final[int] = 0


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------


class Pattern:
    """An immutable pattern that assigns the value 1 (on) or 0 (off) to every neuron.

    A pattern can equivalently be seen as the set of neurons that are on,
    which is available through :attr:`active_indices`.

    Patterns are immutable, so they can be shared freely between objects
    without the risk of one object changing data another object relies on.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Iterable[int | bool]) -> None:
        """Create a pattern from an iterable of 0/1 (or False/True) values.

        :raises ValueError: if a value is not 0 or 1, or the pattern is empty.
        """
        converted: list[int] = []
        for position, value in enumerate(values):
            # `value in (0, 1)` also accepts booleans, which is intended.
            if value not in (OFF, ON):
                raise ValueError(f"Value at position {position} must be 0 or 1, got {value!r}.")
            converted.append(ON if value else OFF)
        if not converted:
            raise ValueError("A pattern must contain at least one value.")
        self._values: tuple[int, ...] = tuple(converted)

    @classmethod
    def from_active(cls, size: int, active_indices: Iterable[int]) -> Pattern:
        """Create a pattern of ``size`` neurons in which only ``active_indices`` are on.

        :raises ValueError: if an index lies outside ``range(size)``.
        """
        active = set(active_indices)
        invalid = {index for index in active if not 0 <= index < size}
        if invalid:
            raise ValueError(f"Indices {sorted(invalid)} are outside the range 0..{size - 1}.")
        return cls(ON if index in active else OFF for index in range(size))

    @classmethod
    def from_grid(cls, grid: str, on_symbol: str = "#", off_symbol: str = ".") -> Pattern:
        """Create a pattern from a textual grid, reading row by row.

        Whitespace (including line breaks) is ignored, so a 5x5 grid can be
        written as five readable lines of five symbols each.

        :raises ValueError: if the grid contains an unknown symbol.
        """
        values: list[int] = []
        for symbol in grid:
            if symbol.isspace():
                continue
            if symbol == on_symbol:
                values.append(ON)
            elif symbol == off_symbol:
                values.append(OFF)
            else:
                raise ValueError(f"Unknown grid symbol {symbol!r}.")
        return cls(values)

    @property
    def active_indices(self) -> frozenset[int]:
        """The set of neuron indices that are on in this pattern."""
        return frozenset(index for index, value in enumerate(self._values) if value == ON)

    def hamming_distance(self, other: Pattern) -> int:
        """Return the number of positions in which this pattern differs from ``other``.

        :raises ValueError: if the patterns differ in length.
        """
        if len(self) != len(other):
            raise ValueError("Patterns must have the same length to be compared.")
        return sum(mine != theirs for mine, theirs in zip(self, other))

    def to_grid(self, width: int, on_symbol: str = "#", off_symbol: str = ".") -> str:
        """Render the pattern as a textual grid of rows of ``width`` symbols.

        :raises ValueError: if the pattern length is not a multiple of ``width``.
        """
        if width <= 0 or len(self) % width != 0:
            raise ValueError(f"Pattern of length {len(self)} cannot be split into rows of {width}.")
        symbols = [on_symbol if value == ON else off_symbol for value in self._values]
        rows = (" ".join(symbols[start:start + width]) for start in range(0, len(self), width))
        return "\n".join(rows)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[int]:
        return iter(self._values)

    def __getitem__(self, index: int) -> int:
        return self._values[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pattern):
            return NotImplemented
        return self._values == other._values

    def __hash__(self) -> int:
        return hash(self._values)

    def __repr__(self) -> str:
        return f"Pattern({list(self._values)})"


# ---------------------------------------------------------------------------
# Protocols
#
# Deliberately small (interface segregation): something that only updates
# neurons does not need to know how neurons learn, and vice versa. Classes
# depend on these protocols rather than on concrete classes (dependency inversion).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Neuron
# ---------------------------------------------------------------------------


class Neuron:
    """A neuron that tracks its weighted connections, learns from patterns and updates its state.

    Every connection is stored on both neurons involved, so each neuron is
    responsible for its own weights. Because both neurons apply the same
    learning rule to the same pair of states, the weights stay symmetric
    (``w_ij == w_ji``) without the neurons having to modify each other.
    """

    def __init__(self) -> None:
        """Create a neuron that is off and has no connections."""
        self._active: bool = False
        self._weights: dict[Neuron, float] = {}

    @property
    def is_active(self) -> bool:
        """Whether this neuron is currently on."""
        return self._active

    @property
    def connection_count(self) -> int:
        """The number of neurons this neuron is connected to."""
        return len(self._weights)

    def connect_to(self, other: Neuron) -> None:
        """Create a connection with weight 0 from this neuron to ``other``.

        :raises TypeError: if ``other`` is not a :class:`Neuron`.
        :raises ValueError: if ``other`` is this neuron or is already connected.
        """
        if not isinstance(other, Neuron):
            raise TypeError(f"A neuron can only connect to another Neuron, not {type(other).__name__}.")
        if other is self:
            raise ValueError("A neuron cannot be connected to itself.")
        if other in self._weights:
            raise ValueError("These neurons are already connected.")
        self._weights[other] = 0.0

    def weight_to(self, other: Neuron) -> float:
        """Return the weight of the connection to ``other`` (0 for this neuron itself).

        :raises KeyError: if ``other`` is not connected to this neuron.
        """
        if other is self:
            return 0.0  # w_ii = 0: neurons are never connected to themselves.
        if other not in self._weights:
            raise KeyError("The given neuron is not connected to this neuron.")
        return self._weights[other]

    def present(self, active: bool) -> None:
        """Set the state of this neuron, e.g. as part of a presented pattern."""
        self._active = bool(active)

    def learn(self, alpha: float) -> None:
        """Adapt all connection weights to the current states, using learning rate ``alpha``.

        Connections between neurons in the same state (both on or both off)
        are strengthened towards 1, all other connections decay towards 0.
        """
        for other in self._weights:
            decayed = (1 - alpha) * self._weights[other]
            same_state = other.is_active == self._active
            self._weights[other] = alpha + decayed if same_state else decayed

    def activation(self) -> float:
        """Return the summed weight of the connections to currently active neurons."""
        return sum(weight for other, weight in self._weights.items() if other.is_active)

    def next_state(self, theta: float) -> bool:
        """Return whether this neuron would be on given threshold ``theta``, without changing it."""
        return self.activation() >= theta

    def update(self, theta: float) -> bool:
        """Move this neuron to its next state and return whether its state changed."""
        new_state = self.next_state(theta)
        changed = new_state != self._active
        self._active = new_state
        return changed

    def __repr__(self) -> str:
        state = "on" if self._active else "off"
        return f"Neuron({state}, connections={self.connection_count})"


# ---------------------------------------------------------------------------
# Update strategies
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Hopfield network
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Demo: a network of 25 neurons (a 5x5 grid) recovers learned patterns
# ---------------------------------------------------------------------------

GRID_WIDTH: Final[int] = 5

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
