"""A single neuron of a Hopfield network."""

from __future__ import annotations


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
