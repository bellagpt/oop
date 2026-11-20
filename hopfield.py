"""
File: hopfield.py
Author: Isabella Ko (S6379338), Kaja Sledzinska (S6350992)
Description: Hopfield network implemented in an object oriented way.
With settable parameters --alpha and --theta.
"""

from __future__ import annotations
import argparse
import importlib


class Neuron:
    """
    A neuron that tracks its connections to other neurons, learns from patterns and updates
    """

    def __init__(self) -> None:
        self._active = False
        self._weights: dict[Neuron, float] = {}  # Connected neuron -> weight w_ij

    @property
    def is_active(self) -> bool:
        """
        Whether this neuron is currently on
        """
        return self._active

    def connect_to(self, other: Neuron) -> None:
        """
        Connects this neuron to another neuron with weight 0
        :param other: Neuron to connect to, which cannot be this neuron itself
        """
        if other is self:
            raise ValueError("A neuron cannot be connected to itself")
        self._weights[other] = 0.0

    def weight_to(self, other: Neuron) -> float:
        """
        :param other: Neuron at the other end of the connection
        :return: Weight of the connection, which is 0 for the neuron itself (w_ii = 0)
        """
        return 0.0 if other is self else self._weights[other]

    def set_state(self, active: bool) -> None:
        """
        Turns this neuron on or off
        :param active: True for on, False for off
        """
        self._active = active

    def learn(self, alpha: float) -> None:
        """
        Updates every connection weight to the current states of both neurons:
        w := alpha + (1 - alpha) * w if both are on or both are off, otherwise w := (1 - alpha) * w
        :param alpha: Learning parameter between 0 and 1
        """
        for other, weight in self._weights.items():
            same_state = other.is_active == self._active
            self._weights[other] = (alpha if same_state else 0.0) + (1 - alpha) * weight

    def next_state(self, theta: float) -> bool:
        """
        :param theta: Threshold the activation must reach
        :return: Whether this neuron should be on, based on the summed weight from active neurons
        """
        activation = sum(weight for other, weight in self._weights.items() if other.is_active)
        return activation >= theta


class HopfieldNetwork:
    """
    A fully connected network of neurons that learns patterns and recalls them from partial input
    """
    Pattern = list[int]  # One value per neuron: 1 (on) or 0 (off)
    MAX_STEPS = 100

    def __init__(self, size: int, alpha: float, theta: float) -> None:
        """
        :param size: Number of neurons
        :param alpha: Learning parameter between 0 and 1
        :param theta: Threshold a neuron's activation must reach to turn on
        """
        self.alpha = alpha
        self.theta = theta
        self._neurons = tuple(Neuron() for _ in range(size))
        self._connect_neurons()

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not 0 <= value <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self._alpha = value

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        self._theta = value

    @property
    def state(self) -> HopfieldNetwork.Pattern:
        """
        The current state of the network as a new list, so it cannot change the neurons
        """
        return [int(neuron.is_active) for neuron in self._neurons]

    def train(self, patterns: list[HopfieldNetwork.Pattern]) -> None:
        """
        Presents the patterns one by one and lets every neuron learn from each of them
        :param patterns: Patterns to learn
        """
        for pattern in patterns:
            self._present(pattern)
            for neuron in self._neurons:
                neuron.learn(self._alpha)

    def recall(self, pattern: HopfieldNetwork.Pattern) -> HopfieldNetwork.Pattern:
        """
        Presents a (partial) pattern and updates the neurons until a stable state is reached
        :param pattern: Pattern to start from
        :return: The stable state of the network
        """
        self._present(pattern)
        for _ in range(self.MAX_STEPS):
            if not self._update():
                return self.state  # Nothing changed, so S_n == S_n+1
        raise RuntimeError(f"No stable state reached within {self.MAX_STEPS} steps")

    def _update(self) -> bool:
        """
        Updates all neurons at the same time (synchronously)
        :return: Whether any neuron changed its state
        """
        # Compute all new states first, so no neuron sees a partially updated network
        new_states = [neuron.next_state(self._theta) for neuron in self._neurons]
        changed = new_states != [neuron.is_active for neuron in self._neurons]
        for neuron, new_state in zip(self._neurons, new_states):
            neuron.set_state(new_state)
        return changed

    def _connect_neurons(self) -> None:
        for neuron in self._neurons:
            for other in self._neurons:
                if other is not neuron:
                    neuron.connect_to(other)

    def _present(self, pattern: HopfieldNetwork.Pattern) -> None:
        if len(pattern) != len(self):
            raise ValueError(f"A pattern must have {len(self)} values")
        for neuron, value in zip(self._neurons, pattern):
            neuron.set_state(value == 1)

    def __len__(self) -> int:
        return len(self._neurons)


def main() -> None:
    """
    Parses the command line arguments, trains a network on the demo case's patterns
    and tries to recover them from the demo case's damaged inputs
    """
    parser = argparse.ArgumentParser(
        description="Train a Hopfield network on a demo case and recall its patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--alpha", type=float, default=0.2, help="Learning parameter between 0 and 1")
    parser.add_argument("--theta", type=float, default=1.75, help="Threshold a neuron's activation must reach")
    parser.add_argument("--demo", default="demo", help="Demo case to run: name of a .py file, without .py")
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    try:
        demo = importlib.import_module(args.demo)
    except ModuleNotFoundError:
        parser.error(f"demo case '{args.demo}.py' not found")

    network = HopfieldNetwork(len(demo.PATTERNS[0]), args.alpha, args.theta)
    network.train(demo.PATTERNS)

    width = demo.GRID_WIDTH
    for number, (damaged, expected) in enumerate(demo.TEST_CASES, start=1):
        try:
            recalled = network.recall(damaged)
        except RuntimeError as error:
            print(f"Test {number}: {error}\n")
            continue
        result = "pattern recovered successfully" if recalled == expected else "pattern NOT recovered"
        print(f"Test {number}: {result}")
        print("input".ljust(width * 2 + 3) + "output")
        # Print the input and output next to each other, one row of the grid at a time
        for start in range(0, len(damaged), width):
            print("    ".join(" ".join(map(str, pattern[start:start + width]))
                             for pattern in (damaged, recalled)))
        print()


if __name__ == "__main__":
    main()
