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
    One neuron in the network. It tracks its connections to other neurons, learns from patterns and updates
    """

    def __init__(self) -> None:
        self._active = False
        self._weights: dict[Neuron, float] = {}  # weights between self and other neurons (w_ij)

    @property
    def is_active(self) -> bool:
        """
        Check if a neuron is on (return True when it's on, return False otherwise)
        """
        return self._active

    def connect_to(self, other: Neuron) -> None:
        """
        Connects this neuron to another neuron (initial weigth = 0)
        :param other: Neuron to connect to (cannot be the neuron itself)
        """
        if other is not self:
            self._weights[other] = 0.0

    def set_state(self, active: bool) -> None:
        """
        Turns this neuron on or off
        :param active: True for on, False for off
        """
        self._active = active

    def learn(self, alpha: float) -> None:
        """
        Updates every connection based on the learning rule 
        (if both neurons in the connection have the same state: w := alpha + (1 - alpha) * w
        otherwise: w := (1 - alpha) * w)
        :param alpha: Learning parameter between 0 and 1
        """
        for other, weight in self._weights.items():
            same_state = other.is_active == self._active
            self._weights[other] = (alpha if same_state else 0.0) + (1 - alpha) * weight

    def update(self, theta: float) -> bool:
        """
        Calculates subsequent states (activation) according to updating formula 
        Returns True when activation is >= theta, otherwise False
        :param theta: Threshold the activation must reach
        """
        activation = sum(weight for other, weight in self._weights.items() if other.is_active)
        return activation >= theta


class HopfieldNetwork:
    """
    The fully connected network of neurons that learns patterns and recalls them from partial input
    """
    Pattern = list[int] 
    MAX_STEPS = 100  # Prevents looping: if no stable state has been reached by 100 updates, it stops with an error.  

    def __init__(self, size: int, alpha: float, theta: float) -> None:
        """
        :param size: Number of neurons
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
        Returns a list of current states off all neurons
        """
        return [int(neuron.is_active) for neuron in self._neurons]

    def train(self, patterns: list[HopfieldNetwork.Pattern]) -> None:
        """
        Trains every neuron by presnting a pattern (for each pattern) 
        :param patterns: List of patterns to learn
        """
        for pattern in patterns:
            self._present(pattern)
            for neuron in self._neurons:
                neuron.learn(self._alpha)

    def recall(self, pattern: HopfieldNetwork.Pattern) -> HopfieldNetwork.Pattern:
        """
        Presents a partial pattern (input) and updates the neurons until a stable state is reached.
        That stable state is then returned
        :param pattern: Pattern to start from
        """
        self._present(pattern)
        for _ in range(self.MAX_STEPS):
            if not self._update():
                return self.state  
        return None # no stable state was reached after MAX_STEPS iterations

    def _update(self) -> bool:
        """
        Updates all neurons at the same time (synchronously),
        Returns True if the network changed, and False otherwise
        """
        new_states = [neuron.next_state(self._theta) for neuron in self._neurons]
        changed = new_states != [neuron.is_active for neuron in self._neurons]
        for i in range(len(self._neurons)):
            self._neurons[i].set_state(new_states[i])
        return changed

    def _connect_neurons(self) -> None:
        """
        Connects every neuron (self) to every other neuron (not self)
        """
        for neuron in self._neurons:
            for other in self._neurons:
                if other is not neuron:
                    neuron.connect_to(other)

    def _present(self, pattern: HopfieldNetwork.Pattern) -> None:
         """
        Sets every neuron to its value in the pattern: 1 is on, 0 is off
        :param pattern: Pattern to present, with one value per neuron
        """
        for i in range(len(self._neurons)):
            self._neurons[i].set_state(pattern[i] == 1)

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
