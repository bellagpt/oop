"""Unit tests for the Hopfield network. Run with ``python -m unittest``."""

from __future__ import annotations

import random
import unittest

from hopfield import (
    AsynchronousUpdate,
    HopfieldNetwork,
    Neuron,
    NotStableError,
    Pattern,
    SynchronousUpdate,
)


class PatternTest(unittest.TestCase):
    """Tests for the Pattern value object."""

    def test_grid_round_trip(self) -> None:
        pattern = Pattern.from_grid("#.\n.#")
        self.assertEqual(list(pattern), [1, 0, 0, 1])
        self.assertEqual(pattern.to_grid(2), "# .\n. #")
        self.assertEqual(pattern.active_indices, frozenset({0, 3}))

    def test_from_active(self) -> None:
        self.assertEqual(Pattern.from_active(4, [1, 2]), Pattern([0, 1, 1, 0]))
        with self.assertRaises(ValueError):
            Pattern.from_active(4, [4])

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            Pattern([0, 2])
        with self.assertRaises(ValueError):
            Pattern([])

    def test_hamming_distance(self) -> None:
        self.assertEqual(Pattern([1, 0, 1]).hamming_distance(Pattern([0, 0, 0])), 2)


class NeuronTest(unittest.TestCase):
    """Tests for the Neuron class."""

    def setUp(self) -> None:
        self.first, self.second = Neuron(), Neuron()
        self.first.connect_to(self.second)
        self.second.connect_to(self.first)

    def test_initial_weight_is_zero(self) -> None:
        self.assertEqual(self.first.weight_to(self.second), 0.0)
        self.assertEqual(self.first.weight_to(self.first), 0.0)

    def test_cannot_connect_to_itself_or_twice(self) -> None:
        with self.assertRaises(ValueError):
            self.first.connect_to(self.first)
        with self.assertRaises(ValueError):
            self.first.connect_to(self.second)
        with self.assertRaises(TypeError):
            self.first.connect_to("not a neuron")  # type: ignore[arg-type]

    def test_learning_rule(self) -> None:
        alpha = 0.5
        # Same state: w := alpha + (1 - alpha) * w
        self.first.present(True)
        self.second.present(True)
        self.first.learn(alpha)
        self.assertAlmostEqual(self.first.weight_to(self.second), 0.5)
        # Different state: w := (1 - alpha) * w
        self.second.present(False)
        self.first.learn(alpha)
        self.assertAlmostEqual(self.first.weight_to(self.second), 0.25)

    def test_update_reports_change(self) -> None:
        self.first.present(True)
        self.first.learn(1.0)
        self.second.learn(1.0)  # w = 0 for differing states, so second stays off ...
        self.assertFalse(self.second.update(theta=0.5))
        self.assertTrue(self.second.update(theta=0.0))  # ... unless the threshold is 0.


class HopfieldNetworkTest(unittest.TestCase):
    """Tests for the HopfieldNetwork class."""

    SQUARE = Pattern.from_grid("##### #...# #...# #...# #####")
    BAR = Pattern.from_grid("..... ##### ##### ##### .....")

    def test_weights_are_symmetric_and_zero_on_diagonal(self) -> None:
        network = HopfieldNetwork(size=25, alpha=0.3)
        network.train(self.SQUARE, self.BAR)
        for i in range(network.size):
            self.assertEqual(network.weight(i, i), 0.0)
            for j in range(network.size):
                self.assertAlmostEqual(network.weight(i, j), network.weight(j, i))

    def test_parameter_validation(self) -> None:
        network = HopfieldNetwork(size=4)
        network.alpha = 0.7
        network.theta = -2
        self.assertEqual((network.alpha, network.theta), (0.7, -2.0))
        with self.assertRaises(ValueError):
            network.alpha = 1.5
        with self.assertRaises(TypeError):
            network.theta = "high"  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            network.theta = float("nan")
        with self.assertRaises(ValueError):
            HopfieldNetwork(size=1)

    def test_rejects_pattern_of_wrong_size(self) -> None:
        network = HopfieldNetwork(size=4)
        with self.assertRaises(ValueError):
            network.train([1, 0, 1])
        with self.assertRaises(ValueError):
            network.recall([1, 0, 1, 0, 1])

    def test_recalls_learned_patterns(self) -> None:
        partial_square = Pattern.from_grid("###.. #.... #.... #.... #####")
        partial_bar = Pattern.from_grid("..... ##### ##### ..... .....")
        for strategy in (SynchronousUpdate(), AsynchronousUpdate(random.Random(0))):
            with self.subTest(strategy=strategy):
                network = HopfieldNetwork(size=25, alpha=0.2, theta=1.75, update_strategy=strategy)
                network.train(self.SQUARE, self.BAR)
                self.assertEqual(network.recall(partial_square), self.SQUARE)
                self.assertEqual(network.recall(partial_bar), self.BAR)

    def test_recall_does_not_change_weights(self) -> None:
        network = HopfieldNetwork(size=25, alpha=0.2, theta=1.75)
        network.train(self.SQUARE, self.BAR)
        before = [network.weight(i, j) for i in range(25) for j in range(25)]
        network.recall(self.SQUARE)
        self.assertEqual(before, [network.weight(i, j) for i in range(25) for j in range(25)])

    def test_raises_when_not_stable(self) -> None:
        # Two neurons that only activate each other keep swapping states under synchronous updating.
        network = HopfieldNetwork(size=2, alpha=1.0, theta=0.5, max_steps=10)
        network.train([1, 1])  # w_01 = 1
        with self.assertRaises(NotStableError):
            network.recall([1, 0])  # synchronous: [1,0] -> [0,1] -> [1,0] -> ...


if __name__ == "__main__":
    unittest.main()
