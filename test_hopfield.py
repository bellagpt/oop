"""Unit tests for the Hopfield network. Run with ``python3 -m unittest``."""

import unittest

from hopfield import BAR, SQUARE, HopfieldNetwork, Neuron, from_grid


class NeuronTest(unittest.TestCase):
    """Tests for the Neuron class."""

    def test_learning_rule(self) -> None:
        first, second = Neuron(), Neuron()
        first.connect_to(second)
        first.set_state(True)
        second.set_state(True)
        first.learn(0.5)  # same state: 0.5 + 0.5 * 0 = 0.5
        self.assertAlmostEqual(first.weight_to(second), 0.5)
        second.set_state(False)
        first.learn(0.5)  # different state: 0.5 * 0.5 = 0.25
        self.assertAlmostEqual(first.weight_to(second), 0.25)

    def test_no_self_connection(self) -> None:
        neuron = Neuron()
        self.assertEqual(neuron.weight_to(neuron), 0.0)
        with self.assertRaises(ValueError):
            neuron.connect_to(neuron)


class HopfieldNetworkTest(unittest.TestCase):
    """Tests for the HopfieldNetwork class."""

    def test_weights_are_symmetric(self) -> None:
        network = HopfieldNetwork(size=25)
        network.train(SQUARE, BAR)
        for i in range(25):
            for j in range(25):
                self.assertAlmostEqual(network.weight(i, j), network.weight(j, i))

    def test_parameters_are_settable_and_checked(self) -> None:
        network = HopfieldNetwork(size=4)
        network.alpha, network.theta = 0.7, 3
        self.assertEqual((network.alpha, network.theta), (0.7, 3.0))
        with self.assertRaises(ValueError):
            network.alpha = 1.5

    def test_rejects_invalid_pattern(self) -> None:
        network = HopfieldNetwork(size=4)
        with self.assertRaises(ValueError):
            network.train([1, 0, 1])
        with self.assertRaises(ValueError):
            network.recall([1, 0, 2, 0])

    def test_recalls_learned_patterns(self) -> None:
        network = HopfieldNetwork(size=25, alpha=0.2, theta=1.75)
        network.train(SQUARE, BAR)
        self.assertEqual(network.recall(from_grid("###.. #.... #.... #.... #####")), SQUARE)
        self.assertEqual(network.recall(from_grid("..... ##### ##### ..... .....")), BAR)

    def test_raises_when_not_stable(self) -> None:
        # Two neurons that only activate each other keep swapping states.
        network = HopfieldNetwork(size=2, alpha=1.0, theta=0.5, max_steps=10)
        network.train([1, 1])
        with self.assertRaises(RuntimeError):
            network.recall([1, 0])


if __name__ == "__main__":
    unittest.main()
