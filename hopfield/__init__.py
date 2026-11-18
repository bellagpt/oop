"""An object oriented implementation of a Hopfield network."""

from .network import HopfieldNetwork, NotStableError
from .neuron import Neuron
from .pattern import Pattern
from .protocols import HasState, Learner, Presentable, Updatable, UpdateStrategy
from .update_strategies import AsynchronousUpdate, SynchronousUpdate

__all__ = [
    "AsynchronousUpdate",
    "HasState",
    "HopfieldNetwork",
    "Learner",
    "Neuron",
    "NotStableError",
    "Pattern",
    "Presentable",
    "SynchronousUpdate",
    "Updatable",
    "UpdateStrategy",
]
