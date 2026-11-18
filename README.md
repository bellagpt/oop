# Assignment 1: Hopfield network

An object oriented implementation of a Hopfield network, including a demo that
recovers learned patterns in a network of 25 neurons (a 5x5 grid).

All code is in a single file, `hopfield.py`; the unit tests are in `test_hopfield.py`.

## Requirements

- Python 3.10 or newer
- No external dependencies: only the Python standard library is used
  (`requirements.txt` is therefore empty).

## Running the code in a venv

```bash
git clone https://github.com/bellagpt/oop.git
cd oop
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # installs nothing, kept for completeness
python hopfield.py
```

Run the unit tests with:

```bash
python -m unittest -v
```

## Usage

```python
from hopfield import HopfieldNetwork, Pattern, AsynchronousUpdate

square = Pattern.from_grid("""
    # # # # #
    # . . . #
    # . . . #
    # . . . #
    # # # # #
""")

network = HopfieldNetwork(size=25, alpha=0.2, theta=1.75)
network.train(square)               # several patterns: network.train(a, b, c)
network.alpha = 0.3                 # alpha and theta are settable properties
recalled = network.recall(partial)  # returns the stable state as a Pattern

# Asynchronous updating instead of the default synchronous rule:
network = HopfieldNetwork(size=25, update_strategy=AsynchronousUpdate())
```

## Design

`hopfield.py` contains the following classes:

| Class | Responsibility |
|-------|----------------|
| `Neuron` | Tracks its connections (weights), learns from the presented pattern, updates its own state. |
| `HopfieldNetwork` | Presents patterns to its neurons, starts updating and stops once a stable state is reached. Holds the settable properties `alpha` and `theta`. |
| `SynchronousUpdate`, `AsynchronousUpdate` | Decide the order in which neurons update (all at once, or one at a time in random order). |
| `Pattern` | Immutable 0/1 pattern; conversion from/to a text grid. |
| `HasState`, `Presentable`, `Learner`, `Updatable`, `UpdateStrategy` | Small protocols describing what the parts expect from each other. |

Learning rule (applied by every neuron to each of its connections):

- `w_ij := α + (1 − α) · w_ij` if neurons *i* and *j* are both on or both off;
- `w_ij := (1 − α) · w_ij` otherwise; `w_ii = 0` (no self-connections).

Update rule: neuron *i* is on in the next state when `Σ_{j ∈ S_n} w_ij ≥ θ`.
Updating stops as soon as a step changes nothing (`S_n = S_{n+1}`); if that does
not happen within `max_steps` steps a `NotStableError` is raised.

OOP choices:

- **Encapsulation** – state and weights are private; the network exposes the
  state only as an immutable `Pattern`, and its neurons as a tuple that is never
  handed out.
- **Open/closed & dependency inversion** – the network depends on the
  `UpdateStrategy` protocol, so a new update rule can be added without changing
  `HopfieldNetwork`.
- **Interface segregation** – update strategies only need the `Updatable`
  protocol, not the learning part of a neuron.
- **Defensive input handling** – `alpha`, `theta`, sizes and patterns are
  validated; a batch of training patterns is validated before any learning starts.
