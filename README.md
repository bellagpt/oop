# Assignment 1: Hopfield network

An object oriented implementation of a Hopfield network, including a demo that
recovers learned patterns in a network of 25 neurons (a 5x5 grid).

| File | Contents |
|------|----------|
| `hopfield.py` | The network (`Neuron`, `HopfieldNetwork`) and a `main()` that reads the command line and runs a demo case |
| `demo.py` | The demo case: two 5x5 patterns and damaged versions of them |

## Requirements

- Python 3.10 or newer
- No external dependencies: only the Python standard library is used,
  so there is nothing to install.

## Running the code in a venv

```bash
git clone https://github.com/bellagpt/oop.git
cd oop
python3 -m venv .venv            # Windows: py -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python hopfield.py               # inside the venv, `python` points to Python 3
```

## Command line arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `--alpha` | `0.2` | Learning parameter between 0 and 1 |
| `--theta` | `1.75` | Threshold a neuron's activation must reach |
| `--demo` | `demo` | Demo case to run: name of a `.py` file, without `.py` |

```bash
python hopfield.py --alpha 0.3 --theta 2.0
python hopfield.py --help
```

To run a different demo case, create e.g. `my_demo.py` that defines `GRID_WIDTH`, `PATTERNS`
(the patterns to learn) and `TEST_CASES` (a list of `(damaged input, expected pattern)` pairs),
and run `python hopfield.py --demo my_demo`.

Without a venv, run `python3 hopfield.py` (Linux/macOS) or `py hopfield.py` (Windows).
On Debian/Ubuntu, `python3 -m venv` may require `sudo apt install python3-venv` first.

## Usage

```python
from hopfield import HopfieldNetwork

network = HopfieldNetwork(size=25, alpha=0.2, theta=1.75)
network.train([pattern_a, pattern_b])  # patterns are lists of 25 values that are 0 or 1
network.theta = 2.0                    # alpha and theta are settable properties
recalled = network.recall(partial)     # returns the stable state as a list of 0/1 values
```

## Design

| Class in `hopfield.py` | Responsibility |
|-------|----------------|
| `Neuron` | Tracks its connections (weights) to the other neurons, learns from the presented pattern and computes its own next state. |
| `HopfieldNetwork` | Creates and connects the neurons, presents patterns to them, updates them synchronously and stops once a stable state is reached. Holds the settable properties `alpha` and `theta`. |

Learning rule (applied by every neuron to each of its connections):

- `w_ij := α + (1 − α) · w_ij` if neurons *i* and *j* are both on or both off;
- `w_ij := (1 − α) · w_ij` otherwise; `w_ii = 0` (no self-connections).

Update rule (synchronous): neuron *i* is on in the next state when `Σ_{j ∈ S_n} w_ij ≥ θ`.
Updating stops as soon as a step changes nothing (`S_n = S_{n+1}`); if that does not
happen within `HopfieldNetwork.MAX_STEPS` steps a `RuntimeError` is raised.
