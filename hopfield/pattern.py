"""Immutable representation of a binary activation pattern."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Final

ON: Final[int] = 1
OFF: Final[int] = 0


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
