import math

import numpy as np
import numpy.typing as npt
from attrs import Factory, cmp_using, define, field, setters, validators

from sudokutoolbox.util import is_square, valid_square

_1 = np.int_(1)


@define(kw_only=True)
class Constraint:
    N: int = field(
        default=9,
        on_setattr=setters.frozen,
        validator=[validators.instance_of(int), validators.gt(0), valid_square],  # type: ignore
    )  # type: ignore
    _RN: int = field(
        init=False,
        default=Factory(lambda self: int(math.sqrt(self.N)), takes_self=True),
    )
    r: tuple[int, ...] = field(
        init=False,
        repr=False,
        default=Factory(
            lambda self: tuple(i // self.N for i in range(self.N**2)), takes_self=True
        ),
    )
    c: tuple[int, ...] = field(
        init=False,
        repr=False,
        default=Factory(
            lambda self: tuple(i % self.N for i in range(self.N**2)), takes_self=True
        ),
    )
    b: tuple[tuple[int, int], ...] = field(
        init=False,
        repr=False,
        default=Factory(
            lambda self: tuple(
                (self.r[i] // self._RN, self.c[i] // self._RN)
                for i in range(self.N**2)
            ),
            takes_self=True,
        ),
    )
    row: npt.NDArray[np.int_] = field(
        converter=np.array,
        eq=cmp_using(np.array_equal),
        default=Factory(lambda self: np.zeros(self.N, dtype=np.int_), takes_self=True),
    )
    col: npt.NDArray[np.int_] = field(
        converter=np.array,
        eq=cmp_using(np.array_equal),
        default=Factory(lambda self: np.zeros(self.N, dtype=np.int_), takes_self=True),
    )
    box: npt.NDArray[np.int_] = field(
        converter=np.array,
        eq=cmp_using(np.array_equal),
        default=Factory(
            lambda self: np.zeros(tuple(self._RN for _ in range(2)), dtype=np.int_),
            takes_self=True,
        ),
    )

    @classmethod
    def from_puzzle(cls, puzzle: npt.NDArray[np.int_]) -> "Constraint":
        if not is_square(puzzle.size):
            raise ValueError("Puzzle must be a perfect square.")
        root_n = int(math.sqrt(puzzle.size))
        root_rn = int(math.sqrt(root_n))
        _row = np.zeros((root_n), dtype=np.int_)
        _col = np.zeros((root_n), dtype=np.int_)
        _box = np.zeros((root_rn, root_rn), dtype=np.int_)
        for i, j in np.ndindex(puzzle.shape):
            if puzzle[i, j] != 0:
                _row[i] |= _1 << puzzle[i, j]
                _col[j] |= _1 << puzzle[i, j]
                _box[i // root_rn, j // root_rn] |= _1 << puzzle[i, j]
        return cls(N=root_n, row=_row, col=_col, box=_box)

    def is_safe(self, i: int, v: np.int_) -> bool:
        """If the constraint isn't already present, update the constraint and return True."""
        shifted: np.int_ = _1 << v
        if (
            (self.row[self.r[i]] & shifted)
            or (self.col[self.c[i]] & shifted)
            or (self.box[self.b[i]] & shifted)
        ):
            return False
        return True

    def update(self, i: int, v: np.int_) -> None:
        shifted: np.int_ = _1 << v
        self.row[self.r[i]] |= shifted
        self.col[self.c[i]] |= shifted
        self.box[self.b[i]] |= shifted

    def remove(self, i: int, v: np.int_) -> None:
        """Remove the constraint."""
        shifted: np.int_ = _1 << v
        self.row[self.r[i]] &= ~shifted
        self.col[self.c[i]] &= ~shifted
        self.box[self.b[i]] &= ~shifted

    def count_constraints(self, i: int) -> int:
        """Return the number of constraints on the cell."""
        # This double counts a constraint when it is in both the row and column or box.
        # However, it appears to be faster then single counting.
        # Single counting is commented out below.
        # Single counting is the same as sorting by fewest to most candidates.
        return (
            self.row[self.r[i]].bit_count()
            + self.col[self.c[i]].bit_count()
            + self.box[self.b[i]].bit_count()
        )
        # return (self.row[self.r[i]] | self.col[self.c[i]] | self.box[self.b[i]]).bit_count()
