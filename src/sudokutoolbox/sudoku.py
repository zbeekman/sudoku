import math
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
from attrs import Factory, cmp_using, define, field, setters, validators

_1 = np.int_(1)


def is_square(N: int) -> bool:
    return math.sqrt(N).is_integer()


def valid_square(instance, attribute, value):
    if not is_square(value):
        raise ValueError(
            f"{attribute.name} of {instance.name} must have an integer square root."
        )


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


def _convert(
    input: Union[
        str,
        npt.NDArray[np.int_],
        Tuple[int, ...],
        Tuple[Tuple[int, ...]],
        List[int],
        List[List[int]],
    ]
) -> npt.NDArray[np.int_]:
    """Converts input to a 2D numpy array of integers.
    "." and "0" are treated as empty cells.
    """
    _n: int = 0
    if isinstance(input, str):
        input = input.replace(",", "")
        input = "".join(input.split())
        input = input.replace(".", "0")
        input = np.array([int(c, base=36) for c in input])  # Allow for 36x36 puzzles
    elif isinstance(input, (tuple, list)):
        input = np.array(input)
    if input.ndim == 1:
        if not is_square(input.size):
            raise ValueError(
                f"Input string length ({len(input)}) is not a perfect square."
            )
        _n = int(math.sqrt(input.size))
        input = input.reshape((_n, _n))
    return input


@define
class Sudoku:
    _N: int = field(
        default=9,
        on_setattr=setters.frozen,
        validator=[validators.instance_of(int), validators.gt(0), valid_square],  # type: ignore
        alias="_N",
        kw_only=True,
    )
    puzzle: npt.NDArray[np.int_] = field(
        eq=cmp_using(eq=np.array_equal),
        converter=_convert,
        on_setattr=setters.frozen,
        default=Factory(
            lambda self: np.zeros((self._N, self._N), dtype=np.int_), takes_self=True
        ),
    )
    solution: npt.NDArray[np.int_] = field(
        eq=cmp_using(eq=np.array_equal),
        init=False,
        default=Factory(lambda self: self.puzzle.flatten(), takes_self=True),
    )
    constraint: Constraint = field(
        init=False,
        default=Factory(
            lambda self: Constraint.from_puzzle(self.puzzle), takes_self=True
        ),
        repr=False,
    )
    candidates: list[list[np.int_]] = field(init=False, on_setattr=setters.frozen)

    @candidates.default  # type: ignore
    def _set_candidates(self) -> list[list[int]]:
        frequency = [0 for _ in range(self._N)]
        c = [list() for _ in self.puzzle.flatten()]
        for i, j in np.ndindex(self.puzzle.shape):
            if self.puzzle[i, j] != 0:
                continue
            for k in range(self._N):
                if self.constraint.is_safe(i * self._N + j, k + _1):
                    c[i * self._N + j].append(np.int_(k + 1))
                    frequency[k - 1] += 1
        # for possible in c:
        #     possible.sort(key=lambda x: frequency[x-1],reverse=True)
        return c

    # install line_profiler with pip install line_profiler
    # kernprof -l -v sudoku.py to get a line-by-line profile
    # @profile
    def solve(self, i) -> bool:
        if i >= self._N**2:
            return True
        if self.solution[i] != 0:
            return self.solve(i + 1)
        for v in self.candidates[i]:
            if self.constraint.is_safe(i, v):
                self.solution[i] = v
                self.constraint.update(i, v)
                if self.solve(i + 1):
                    return True
                self.solution[i] = 0
                self.constraint.remove(i, v)
        return False


def main():
    nyt_hard = "...2........59.3.7......69......8....19....83..4.6....3...2.7.1.57..4....8..3...."
    puz4 = Sudoku(nyt_hard)
    puz4.solve(0)
    print(puz4)


if __name__ == "__main__":
    main()
