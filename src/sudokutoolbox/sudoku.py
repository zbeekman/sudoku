import math
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
from attrs import Factory, cmp_using, define, field, setters, validators

from sudokutoolbox.util import is_square, valid_square
from sudokutoolbox.constraint import Constraint, _1


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
        for possible in c:
            possible.sort(key=lambda x: frequency[x - 1], reverse=True)
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
