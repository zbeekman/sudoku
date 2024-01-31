import numpy as np
import numpy.typing as npt
from attrs import define, field, Factory, cmp_using, setters

@define
class Sudoku:
    _N: int = field(default=9, kw_only=True, alias='_N', on_setattr=setters.NO_OP)

    puzzle: npt.NDArray[np.int_] = field(eq=cmp_using(eq=np.array_equal), converter=np.array, on_setattr=setters.NO_OP)
    @puzzle.default # type: ignore
    def _puzzle_default(self) -> npt.NDArray[np.int_]:
        return np.zeros((self._N, self._N), dtype=np.int_)

    solution: npt.NDArray[np.int_] = field(eq=cmp_using(eq=np.array_equal), converter=np.array)
    @solution.default # type: ignore
    def _solution_default(self) -> npt.NDArray[np.int_]:
        return self.puzzle.copy()

if __name__ == '__main__':
    puz1 = Sudoku([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(puz1)
    print(puz1.puzzle)
    puz2=Sudoku()
    print(puz2)
    print(puz2.puzzle.shape)
