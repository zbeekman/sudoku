from attrs import define, frozen, field, setters
from math import sqrt
import inspect
from colorama import init as colorama_init
from colorama import Fore, Style

colorama_init()

@define
class Sudoku:
    """A class representing a sudoku puzzle"""
    _N = 9
    _NN = _N*_N
    _BL = int(sqrt(_N))

    puzzle: tuple[int, ...]
    solution: list[int]
    cell_candidates: list[set[int]]
    cols: list[set[int]]
    rows: list[set[int]]
    squares: dict[tuple[int, int], set[int]]
    cells_to_solve: int = _NN

    @staticmethod
    def _r(i: int) -> int:
        """Returns the row index of the ith element in the puzzle"""
        return i // Sudoku._N
    @staticmethod
    def _c(i: int) -> int:
        """Returns the column index of the ith element in the puzzle"""
        return i % Sudoku._N
    @staticmethod
    def _b(i: int) -> tuple[int, int]:
        """Returns the block index of the ith element in the puzzle"""
        return (Sudoku._r(i) // Sudoku._BL, Sudoku._c(i) // Sudoku._BL)
    
    def cell_is_valid(self, i) -> bool:
        """Returns whether the puzzle is valid and populates the rows, cols, and squares sets"""
        _r = self._r
        _c = self._c
        _b = self._b
        if self.solution[i] == 0:
            return True
        if self.solution[i] in self.rows[_r(i)] | self.cols[_c(i)] | self.squares[_b(i)]:
            return False
        self.rows[_r(i)].add(self.solution[i])
        self.cols[_c(i)].add(self.solution[i])
        self.squares[_b(i)].add(self.solution[i])
        return True
    
    def prune_candidates(self, i) -> None:
        """Prunes the candidates for each cell and updates the solution, rows, cols and boxes if a cell has only one candidate"""
        _r = self._r
        _c = self._c
        _b = self._b
        if self.solution[i] == 0:
            self.cell_candidates[i] -= self.rows[_r(i)] | self.cols[_c(i)] | self.squares[_b(i)]
            if len(self.cell_candidates[i]) == 1:
                self.solution[i] = self.cell_candidates[i].pop()
                self.rows[_r(i)].add(self.solution[i])
                self.cols[_c(i)].add(self.solution[i])
                self.squares[_b(i)].add(self.solution[i])
                self.cells_to_solve -= 1
        else:
            self.cell_candidates[i] = set()
    
    def solve(self,i) -> bool:
        """Recursively solves the puzzle using back tracking"""
        _r = self._r
        _c = self._c
        _b = self._b
        if i >= Sudoku._NN:
            return True
        if self.solution[i] != 0:
            return self.solve(i+1)
        else:
            for candidate in self.cell_candidates[i]:
                self.solution[i] = candidate
                if self.cell_is_valid(i):
                    self.cells_to_solve -= 1
                    if self.solve(i+1):
                        return True
                    self.cells_to_solve += 1
                    self.rows[_r(i)].remove(self.solution[i])
                    self.cols[_c(i)].remove(self.solution[i])
                    self.squares[_b(i)].remove(self.solution[i])
                self.solution[i] = 0
            return False


    def __init__(self, puzzle: tuple[int, ...]) -> None:
        if not len(puzzle) == Sudoku._NN:
            raise ValueError(f"Invalid puzzle, must be {Sudoku._NN} elements long")
        self.cells_to_solve = Sudoku._NN
        self.puzzle = puzzle
        self.solution = list(puzzle)
        self.cols = [set() for _ in range(Sudoku._N)]
        self.rows = [set() for _ in range(Sudoku._N)]
        self.squares = {(i,j): set() for i in range(Sudoku._BL) for j in range(Sudoku._BL)}
        for i in range(Sudoku._NN):
            if self.puzzle[i] != 0:
                self.cells_to_solve -= 1
            if not self.cell_is_valid(i):
                raise ValueError("Invalid puzzle")
        self.cell_candidates = [set(range(1,10)) for i in range(Sudoku._NN)]
        _to_solve = self.cells_to_solve
        _n = 1
        while _n > 0:
            for i in range(Sudoku._NN):
                self.prune_candidates(i)
            _n = self.cells_to_solve - _to_solve
            _to_solve = self.cells_to_solve

    def __str__(self) -> str:
        """Returns a pretty string representation of the puzzle"""
        board_ = ""
        for i in range(Sudoku._N):
            if i % Sudoku._BL == 0 and i != 0:
                board_ += "\n"
            for j in range(Sudoku._N):
                if j % Sudoku._BL == 0 and j != 0:
                    board_ += " "
                cell_ = self.solution[i*Sudoku._N+j]
                if self.puzzle[i*Sudoku._N+j] != 0:
                    board_ += Fore.BLUE + str(cell_) + Style.RESET_ALL + " "
                else:
                    board_ += (str(cell_) if cell_ != 0 else ".") + " "
            board_ += "\n"
        return board_


if __name__ == "__main__":
    # from pyinstrument import Profiler
    # with Profiler(interval=0.00001) as profiler:
    board = Sudoku((
            0,0,0, 2,0,0, 0,0,0,
            0,0,0, 5,9,0, 3,0,7,
            0,0,0, 0,0,0, 6,9,0,

            0,0,0, 0,0,8, 0,0,0,
            0,1,9, 0,0,0, 0,8,3,
            0,0,4, 0,6,0, 0,0,0,

            3,0,0, 0,2,0, 7,0,1,
            0,5,7, 0,0,4, 0,0,0,
            0,8,0, 0,3,0, 0,0,0,
            ))
    # print(profiler.output_text(unicode=True, color=True))
    # print(board)
    # print(board.__repr__())
    # print(inspect.getsource(Sudoku.__repr__))
    # with Profiler(interval=0.001) as profiler2:
    board.solve(0)
    # print(profiler2.output_text(unicode=True, color=True))
    print(board)
    # print(board.__repr__())