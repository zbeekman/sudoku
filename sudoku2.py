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

    puzzle: tuple[tuple[int, ...], ...]
    solution: list[list[int]]
    cell_candidates: list[list[set[int]]]
    cols: list[set[int]]
    rows: list[set[int]]
    squares: dict[tuple[int, int], set[int]]

    def puzzle_is_valid(self) -> bool:
        """Returns whether the puzzle is valid"""
        self.rows = [set() for _ in range(Sudoku._N)]
        self.cols = [set() for _ in range(Sudoku._N)]
        self.squares = {(i,j): set() for i in range(Sudoku._BL) for j in range(Sudoku._BL)}
        for i in range(Sudoku._N):
            for j in range(Sudoku._N):
                if self.solution[i][j] == 0:
                    continue
                if (self.solution[i][j] in self.rows[i] or
                    self.solution[i][j] in self.cols[j] or
                    self.solution[i][j] in self.squares[(i//Sudoku._BL, j//Sudoku._BL)]):
                    return False
                self.rows[i].add(self.solution[i][j])
                self.cols[j].add(self.solution[i][j])
                self.squares[(i//Sudoku._BL, j//Sudoku._BL)].add(self.solution[i][j])
        return True

    def is_safe_to_insert(self, i, j, val) -> bool:
        """Returns whether it is safe to insert val at (i,j)"""
        if (val in self.solution[i] or
            val in [self.solution[k][j] for k in range(Sudoku._N)] or
            val in [self.solution[i//Sudoku._BL*Sudoku._BL+k][j//Sudoku._BL*Sudoku._BL+l] for k in range(Sudoku._BL) for l in range(Sudoku._BL)]):
            return False
        return True

    def solve(self,i,j) -> bool:
        """Recursively solves the puzzle using back tracking"""
        if i >= Sudoku._N:
            return True
        if j >= Sudoku._N:
            return self.solve(i+1, 0)
        if self.solution[i][j] != 0:
            return self.solve(i, j+1)
        else:
            for candidate in self.cell_candidates[i][j]:
                if self.is_safe_to_insert(i, j, candidate):
                    self.solution[i][j] = candidate
                    if self.solve(i, j+1):
                        return True
                    self.solution[i][j] = 0
            return False

    def prune_candidates(self) -> bool:
        """Prunes the candidates for each cell and updates the solution, rows, cols and boxes if a cell has only one candidate"""
        modified = False
        for i in range(Sudoku._N):
            for j in range(Sudoku._N):
                if self.solution[i][j] == 0:
                    self.cell_candidates[i][j] -= self.rows[i] | self.cols[j] | self.squares[(i//Sudoku._BL, j//Sudoku._BL)]
                    if len(self.cell_candidates[i][j]) == 1:
                        self.solution[i][j] = self.cell_candidates[i][j].pop()
                        self.rows[i].add(self.solution[i][j])
                        self.cols[j].add(self.solution[i][j])
                        self.squares[(i//Sudoku._BL,j//Sudoku._BL)].add(self.solution[i][j])
                        modified = True
                else:
                    self.cell_candidates[i][j] = set()
        if modified:
            return self.prune_candidates()
        return modified

    def __init__(self, puzzle: tuple[tuple[int, ...]]) -> None:
        if not len(puzzle) == Sudoku._N or not all(len(row) == Sudoku._N for row in list(puzzle)):
            raise ValueError(f"Invalid puzzle, must be {Sudoku._N} by {Sudoku._N} square")
        self.puzzle = puzzle
        self.solution = [list(row) for row in list(puzzle)]
        self.cell_candidates = [[set(range(1,Sudoku._N + 1)) for _ in range(Sudoku._N)] for _ in range(Sudoku._N)]
        if not self.puzzle_is_valid():
            print(self.__repr__() + "\n")
            raise ValueError(f"Invalid puzzle")
        self.prune_candidates()

    def __str__(self) -> str:
        """Returns a pretty string representation of the puzzle"""
        board_ = ""
        for i in range(Sudoku._N):
            if i % Sudoku._BL == 0 and i != 0:
                board_ += "\n"
            for j in range(Sudoku._N):
                if j % Sudoku._BL == 0 and j != 0:
                    board_ += " "
                cell_ = self.solution[i][j]
                if self.puzzle[i][j] != 0:
                    board_ += Fore.BLUE + str(cell_) + Style.RESET_ALL + " "
                else:
                    board_ += (str(cell_) if cell_ != 0 else ".") + " "
            board_ += "\n"
        return board_


if __name__ == "__main__":
    # from pyinstrument import Profiler
    # with Profiler(interval=0.00001) as profiler:
    bo = (
            (0,0,0, 2,0,0, 0,0,0),
            (0,0,0, 5,9,0, 3,0,7),
            (0,0,0, 0,0,0, 6,9,0),

            (0,0,0, 0,0,8, 0,0,0),
            (0,1,9, 0,0,0, 0,8,3),
            (0,0,4, 0,6,0, 0,0,0),

            (3,0,0, 0,2,0, 7,0,1),
            (0,5,7, 0,0,4, 0,0,0),
            (0,8,0, 0,3,0, 0,0,0),
            )
    board = Sudoku(bo) # type: ignore
    # print(profiler.output_text(unicode=True, color=True))
    print(board)
    # print(board.__repr__())
    # print(inspect.getsource(Sudoku.__repr__))
    # with Profiler(interval=0.001) as profiler2:
    board.solve(0,0)
    # print(profiler2.output_text(unicode=True, color=True))
    print(board)
    # print(board.__repr__())
