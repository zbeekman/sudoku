from attrs import define, frozen, field, setters
from math import sqrt
import inspect
from colorama import init as colorama_init
from colorama import Fore, Style
from hashlib import sha1

colorama_init()

@define
class Sudoku:
    """A class representing a sudoku puzzle"""
    _N = 9
    _NN = _N*_N
    _BL = int(sqrt(_N))

    _r: tuple[int, ...] = field(repr=False)
    _c: tuple[int, ...] = field(repr=False)
    _b: tuple[tuple[int, int], ...] = field(repr=False)
    puzzle_id: str
    puzzle: tuple[int, ...]
    solution_id: str
    solution: list[int]
    _temp: list[int] = field(init=False, repr=False)
    cell_candidates: list[list[int]]
    cols: list[set[int]]
    rows: list[set[int]]
    squares: dict[tuple[int, int], set[int]]
    solution_order: tuple[int, ...]
    n_solutions: int

    @staticmethod
    def _get_rcb(i: int) -> tuple[int, int, tuple[int, int]]:
        """Returns the row, column and block index of the ith element in the puzzle"""
        r = i // Sudoku._N
        c = i % Sudoku._N
        b = (r // Sudoku._BL, c // Sudoku._BL)
        return (r, c, b)

    def cell_is_valid(self, i) -> bool:
        """Returns whether the puzzle is valid and populates the rows, cols, and squares sets"""
        if (self.solution[i] in self.rows[self._r[i]] or
            self.solution[i] in self.cols[self._c[i]] or
            self.solution[i] in self.squares[self._b[i]]):
            return False
        if self.solution[i] == 0:
            return True
        self.rows[self._r[i]].add(self.solution[i])
        self.cols[self._c[i]].add(self.solution[i])
        self.squares[self._b[i]].add(self.solution[i])
        return True

    def candidate_inserted_if_valid(self, i, candidate) -> bool:
        """Inserts the candidate into the solution if it is valid and returns whether the candidate was inserted"""
        if (candidate in self.rows[self._r[i]] or
            candidate in self.cols[self._c[i]] or
            candidate in self.squares[self._b[i]]):
            return False
        self.solution[i] = candidate
        self.rows[self._r[i]].add(candidate)
        self.cols[self._c[i]].add(candidate)
        self.squares[self._b[i]].add(candidate)
        return True

    def solution_is_valid(self) -> bool:
        """Returns whether the solution is valid"""
        self.cols = [set() for _ in range(Sudoku._N)]
        self.rows = [set() for _ in range(Sudoku._N)]
        self.squares = {(i,j): set() for i in range(Sudoku._BL) for j in range(Sudoku._BL)}
        for i in range(Sudoku._NN):
            if not self.cell_is_valid(i):
                return False
        return True

    def prune_candidates(self, i) -> None:
        """Prunes the candidates for each cell and updates the solution, rows, cols and boxes if a cell has only one candidate.
        This method can be called repeatedly to solve the puzzle using elimination. Solving the puzzle using elimination
        should be faster than using back tracking. It also gaurantees that any cells filled in are correct and unique given
        the starting puzzle state."""
        _r, _c, _b = self._get_rcb(i)
        if self.solution[i] == 0:
            self.cell_candidates[i] = list(set(self.cell_candidates[i]) - (self.rows[_r] | self.cols[_c] | self.squares[_b]))
            if len(self.cell_candidates[i]) == 1:
                self.solution[i] = self.cell_candidates[i].pop()
                self.rows[_r].add(self.solution[i])
                self.cols[_c].add(self.solution[i])
                self.squares[_b].add(self.solution[i])
                return
        else:
            self.cell_candidates[i] = list()
    def solve(self, i, find_duplicates=False) -> bool:
        print(f"finding duplicates: {find_duplicates}")
        is_solved = self._solve(i, find_duplicates)
        if self.n_solutions > 0:
            self.solution = self._temp
            self.solution_id = sha1(str(self.solution).encode()).hexdigest()
            return True
        return is_solved

    def _sort_by_candidate_frequency(self) -> list[float]:
        """Sorts each set of candidates by frequency of occurance and returns a list
        of the sums of the frequencies for each possible candidate a given cell"""
        candidate_frequency = [0 for _ in range(Sudoku._N)]
        for i in range(Sudoku._NN):
            for candidate in self.cell_candidates[i]:
                candidate_frequency[candidate-1] = candidate_frequency[candidate-1] + 1
        for i in range(Sudoku._NN):
            self.cell_candidates[i].sort(key=lambda candidate: candidate_frequency[candidate-1],reverse=True)
        # Return the average frequency of candidates for each cell
        frequencies = [sum(candidate_frequency[candidate-1] for candidate in self.cell_candidates[i])/len(self.cell_candidates[i]) if len(self.cell_candidates[i]) > 0 else 0 for i in range(Sudoku._NN)]
        # numbers = [i for i in range(1, Sudoku._N + 1)]
        # print(numbers)
        # print(candidate_frequency)
        # numbers.sort(key=lambda i: candidate_frequency[i-1])
        # print(numbers)

        return frequencies

    def _solve(self,i, find_duplicates=False) -> bool:
        """Recursively solves the puzzle using back tracking"""
        if i >= len(self.solution_order):
            if self.n_solutions == 0:
                self._temp = self.solution.copy()
            self.n_solutions = self.n_solutions + 1
            return True
        else:
            have_solution = False
            for candidate in self.cell_candidates[self.solution_order[i]]:
                if self.candidate_inserted_if_valid(self.solution_order[i], candidate):
                    if (have_solution := self._solve(i+1, find_duplicates)):
                        if (not find_duplicates) or self.n_solutions > 1:
                            return True
                    self.rows[self._r[self.solution_order[i]]].remove(candidate)
                    self.cols[self._c[self.solution_order[i]]].remove(candidate)
                    self.squares[self._b[self.solution_order[i]]].remove(candidate)
                    self.solution[self.solution_order[i]] = 0
            return have_solution


    def __init__(self, puzzle: tuple[int, ...]) -> None:
        if not len(puzzle) == Sudoku._NN:
            raise ValueError(f"Invalid puzzle, must be {Sudoku._NN} elements long")
        self._r = tuple(i // Sudoku._N for i in range(Sudoku._NN))
        self._c = tuple(i % Sudoku._N for i in range(Sudoku._NN))
        self._b = tuple((self._r[i] // Sudoku._BL, self._c[i] // Sudoku._BL) for i in range(Sudoku._NN))
        self.puzzle = puzzle
        self._temp = list(puzzle)
        self.puzzle_id = sha1(str(puzzle).encode()).hexdigest()
        self.solution = list(puzzle)
        self.solution_id = ""
        self.n_solutions = 0
        self.cell_candidates = [list(range(1,Sudoku._N + 1)) for i in range(Sudoku._NN)]
        if not self.solution_is_valid():
            raise ValueError(f"Invalid puzzle:\n{self}")
        _solved = self.puzzle.count(0)
        cross_hatch = True
        while cross_hatch:
            for i in range(Sudoku._NN):
                self.prune_candidates(i)
            cross_hatch = False
            if cross_hatch := self.solution.count(0) != _solved:
                _solved = self.solution.count(0)
        solution_order = [i for i in range(Sudoku._NN) if self.solution[i] == 0]
        # solution_order.sort(key=lambda i: self._sort_by_candidate_frequency()[i])
        frequencies = self._sort_by_candidate_frequency()
        # solution_order.sort(key=lambda i: len(self.cell_candidates[i]))
        self.solution_order = tuple(solution_order)
        # for i in solution_order:
        #     print(f"{i}: {self.cell_candidates[i]}")

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
                    color = Fore.BLUE if cell_ == self.puzzle[i*Sudoku._N+j] else Fore.RED
                    board_ += Fore.BLUE + str(cell_) + Style.RESET_ALL + " "
                else:
                    board_ += (str(cell_) if cell_ != 0 else ".") + " "
            board_ += "\n"
        return board_


if __name__ == "__main__":
    # from pyinstrument import Profiler
    # with Profiler(interval=0.00001) as profiler:
    # NYT hard puzzle
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
    # NYT hard puzzle, solution:
    """ 9 6 5  2 7 3  8 1 4
        1 4 8  5 9 6  3 2 7
        7 2 3  4 8 1  6 9 5

        5 7 2  3 4 8  1 6 9
        6 1 9  7 5 2  4 8 3
        8 3 4  1 6 9  5 7 2

        3 9 6  8 2 5  7 4 1
        2 5 7  6 1 4  9 3 8
        4 8 1  9 3 7  2 5 6 """
    # NYT easy puzzle, solved using elimination during pruning
    """ board = Sudoku((
            0,0,4, 0,6,0, 3,0,0,
            0,7,6, 0,3,0, 9,2,0,
            1,0,3, 8,5,0, 0,6,0,

            0,1,0, 0,4,5, 0,0,3,
            0,9,0, 0,0,2, 1,0,5,
            4,6,0, 3,0,1, 0,8,0,

            0,0,0, 9,2,8, 0,3,0,
            8,3,2, 0,0,0, 0,7,9,
            0,0,0, 5,0,3, 0,0,2,
            )) """
    # NYT easy puzzle, solution:
    """
    9 8 4  2 6 7  3 5 1
    5 7 6  1 3 4  9 2 8
    1 2 3  8 5 9  7 6 4

    2 1 8  7 4 5  6 9 3
    3 9 7  6 8 2  1 4 5
    4 6 5  3 9 1  2 8 7

    7 5 1  9 2 8  4 3 6
    8 3 2  4 1 6  5 7 9
    6 4 9  5 7 3  8 1 2 """
    # print(profiler.output_text(unicode=True, color=True))
    print(board)
    print(board.__repr__())
    # print(inspect.getsource(Sudoku.__repr__))
    # with Profiler(interval=0.001) as profiler2:
    if not board.solve(0, find_duplicates=True):
        print(board)
        print(board.__repr__())
        raise ValueError(f"No solution found for puzzle {board.puzzle_id}")

    if not board.solution_is_valid():
        print(board)
        print(board.__repr__())
        raise ValueError(f"Invalid solution for puzzle {board.puzzle_id}")
    diff_puzzle = []
    for i in range(Sudoku._NN):
        if board.puzzle[i] != 0 and board.puzzle[i] != board.solution[i]:
            diff_puzzle[i] = board.solution[i]
    if len(diff_puzzle) != 0:
        board.solution = board.solution.copy()
        raise ValueError(f"Invalid solution for puzzle {board.puzzle_id}")
    # print(profiler2.output_text(unicode=True, color=True))
    print(board)
    print(board.__repr__())

