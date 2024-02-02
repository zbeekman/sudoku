import pytest
import numpy as np
from sudoku import Sudoku

class TestSudoku:

        def test_puzzle_default(self):
            puz = Sudoku()
            assert puz.puzzle.shape == (9, 9), "Default puzzle shape is not (9, 9)"
            assert (puz.puzzle == 0).all(), "Default puzzle is not empty"

        def test_bad_input(self):
            with pytest.raises(ValueError):
                puz = Sudoku("123456789123")

        def test_puzzle_input(self):
            nyt_hard = "..2........59.3.7......69......8....19....83..4.6....3...2.7.1.57..4....8..3....."
            puz = Sudoku(nyt_hard)
            assert puz.puzzle.shape == (9, 9), "Puzzle shape is not (9, 9)"
            assert all([int(cell) in set(range(puz._N+1)) for cell in puz.puzzle.flatten()]), "Puzzle contains invalid cell values"


