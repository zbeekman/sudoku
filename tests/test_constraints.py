import pytest
import numpy as np
import copy
from sudokutoolbox import Sudoku, Constraint

_1 = np.int_(1)

_nyt_hard = np.array([
            [0,0,0, 2,0,0, 0,0,0],
            [0,0,0, 5,9,0, 3,0,7],
            [0,0,0, 0,0,0, 6,9,0],

            [0,0,0, 0,0,8, 0,0,0],
            [0,1,9, 0,0,0, 0,8,3],
            [0,0,4, 0,6,0, 0,0,0],

            [3,0,0, 0,2,0, 7,0,1],
            [0,5,7, 0,0,4, 0,0,0],
            [0,8,0, 0,3,0, 0,0,0],
    ], dtype=np.int_)

_nyth_sol = np.array([
        [9,6,5, 2,7,3, 8,1,4],
        [1,4,8, 5,9,6, 3,2,7],
        [7,2,3, 4,8,1, 6,9,5],

        [5,7,2, 3,4,8, 1,6,9],
        [6,1,9, 7,5,2, 4,8,3],
        [8,3,4, 1,6,9, 5,7,2],

        [3,9,6, 8,2,5, 7,4,1],
        [2,5,7, 6,1,4, 9,3,8],
        [4,8,1, 9,3,7, 2,5,6],
    ], dtype=np.int_)

class TestConstraints:
    def test_duplicate_row_insertion(self):
        nyth_const = Constraint.from_puzzle(_nyt_hard)
        for i, j in np.ndindex(_nyt_hard.shape):
            if _nyt_hard[i, j] != 0:
                for k in range(nyth_const.N):
                    l = i*nyth_const.N + k
                    assert not nyth_const.is_safe(l, _nyt_hard[i, j]), "Duplicate row insertion should not be allowed."

    def test_duplicate_col_insertion(self):
        nyth_const = Constraint.from_puzzle(_nyt_hard)
        for i, j in np.ndindex(_nyt_hard.shape):
            if _nyt_hard[i, j] != 0:
                for k in range(nyth_const.N):
                    l = k*nyth_const.N + j
                    assert not nyth_const.is_safe(l, _nyt_hard[i, j]), "Duplicate col insertion should not be allowed."

    def test_duplicate_box_insertion(self):
        nyth_const = Constraint.from_puzzle(_nyt_hard)
        for i, j in np.ndindex(_nyt_hard.shape):
            if _nyt_hard[i, j] != 0:
                box = nyth_const.b[i*nyth_const.N + j]
                for k in range(nyth_const.N):
                    for l in range(nyth_const.N):
                        if (k//nyth_const.N != box[0] or l//nyth_const.N != box[1]):
                            continue
                        m = k*nyth_const.N + l
                        assert not nyth_const.is_safe(m, _nyt_hard[i, j]), "Duplicate box insertion should not be allowed."

    def test_safe_to_insert_puzzle_solution(self):
        nyth_const = Constraint.from_puzzle(_nyt_hard)
        for i, j in np.ndindex(_nyt_hard.shape):
            if _nyt_hard[i, j] == 0:
                assert nyth_const.is_safe(i*nyth_const.N + j, _nyth_sol[i, j]), "Puzzle solution should be safe to insert."
            else:
                assert not nyth_const.is_safe(i*nyth_const.N + j, _nyth_sol[i, j]), "It should not be safe to doubly insert a known value."

    def test_safe_to_insert_full_solution(self):
        nyth_const = Constraint.from_puzzle(_nyt_hard)
        for i, j in np.ndindex(_nyt_hard.shape):
            if _nyt_hard[i, j] == 0:
                assert nyth_const.is_safe(i*nyth_const.N + j, _nyth_sol[i, j]), "Puzzle solution should be safe to insert, with constraint updating."
                nyth_const.update(i*nyth_const.N + j, _nyt_hard[i, j])
            else:
                assert not nyth_const.is_safe(i*nyth_const.N + j, _nyth_sol[i, j]), "It should not be safe to doubly insert a known value."

    def test_no_insertions_in_completed_puzzle(self):
        nyth_done_const = Constraint.from_puzzle(_nyth_sol)
        for i, j in np.ndindex(_nyt_hard.shape):
            for k in range(nyth_done_const.N):
                assert not nyth_done_const.is_safe(i*nyth_done_const.N + j, k+_1), "No insertions should be allowed in a completed puzzle."
                mask = np.int_(0)
        for l in range(1,nyth_done_const.N+1):
            mask |= np.int_(1) << l
        assert all(nyth_done_const.row == mask), "Row mask was not completely filled."
        assert all(nyth_done_const.col == mask), "Col mask was not completely filled."
        assert all(nyth_done_const.box.flatten() == mask), "Box mask was not completely filled."

    def test_double_add_is_a_noop(self):
        const = Constraint()
        const.update((const.N**2)//2, np.int_(9))
        const2 = copy.deepcopy(const)
        const.update((const.N**2)//2, np.int_(9))
        assert const == const2, "Double adding the same value should be a no-op."
        assert all(const.row == const2.row), "Row mask should not change."
        assert all(const.col == const2.col), "Col mask should not change."
        assert all(const.box.flatten() == const2.box.flatten()), "Box mask should not change."
        const.update((const.N**2)//2, np.int_(8))
        assert const != const2, "Adding a different value should change the constraint."
        assert not all(const.row == const2.row), "Row mask should change."
        assert not all(const.col == const2.col), "Col mask should change."
        assert not all(const.box.flatten() == const2.box.flatten()), "Box mask should change."

    def test_double_remove_is_a_noop(self):
        const = Constraint.from_puzzle(_nyth_sol)
        const2 = copy.deepcopy(const)
        const.remove((const.N**2)//2, np.int_(9))
        assert const != const2, "Removing a value should change the constraint."
        assert not all(const.row == const2.row), "Row mask should change."
        assert not all(const.col == const2.col), "Col mask should change."
        assert not all(const.box.flatten() == const2.box.flatten()), "Box mask should change."
        const2 = copy.deepcopy(const)
        assert const == const2, "Constraints should be the same after deep copy."
        const.remove((const.N**2)//2, np.int_(9))
        assert const == const2, "Double removing the same value should be a no-op."


    def test_removal(self):
        nyth_const = Constraint.from_puzzle(_nyt_hard)
        nyth_sol_const = Constraint.from_puzzle(_nyth_sol)
        assert nyth_const != nyth_sol_const, "Solution constraint should be different from the original constraint."
        for i, j in np.ndindex(_nyt_hard.shape):
            if _nyt_hard[i, j] != 0:
                assert _nyt_hard[i, j] == _nyth_sol[i, j], "Puzzle and solution should match for known values."
                const1 = copy.deepcopy(nyth_sol_const)
                assert not nyth_sol_const.is_safe(i*nyth_sol_const.N + j, _nyt_hard[i, j]), "It should not be safe to reinsert a known value."
                nyth_sol_const.remove(i*nyth_sol_const.N + j, _nyt_hard[i, j])
                print(nyth_sol_const)
                assert nyth_sol_const.is_safe(i*nyth_sol_const.N + j, _nyth_sol[i, j]), "Puzzle solution should be safe to insert after item removed."
                nyth_sol_const.update(i*nyth_sol_const.N + j, _nyth_sol[i, j])
                assert not nyth_sol_const.is_safe(i*nyth_sol_const.N + j, _nyt_hard[i, j]), "Backtracking restoration of constraint should allow reinsertion and detect attempted duplicate."
                print(const1)
                print(nyth_sol_const)
                assert const1 == nyth_sol_const, "Restored constraint should be the same as the original constraint."
        for i, j in np.ndindex(_nyt_hard.shape):
            if _nyt_hard[i, j] == 0:
                nyth_sol_const.remove(i*nyth_sol_const.N + j, _nyth_sol[i, j])
                assert nyth_sol_const.is_safe(i*nyth_sol_const.N + j, _nyt_hard[i, j]), "Puzzle solution should be safe to insert after item removed."
        assert nyth_sol_const == Constraint.from_puzzle(_nyt_hard), "Solution constraint should be the same as the original constraint after removal and reinsertion of solved cells."
        assert nyth_const == nyth_sol_const, "Solution constraint should be the same as the original constraint after removal of solved cells."
