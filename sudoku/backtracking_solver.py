"""
backtracking_solver.py
An exact, deterministic Sudoku solver using recursive backtracking.
Acts as our ground-truth benchmark and data generator.
"""

import time
from typing import Optional, Tuple
import numpy as np
from sudoku.board_utils import is_valid_move, print_board


def find_empty_cell(board: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Finds the first empty cell (value 0) on the board.
    Returns (row, col) or None if the board is completely filled.
    """
    for r in range(9):
        for c in range(9):
            if board[r, c] == 0:
                return (r, c)
    return None


def solve_sudoku(board: np.ndarray) -> bool:
    """
    Solves a Sudoku board in-place using recursive backtracking.
    Returns True if a solution was found, False if unsolvable.
    """
    empty = find_empty_cell(board)
    if empty is None:
        # Base case: No empty cells left, puzzle is solved!
        return True

    row, col = empty

    for num in range(1, 10):
        if is_valid_move(board, row, col, num):
            board[row, col] = num  # Tentatively place 'num'

            # Recurse: Try solving the rest of the board
            if solve_sudoku(board):
                return True

            # If placing 'num' didn't work out, undo it (Backtrack!)
            board[row, col] = 0

    return False


if __name__ == "__main__":
    # Test on a classic puzzle
    puzzle = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ], dtype=int)

    print("--- Unsolved Puzzle ---")
    print_board(puzzle)

    board_copy = puzzle.copy()
    start_time = time.perf_counter()
    success = solve_sudoku(board_copy)
    duration_ms = (time.perf_counter() - start_time) * 1000

    if success:
        print(f"--- Solved in {duration_ms:.2f} ms ---")
        print_board(board_copy)
    else:
        print("No solution exists.")
