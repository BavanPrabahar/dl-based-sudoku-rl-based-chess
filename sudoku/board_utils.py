"""
board_utils.py
Foundational utilities for Sudoku AI:
1. One-hot encoding and decoding
2. Constraint validation (Row, Column, 3x3 Box)
3. Board printing utilities
"""

import numpy as np


def print_board(board: np.ndarray) -> None:
    """Prints a 9x9 board in a human-friendly format with 3x3 dividers."""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - -")
        row_str = []
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row_str.append("|")
            val = board[i, j]
            row_str.append(str(val) if val != 0 else ".")
        print(" ".join(row_str))
    print()


def to_one_hot(board: np.ndarray) -> np.ndarray:
    """
    Converts a 9x9 board (integers 0..9) into a 9x9x9 binary tensor.
    Shape: (channels=9, rows=9, cols=9)
    Channel 0 corresponds to digit 1, Channel 8 corresponds to digit 9.
    Empty cells (0) remain all zeros across all 9 channels.
    """
    tensor = np.zeros((9, 9, 9), dtype=np.float32)
    for num in range(1, 10):
        # Wherever the board has 'num', set the corresponding channel slice to 1.0
        tensor[num - 1] = (board == num).astype(np.float32)
    return tensor


def from_one_hot(tensor: np.ndarray) -> np.ndarray:
    """
    Converts a (9, 9, 9) one-hot tensor back into a 2D 9x9 board.
    Takes the argmax across channel axis (axis 0).
    """
    has_digit = np.any(tensor > 0.5, axis=0)
    digits = np.argmax(tensor, axis=0) + 1  # 0..8 -> 1..9
    return np.where(has_digit, digits, 0).astype(int)


def is_valid_move(board: np.ndarray, row: int, col: int, num: int) -> bool:
    """
    Checks whether placing 'num' at (row, col) violates any Sudoku rules:
    1. Row check: 'num' must not be in the same row.
    2. Col check: 'num' must not be in the same column.
    3. Box check: 'num' must not be in the same 3x3 subgrid.
    """
    # 1. Check row
    if num in board[row, :]:
        return False

    # 2. Check column
    if num in board[:, col]:
        return False

    # 3. Check 3x3 subgrid
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3
    box = board[start_row : start_row + 3, start_col : start_col + 3]
    if num in box:
        return False

    return True


if __name__ == "__main__":
    sample_board = np.array([
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

    print("--- 1. Original Board ---")
    print_board(sample_board)

    tensor = to_one_hot(sample_board)
    print(f"One-Hot Tensor Shape: {tensor.shape} (Channels, Rows, Columns)")
    print(f"Channel 4 (representing digit 5) at (row 0, col 0): {tensor[4, 0, 0]}")

    restored = from_one_hot(tensor)
    assert np.array_equal(sample_board, restored), "Roundtrip failed!"
    print("✓ Roundtrip conversion (Board -> Tensor -> Board) succeeded!")

    print(f"Can place 5 at (0, 2)? {is_valid_move(sample_board, 0, 2, 5)} (Expected: False)")
    print(f"Can place 4 at (0, 2)? {is_valid_move(sample_board, 0, 2, 4)} (Expected: True)")
