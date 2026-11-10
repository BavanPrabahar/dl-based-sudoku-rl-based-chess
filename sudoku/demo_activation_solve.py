"""
demo_activation_solve.py
Teaching illustration (no neural network involved): hand-written rule scores play the role of
"logits" to show how conflicting digits get suppressed and the winner is picked with argmax.
For the trained network's decisions see sudoku/solve.py.
"""

import numpy as np
from sudoku.board_utils import is_valid_move, print_board

# Sample Sudoku puzzle
board = np.array([
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

print("=" * 45)
print("       CURRENT SUDOKU BOARD")
print("=" * 45)
print_board(board)

# We inspect Cell (0, 2) which is currently empty (0)
target_row, target_col = 0, 2
print(f"Inspecting Empty Cell at: Row {target_row}, Column {target_col}")
print("-" * 45)

# Simulate the activation scoring:
# Base activation starts at 0.
# If a digit violates Row, Col, or Box rules -> subtract large penalty.
# If a digit is clean and fits perfectly -> add positive activation.
logits = []
for num in range(1, 10):
    score = 0.0
    conflicts = []
    
    # 1. Check Row
    if num in board[target_row, :]:
        score -= 5.0
        conflicts.append(f"Row {target_row}")
        
    # 2. Check Column
    if num in board[:, target_col]:
        score -= 5.0
        conflicts.append(f"Col {target_col}")
        
    # 3. Check 3x3 Box
    start_r = (target_row // 3) * 3
    start_c = (target_col // 3) * 3
    box = board[start_r:start_r+3, start_c:start_c+3]
    if num in box:
        score -= 5.0
        conflicts.append("3x3 Box")
        
    if len(conflicts) == 0:
        score += 8.5  # Valid move receives high positive activation!
        print(f"Digit {num}: Logit = {score:+5.1f}  --> [CLEAN! No conflicts]")
    else:
        reason = ", ".join(conflicts)
        print(f"Digit {num}: Logit = {score:+5.1f}  --> [Conflict in {reason}]")
        
    logits.append(score)

logits = np.array(logits)
best_digit = np.argmax(logits) + 1  # 0-indexed to 1..9
print("-" * 45)
print(f"All 9 Logits: {list(np.round(logits, 1))}")
print(f">>> WINNER: Digit {best_digit} with highest activation ({logits[best_digit - 1]:+.1f})!")
print("=" * 45)

# Fill in the cell and show updated board
board[target_row, target_col] = best_digit
print("\nBoard updated with model's decision:")
print_board(board)
