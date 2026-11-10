"""
sudoku/solve.py
Solve one 81-character Sudoku puzzle (0 or . = blank) with
  1. the exact backtracking solver (reference), and
  2. the trained network: one-shot and iterative decoding, each checked against the Sudoku rules.

Example:
  python -m sudoku.solve --puzzle 530070000600195000098000060800060003400803001700020006060000280000419005000080079
"""

import argparse
import os
import time

import numpy as np
import torch

from sudoku.backtracking_solver import solve_sudoku
from sudoku.board_utils import print_board
from sudoku.infer import is_solved, solve_iterative, solve_one_shot
from sudoku.model import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description="Solve a Sudoku with backtracking and with the trained network")
    ap.add_argument("--puzzle", default="530070000600195000098000060800060003400803001700020006060000280000419005000080079")
    ap.add_argument("--checkpoint", default="checkpoints/sudoku_constraint_ft.pt")
    args = ap.parse_args()
    puzzle = args.puzzle.strip().replace(".", "0")
    if len(puzzle) != 81 or not puzzle.isdigit():
        raise SystemExit("Error: the puzzle must be 81 characters, digits 1-9 and 0 or . for blanks.")

    board = np.array([int(c) for c in puzzle]).reshape(9, 9)
    print("Puzzle:")
    print_board(board)

    bt = board.copy()
    t = time.perf_counter()
    ok = solve_sudoku(bt)
    ms = (time.perf_counter() - t) * 1000
    if ok:
        print(f"Backtracking (exact): solved in {ms:.1f} ms")
        print_board(bt)
    else:
        print(f"Backtracking (exact): NO SOLUTION exists for this puzzle ({ms:.1f} ms)")

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"No trained checkpoint at {args.checkpoint}: train one with `python -m sudoku.train`.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device)
    p = torch.from_numpy(board.ravel().astype(np.uint8)).unsqueeze(0).to(device)
    for name, fn in (("one-shot", solve_one_shot), ("iterative", solve_iterative)):
        t = time.perf_counter()
        out = fn(model, p)
        ms = (time.perf_counter() - t) * 1000
        valid = bool(is_solved(out, p)[0])
        print(f"Network, {name} decoding ({ms:.1f} ms): {'VALID solution' if valid else 'NOT a valid solution'}")
        print_board(out[0].cpu().numpy().reshape(9, 9))


if __name__ == "__main__":
    main()
