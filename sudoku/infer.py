"""
sudoku/infer.py
Batched GPU inference for the Sudoku networks.

* solve_one_shot  - one forward pass, every blank gets its arg-max digit (fast, often inconsistent).
* solve_iterative - fill ONE cell per step: the blank whose most likely *legal* digit is most confident,
                    then re-run the network on the updated board. Digits that already appear in the cell's
                    row, column or box are masked out, so the network only chooses between legal options.
                    No search / backtracking: a wrong early choice is never undone.
* is_solved       - a board counts as solved if it is complete, obeys all Sudoku rules and keeps every clue.
"""

import torch

from sudoku.dataset import one_hot_batch

_IDX = torch.arange(81)
ROW, COL = _IDX // 9, _IDX % 9
BOX = (ROW // 3) * 3 + COL // 3


def group_matrix(device) -> torch.Tensor:
    """(81, 27) incidence matrix: cell -> its row (0..8), column (9..17) and box (18..26)."""
    g = torch.zeros(81, 27, device=device)
    g[_IDX, ROW] = 1
    g[_IDX, 9 + COL] = 1
    g[_IDX, 18 + BOX] = 1
    return g


def legal_digits(boards: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """(B, 81) boards -> (B, 9, 81) bool: digit d+1 is not yet used in the cell's row/column/box."""
    oh = one_hot_batch(boards).view(-1, 9, 81)               # (B, 9, 81)
    used_in_group = oh @ groups                              # (B, 9, 27) count of digit in each group
    used_for_cell = used_in_group @ groups.T                 # (B, 9, 81) >0 if any group of the cell has it
    return used_for_cell == 0


@torch.no_grad()
def solve_one_shot(model, puzzles: torch.Tensor) -> torch.Tensor:
    logits = model(one_hot_batch(puzzles)).view(-1, 9, 81)
    pred = logits.argmax(dim=1).to(torch.uint8) + 1
    return torch.where(puzzles > 0, puzzles, pred)


@torch.no_grad()
def solve_iterative(model, puzzles: torch.Tensor, groups: torch.Tensor = None) -> torch.Tensor:
    groups = group_matrix(puzzles.device) if groups is None else groups
    boards = puzzles.clone()
    rows = torch.arange(boards.shape[0], device=boards.device)
    for _ in range(81):
        blank = boards == 0
        active = blank.any(dim=1)
        if not active.any():
            break
        logits = model(one_hot_batch(boards)).view(-1, 9, 81).float()
        logits = logits.masked_fill(~legal_digits(boards, groups), float("-inf"))
        probs = torch.softmax(logits, dim=1).nan_to_num(0.0)  # a blank with no legal digit -> all zeros
        conf, digit = probs.max(dim=1)                         # (B, 81)
        conf = conf.masked_fill(~blank, -1.0)
        cell = conf.argmax(dim=1)
        ok = active & (conf[rows, cell] > 0)                   # dead end (no legal digit) -> leave the board
        boards[rows[ok], cell[ok]] = (digit[rows, cell][ok] + 1).to(boards.dtype)
        if not ok.any():
            break
    return boards


def is_solved(boards: torch.Tensor, puzzles: torch.Tensor, groups: torch.Tensor = None) -> torch.Tensor:
    groups = group_matrix(boards.device) if groups is None else groups
    complete = (boards > 0).all(dim=1)
    counts = one_hot_batch(boards).view(-1, 9, 81) @ groups    # every digit exactly once per group
    rules = (counts == 1).all(dim=2).all(dim=1)
    clues = ((puzzles == 0) | (boards == puzzles)).all(dim=1)
    return complete & rules & clues
