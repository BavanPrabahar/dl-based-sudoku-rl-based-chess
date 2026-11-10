"""
sudoku/dataset.py
Sudoku data loading.

* SudokuDataset          - small CSV files (`quizzes,solutions` / `puzzle,solution` columns), used by the
                           notebook and data/sample_sudoku.csv.
* prepare_sudoku_extreme - converts the Sudoku-Extreme CSVs (Hugging Face `sapientinc/sudoku-extreme`,
                           columns source,question,answer,rating; '.' = blank) into compact .npz arrays.
* load_npz / augment     - fast GPU-side batches with the Sudoku symmetries as data augmentation.
"""

import csv
import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from sudoku.board_utils import to_one_hot


class SudokuDataset(Dataset):
    """Loads puzzles and solutions from a small CSV file (rows with malformed strings are rejected loudly)."""

    def __init__(self, csv_path: str, max_samples: Optional[int] = None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at: {csv_path}")
        self.quizzes, self.solutions = [], []
        skipped = 0
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            names = reader.fieldnames
            quiz_col = next((c for c in names if c.lower().strip() in ("quizzes", "puzzle", "quiz", "question")), names[0])
            sol_col = next((c for c in names if c.lower().strip() in ("solutions", "solution", "sol", "answer")), names[1])
            for idx, row in enumerate(reader):
                if max_samples and idx >= max_samples:
                    break
                q = row[quiz_col].strip().replace(".", "0")
                s = row[sol_col].strip()
                if len(q) == 81 and len(s) == 81 and q.isdigit() and s.isdigit():
                    self.quizzes.append(q)
                    self.solutions.append(s)
                else:
                    skipped += 1
        msg = f"Loaded {len(self.quizzes):,} Sudoku puzzle pairs from {os.path.basename(csv_path)}"
        print(msg + (f" (skipped {skipped} malformed rows)" if skipped else "") + ".")

    def __len__(self) -> int:
        return len(self.quizzes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        quiz = np.array([int(c) for c in self.quizzes[idx]], dtype=int).reshape(9, 9)
        sol = np.array([int(c) for c in self.solutions[idx]], dtype=int).reshape(9, 9)
        # Input: (9, 9, 9) one-hot board. Target: digit classes 0..8 for CrossEntropyLoss.
        return torch.from_numpy(to_one_hot(quiz)), torch.from_numpy(sol - 1).long()


# ----------------------------------------------------------------------------- Sudoku-Extreme
def prepare_sudoku_extreme(csv_path: str, out_path: str) -> dict:
    """Parse a Sudoku-Extreme CSV into uint8 arrays (fast to load, ~165 bytes per puzzle)."""
    puzzles, solutions, sources, ratings = [], [], [], []
    source_names = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q, a = row["question"].strip(), row["answer"].strip()
            if len(q) != 81 or len(a) != 81:
                continue
            puzzles.append(q.replace(".", "0").encode())
            solutions.append(a.encode())
            sources.append(source_names.setdefault(row["source"], len(source_names)))
            ratings.append(int(row["rating"]))
    to_arr = lambda rows: (np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(-1, 81) - ord("0")).astype(np.uint8)  # noqa: E731
    data = {"puzzles": to_arr(puzzles), "solutions": to_arr(solutions),
            "source": np.array(sources, dtype=np.int8), "rating": np.array(ratings, dtype=np.int32),
            "source_names": np.array(sorted(source_names, key=source_names.get))}
    np.savez(out_path, **data)
    return data


def load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def one_hot_batch(boards: torch.Tensor) -> torch.Tensor:
    """(B, 81) uint8 digits (0 = blank) -> (B, 9, 9, 9) float one-hot, on the boards' device."""
    oh = torch.zeros(boards.shape[0], 10, 81, device=boards.device)
    oh.scatter_(1, boards.long().unsqueeze(1), 1.0)
    return oh[:, 1:].reshape(-1, 9, 9, 9)


def augment(puzzles: torch.Tensor, solutions: torch.Tensor):
    """Random validity-preserving Sudoku symmetries for a (B, 81) batch, applied on the GPU:
    digit relabelling, transpose, permutation of bands/stacks and of rows/columns inside them."""
    b = puzzles.shape[0]
    dev = puzzles.device
    # digit relabelling (0 = blank stays 0)
    perm = torch.argsort(torch.rand(b, 9, device=dev), dim=1) + 1
    lut = torch.cat([torch.zeros(b, 1, dtype=torch.long, device=dev), perm], dim=1)
    p = torch.gather(lut, 1, puzzles.long())
    s = torch.gather(lut, 1, solutions.long())
    p, s = p.view(b, 9, 9), s.view(b, 9, 9)

    def perm_lines():
        bands = torch.argsort(torch.rand(b, 3, device=dev), dim=1)
        inner = torch.argsort(torch.rand(b, 3, 3, device=dev), dim=2)
        return (bands.unsqueeze(2) * 3 + inner).view(b, 9)

    rows, cols = perm_lines(), perm_lines()
    p = torch.gather(p, 1, rows.unsqueeze(2).expand(b, 9, 9)); s = torch.gather(s, 1, rows.unsqueeze(2).expand(b, 9, 9))
    p = torch.gather(p, 2, cols.unsqueeze(1).expand(b, 9, 9)); s = torch.gather(s, 2, cols.unsqueeze(1).expand(b, 9, 9))
    flip = torch.rand(b, device=dev) < 0.5
    p = torch.where(flip.view(b, 1, 1), p.transpose(1, 2), p)
    s = torch.where(flip.view(b, 1, 1), s.transpose(1, 2), s)
    return p.reshape(b, 81), s.reshape(b, 81)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ds = SudokuDataset(os.path.join(here, "..", "data", "sample_sudoku.csv"))
    quiz, target = ds[0]
    print(f"quiz one-hot {tuple(quiz.shape)}, target {tuple(target.shape)}, row 0: {target[0].numpy() + 1}")
