"""
sudoku/evaluate.py
Evaluate Sudoku solvers on the held-out Sudoku-Extreme TEST split (never used for training or model selection).

Metrics, overall and per puzzle source:
  cell_acc            fraction of BLANK cells the network fills correctly in one forward pass
  board_acc_one_shot  whole board correct after one forward pass (all blanks at once)
  board_acc_iterative whole board correct with iterative decoding (one most-confident legal cell at a time)
  board_acc_checked   one-shot board if it passes the rule check, otherwise the iterative board. The check
                      uses only the Sudoku rules and the clues (no solution); every test puzzle has a unique
                      solution, so a board that passes is the correct one
A board counts as solved if it is complete, obeys every Sudoku rule and keeps all clues.
The exact backtracking solver is timed on a sample for reference (always correct when it finishes; a per-puzzle
time limit stops it on the hardest puzzles).

Example:
  python -m sudoku.evaluate --checkpoints checkpoints/sudoku_cnn.pt checkpoints/sudoku_constraint.pt
"""

import argparse
import json
import os
import signal
import time

import numpy as np
import torch

from sudoku.backtracking_solver import solve_sudoku
from sudoku.dataset import load_npz
from sudoku.infer import group_matrix, is_solved, solve_iterative, solve_one_shot
from sudoku.model import load_checkpoint


class UniformModel(torch.nn.Module):
    """No-network baseline: equal scores for every digit. With iterative decoding this is pure constraint
    propagation (cells with a single legal digit are filled first, otherwise the first legal option is taken),
    which shows how much of the accuracy the trained network actually contributes."""

    def forward(self, x):
        return torch.zeros_like(x)


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


@torch.no_grad()
def score(model, P, S, groups, batch=4096):
    blanks = correct = 0
    one = np.zeros(len(P), bool)
    it = np.zeros(len(P), bool)
    for i in range(0, len(P), batch):
        p, s = P[i:i + batch], S[i:i + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=p.is_cuda):
            pred = solve_one_shot(model, p)
            full = solve_iterative(model, p, groups)
        b = p == 0
        blanks += b.sum().item()
        correct += ((pred == s) & b).sum().item()
        one[i:i + batch] = is_solved(pred, p, groups).cpu().numpy()
        it[i:i + batch] = is_solved(full, p, groups).cpu().numpy()
    return correct / blanks, one, it


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default="data/processed/sudoku_extreme_test.npz")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="checkpoint paths; 'uniform' = no-network baseline")
    ap.add_argument("--max-puzzles", type=int, default=0, help="random subset of the test split (0 = all)")
    ap.add_argument("--backtracking-sample", type=int, default=100)
    ap.add_argument("--backtracking-timeout", type=float, default=5.0, help="seconds per puzzle")
    ap.add_argument("--out", default="results/sudoku_results.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_npz(args.test)
    if args.max_puzzles:
        keep = np.sort(np.random.default_rng(0).choice(len(d["puzzles"]), args.max_puzzles, replace=False))
        d = {k: (v[keep] if k != "source_names" else v) for k, v in d.items()}
    P, S =torch.from_numpy(d["puzzles"]).to(dev), torch.from_numpy(d["solutions"]).to(dev)
    src, names = d["source"], [str(n) for n in d["source_names"]]
    groups = group_matrix(dev)
    out = {"test_puzzles": int(len(P)), "sources": {n: int((src == i).sum()) for i, n in enumerate(names)}, "models": []}

    for path in args.checkpoints:
        if path == "uniform":
            model, meta = UniformModel().to(dev), {"params": 0, "step": None}
        else:
            model = load_checkpoint(path, dev)
            meta = torch.load(path, map_location="cpu", weights_only=True)["meta"]
        t0 = time.time()
        cell_acc, one, it = score(model, P, S, groups)
        res = {"checkpoint": "no network (uniform scores)" if path == "uniform" else os.path.basename(path), "params": meta.get("params"), "train_step": meta.get("step"),
               "cell_acc": round(cell_acc, 4), "board_acc_one_shot": round(float(one.mean()), 4),
               "board_acc_iterative": round(float(it.mean()), 4),
               "board_acc_checked": round(float((one | it).mean()), 4), "eval_seconds": round(time.time() - t0, 1),
               "by_source": {}}
        for i, n in enumerate(names):
            m = src == i
            if not m.any():
                continue
            res["by_source"][n] = {"n": int(m.sum()), "board_acc_one_shot": round(float(one[m].mean()), 4),
                                   "board_acc_iterative": round(float(it[m].mean()), 4),
                                   "board_acc_checked": round(float((one[m] | it[m]).mean()), 4)}
        print(json.dumps(res), flush=True)
        out["models"].append(res)

    # exact solver reference on a sample of each source (pure Python backtracking; may be slow on hard puzzles)
    rng = np.random.default_rng(0)
    signal.signal(signal.SIGALRM, _on_alarm)
    bt = {}
    for i, n in enumerate(names):
        if not (src == i).any():
            continue
        idx = rng.choice(np.flatnonzero(src == i), size=min(args.backtracking_sample, int((src == i).sum())), replace=False)
        times, ok, timeouts = [], 0, 0
        for k in idx:
            board = d["puzzles"][k].reshape(9, 9).astype(int).copy()
            t = time.perf_counter()
            signal.setitimer(signal.ITIMER_REAL, args.backtracking_timeout)
            try:
                solved = solve_sudoku(board)
            except _Timeout:
                solved, timeouts = False, timeouts + 1
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            times.append(time.perf_counter() - t)
            ok += bool(solved and (board.ravel() == d["solutions"][k]).all())
        bt[n] = {"n": len(idx), "solved_within_limit": ok / len(idx), "timeouts": timeouts,
                 "time_limit_s": args.backtracking_timeout, "median_ms": round(1000 * float(np.median(times)), 2)}
        print(n, bt[n], flush=True)
    out["backtracking"] = bt
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
