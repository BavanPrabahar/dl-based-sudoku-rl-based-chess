"""
chess_rl/train_supervised.py
Stage 1 (warm start): imitation learning of the solver moves of the supervised puzzle pool.

The policy head learns P(move | position) with cross-entropy on the solution moves; this gives the
reinforcement-learning stage (chess_rl/train_rl.py) a policy that already plays sensible chess, the same
recipe AlphaGo used (supervised policy first, then policy-gradient RL).

Example:
  python -m chess_rl.train_supervised --steps 40000 --save checkpoints/chess_supervised.pt
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from chess_rl.encoding import planes_from_encoded
from chess_rl.model import build, save_checkpoint
from chess_rl.puzzles import build_supervised_arrays

PUZZLES = "data/processed/lichess_puzzles.parquet"


def arrays(split, cache_dir="data/processed"):
    path = os.path.join(cache_dir, f"chess_{split}_samples.npz")
    if not os.path.exists(path):
        df = pd.read_parquet(PUZZLES, columns=["FEN", "Moves", "split"])
        n = build_supervised_arrays(df[df["split"] == split], path)
        print(f"encoded {n:,} {split} positions -> {path}")
    with np.load(path) as z:
        return z["enc"], z["target"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--save", default="checkpoints/chess_supervised.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc, tgt = arrays("sup")
    venc, vtgt = arrays("val")
    E, T = torch.from_numpy(enc).to(dev), torch.from_numpy(tgt).to(dev)
    VE, VT = torch.from_numpy(venc).to(dev), torch.from_numpy(vtgt).to(dev)
    model = build(args.blocks, args.channels).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ChessNet {n_params:,} params | {len(T):,} training positions | {len(VT):,} validation positions")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    log = open(args.save + ".log.jsonl", "w")
    t0, running, best = time.time(), None, -1
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(T), (args.batch,), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits, _ = model(planes_from_encoded(E[idx]))
        loss = F.cross_entropy(logits.float(), T[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            correct = 0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                for i in range(0, len(VT), 4096):
                    lg, _ = model(planes_from_encoded(VE[i:i + 4096]))
                    correct += (lg.argmax(1) == VT[i:i + 4096]).sum().item()
            model.train()
            acc = correct / len(VT)
            rec = {"step": step, "loss": round(running, 4), "val_move_acc_unmasked": round(acc, 4),
                   "minutes": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if acc > best:
                best = acc
                save_checkpoint(args.save, model, {"stage": "supervised", "step": step, "val_move_acc": acc,
                                                   "train_positions": int(len(T)), "args": vars(args)})
    print(f"best validation move accuracy {best:.4f} -> {args.save}")


if __name__ == "__main__":
    main()
