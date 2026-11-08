"""
sudoku/train.py
Train a Sudoku network on Sudoku-Extreme (or any .npz written by sudoku.dataset.prepare_sudoku_extreme).

* data live on the GPU as uint8 and every batch gets random Sudoku symmetries (augment),
* the loss is cross-entropy on the BLANK cells only (copying the clues is not something to learn),
* --reveal-frac F: on a fraction F of every batch, a random share of the blanks is filled with the true
  digits before the forward pass. Iterative decoding feeds the network its own partially solved boards; a
  network trained only on original puzzles has never seen "many digits filled, still hard" boards and
  makes confident mistakes on them (see README, Results),
* a fixed validation slice of the training file (never trained on) is scored every --eval-every steps
  with blank-cell accuracy and full-board accuracy (one-shot and iterative); the best checkpoint
  (iterative board accuracy) is kept.

Example:
  python -m sudoku.train --data data/processed/sudoku_extreme_train.npz --steps 60000 --save checkpoints/sudoku_constraint.pt
  python -m sudoku.train --init checkpoints/sudoku_constraint.pt --reveal-frac 0.5 --steps 10000 --lr 5e-4 \
         --save checkpoints/sudoku_constraint_ft.pt          # fine-tune on partially solved boards
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from sudoku.dataset import augment, load_npz, one_hot_batch
from sudoku.infer import group_matrix, is_solved, solve_iterative, solve_one_shot
from sudoku.model import build_model


@torch.no_grad()
def evaluate(model, puzzles, solutions, groups, iterative=True, batch=4096):
    model.eval()
    cells = correct = 0
    one_shot = iter_ok = 0
    for i in range(0, len(puzzles), batch):
        p, s = puzzles[i:i + batch], solutions[i:i + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=p.is_cuda):
            pred = solve_one_shot(model, p)
            blank = p == 0
            cells += blank.sum().item()
            correct += ((pred == s) & blank).sum().item()
            one_shot += is_solved(pred, p, groups).sum().item()
            if iterative:
                iter_ok += is_solved(solve_iterative(model, p, groups), p, groups).sum().item()
    model.train()
    n = len(puzzles)
    return {"cell_acc": correct / max(cells, 1), "board_acc_one_shot": one_shot / n,
            "board_acc_iterative": iter_ok / n if iterative else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/sudoku_extreme_train.npz")
    ap.add_argument("--arch", default="constraint", choices=["constraint", "cnn"])
    ap.add_argument("--blocks", type=int, default=16, help="constraint blocks (arch=constraint)")
    ap.add_argument("--layers", type=int, default=9, help="conv layers (arch=cnn)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-size", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--reveal-frac", type=float, default=0.0,
                    help="fraction of each batch with a random share of the blanks revealed (0 = off)")
    ap.add_argument("--init", default=None, help="start from this checkpoint (same architecture)")
    ap.add_argument("--save", default="checkpoints/sudoku_constraint.pt")
    ap.add_argument("--log", default=None, help="JSON-lines training log (default: <save>.log.jsonl)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_npz(args.data)
    n = len(data["puzzles"])
    order = np.random.default_rng(args.seed).permutation(n)
    val_idx, train_idx = order[: args.val_size], order[args.val_size:]
    P = torch.from_numpy(data["puzzles"]).to(device)
    S = torch.from_numpy(data["solutions"]).to(device)
    train_idx_t = torch.from_numpy(train_idx).to(device)
    val_p, val_s = P[torch.from_numpy(val_idx).to(device)], S[torch.from_numpy(val_idx).to(device)]
    groups = group_matrix(device)

    arch_kwargs = ({"num_blocks": args.blocks, "hidden_dim": args.hidden} if args.arch == "constraint"
                   else {"num_layers": args.layers, "hidden_dim": args.hidden})
    model = build_model(args.arch, **arch_kwargs).to(device)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=True)
        if ck["arch"] != args.arch or ck["arch_kwargs"] != arch_kwargs:
            raise SystemExit(f"--init {args.init} is {ck['arch']} {ck['arch_kwargs']}, not {args.arch} {arch_kwargs}")
        model.load_state_dict(ck["state_dict"])
        print(f"initialised from {args.init} (step {ck['meta'].get('step')})")
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = min(1000, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    log_path = args.log or args.save + ".log.jsonl"
    log = open(log_path, "w")
    print(f"{args.arch}: {n_params:,} parameters | train {len(train_idx):,} | val {len(val_idx):,} | device {device}")

    best, t0, running = -1.0, time.time(), 0.0
    model.train()
    for step in range(1, args.steps + 1):
        idx = train_idx_t[torch.randint(len(train_idx_t), (args.batch,), device=device)]
        p, s = P[idx], S[idx]
        if not args.no_augment:
            p, s = augment(p, s)
        if args.reveal_frac > 0:
            # reveal a uniformly random share of the blanks (true digits) on part of the batch
            rows = torch.rand(len(p), 1, device=device) < args.reveal_frac
            share = torch.rand(len(p), 1, device=device)
            reveal = rows & (p == 0) & (torch.rand(p.shape, device=device) < share)
            p = torch.where(reveal, s, p)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(one_hot_batch(p))
        logits = logits.float().view(-1, 9, 81)
        target = (s.long() - 1)
        loss_cells = F.cross_entropy(logits, target, reduction="none")      # (B, 81)
        blank = (p == 0).float()
        loss = (loss_cells * blank).sum() / blank.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()

        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(model, val_p, val_s, groups, iterative=True)
            rec = {"step": step, "loss": round(running, 4), "lr": sched.get_last_lr()[0],
                   "minutes": round((time.time() - t0) / 60, 1), **{k: round(v, 4) for k, v in m.items()}}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if m["board_acc_iterative"] > best:
                best = m["board_acc_iterative"]
                torch.save({"arch": args.arch, "arch_kwargs": arch_kwargs, "state_dict": model.state_dict(),
                            "meta": {"step": step, "val": m, "data": os.path.basename(args.data),
                                     "params": n_params, "args": vars(args), "init": args.init}}, args.save)
    print(f"best validation board accuracy (iterative): {best:.4f} -> {args.save}")


if __name__ == "__main__":
    main()
