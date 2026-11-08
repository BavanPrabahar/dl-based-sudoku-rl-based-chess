"""
chess_rl/train_rl.py
Stage 2: reinforcement learning on the Lichess puzzle environment (policy gradient, REINFORCE with a
learned baseline and a KL leash to the warm-start policy).

Every iteration:
  1. sample a batch of puzzles from the RL pool (never used by the supervised stage);
  2. the agent plays them by SAMPLING moves from its own policy; the environment answers with the puzzle
     line; an episode ends at the first wrong move (reward 0) or when the puzzle is solved (reward 1);
  3. update  L = -(R - b(s)) * log pi(a|s)       policy gradient, advantage vs. the value-head baseline
               + c_v * BCE(V(s), R)               baseline regression
               + c_kl * KL(pi || pi_warmstart)    keeps the policy close to sensible chess
               - c_ent * H(pi)                    exploration
No expert move labels are used in this stage: the only learning signal is the reward.

Example:
  python -m chess_rl.train_rl --init checkpoints/chess_supervised.pt --iters 1500 --save checkpoints/chess_rl.pt
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from chess_rl.agent import play_puzzles
from chess_rl.encoding import planes_from_encoded
from chess_rl.model import load_checkpoint, save_checkpoint
from chess_rl.puzzles import Episode

PUZZLES = "data/processed/lichess_puzzles.parquet"


def solve_rate(model, df, device):
    eps = [Episode.from_row(f, m) for f, m in zip(df["FEN"], df["Moves"])]
    play_puzzles(model, eps, device, greedy=True)
    return float(np.mean([e.solved for e in eps]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="checkpoints/chess_supervised.pt")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--episodes", type=int, default=1024, help="puzzles per iteration")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--c-value", type=float, default=0.5)
    ap.add_argument("--c-kl", type=float, default=0.1)
    ap.add_argument("--c-ent", type=float, default=0.001)
    ap.add_argument("--value-warmup", type=int, default=100,
                    help="first iterations train only the value baseline (policy frozen) so advantages are calibrated")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-size", type=int, default=10000)
    ap.add_argument("--save", default="checkpoints/chess_rl.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(PUZZLES, columns=["FEN", "Moves", "Rating", "split"])
    pool = df[df["split"] == "rl"].reset_index(drop=True)
    val = df[df["split"] == "val"].sample(n=args.val_size, random_state=args.seed)
    model, meta = load_checkpoint(args.init, dev)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    model.eval()   # BatchNorm statistics stay frozen; gradients still flow
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log = open(args.save + ".log.jsonl", "w")
    best = solve_rate(model, val, dev)
    print(f"RL pool {len(pool):,} puzzles | warm start val solve rate {best:.4f}", flush=True)
    log.write(json.dumps({"iter": 0, "val_solve_rate": best}) + "\n")
    save_checkpoint(args.save, model, {**meta, "stage": "rl", "iter": 0, "val_solve_rate": best})
    t0, rew_hist = time.time(), []
    for it in range(1, args.iters + 1):
        batch = pool.iloc[rng.integers(len(pool), size=args.episodes)]
        eps = [Episode.from_row(f, m) for f, m in zip(batch["FEN"], batch["Moves"])]
        traj = play_puzzles(model, eps, dev, greedy=False, temperature=args.temperature, record=True)
        reward = torch.tensor([float(e.solved) for e in eps], device=dev)
        R = reward[traj["episode"]]

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            planes = planes_from_encoded(traj["enc"])
            logits, v = model(planes)
            with torch.no_grad():
                ref_logits, _ = ref(planes)
        mask = traj["mask"]
        logp = F.log_softmax(logits.float().masked_fill(~mask, -1e9), dim=1)
        ref_logp = F.log_softmax(ref_logits.float().masked_fill(~mask, -1e9), dim=1)
        p = logp.exp()
        act_logp = logp.gather(1, traj["action"][:, None]).squeeze(1)
        adv = R - torch.sigmoid(v.float()).detach()
        loss_pi = -(adv * act_logp).mean()
        loss_v = F.binary_cross_entropy_with_logits(v.float(), R)
        kl = (p * (logp - ref_logp)).masked_fill(~mask, 0).sum(1).mean()
        ent = -(p * logp).masked_fill(~mask, 0).sum(1).mean()
        if it <= args.value_warmup:
            loss = loss_v                     # calibrate the baseline first; the policy is not changed yet
        else:
            loss = loss_pi + args.c_value * loss_v + args.c_kl * kl - args.c_ent * ent
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        rew_hist.append(reward.mean().item())

        if it % args.eval_every == 0 or it == args.iters:
            rate = solve_rate(model, val, dev)
            rec = {"iter": it, "phase": "value-warmup" if it <= args.value_warmup else "policy-gradient",
                   "train_reward": round(float(np.mean(rew_hist[-args.eval_every:])), 4),
                   "value_loss": round(loss_v.item(), 4),
                   "val_solve_rate": round(rate, 4), "kl": round(kl.item(), 4), "entropy": round(ent.item(), 3),
                   "minutes": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if rate > best:
                best = rate
                save_checkpoint(args.save, model, {**meta, "stage": "rl", "iter": it, "val_solve_rate": rate,
                                                   "rl_args": vars(args)})
    print(f"best validation solve rate {best:.4f} -> {args.save}")


if __name__ == "__main__":
    main()
