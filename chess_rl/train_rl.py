"""
chess_rl/train_rl.py
Stage 2: reinforcement learning on the Lichess puzzle environment (policy gradient with a KL leash to the
warm-start policy). The agent only ever gets the reward (puzzle solved or not), never the solution moves.

Every iteration:
  1. sample puzzles from the RL pool (never used by the supervised stage), by default only those rated
     >= --min-rating: the warm start already solves ~99% of the easy ones, so they carry no signal;
  2. the agent plays every puzzle --group times by SAMPLING moves from its own policy; the environment
     answers with the puzzle line; an episode ends at the first wrong move (reward 0) or when the puzzle is
     solved (reward 1);
  3. update  L = -A * log pi(a|s)  +  c_kl * KL(pi || pi_warmstart)  -  c_ent * H(pi)  (+ value-head BCE)

Advantage A of a move (--group G > 1, "tree advantage"):
  The opponent's replies are fixed, so the G attempts at one puzzle form a small tree of agent moves. For
  every node (puzzle + agent moves so far) the success rate V(node) is the mean reward of the attempts that
  passed through it, and the advantage of a move is  A = V(node after the move) - V(node before).
  A wrong move gets A = -V(before) (it threw away that chance); a correct move gets V(after) - V(before) >= 0.
  So when an attempt fails, the blame goes to the move that was actually wrong, not to the correct moves
  before it. Puzzles the agent always solves (or always fails) give A = 0 everywhere.
  No value network is needed for this.

--group 1 is the original method: plain REINFORCE, A = R - V(s) with the value head as the baseline, every
move of a failed attempt gets the same negative signal. The value head is trained in both modes (on
detached trunk features, so it never changes the policy).

Examples:
  python -m chess_rl.train_rl --init checkpoints/chess_supervised.pt --save checkpoints/chess_rl.pt
  python -m chess_rl.train_rl --group 1 --puzzles 1024 --min-rating 0 --value-warmup 100 --iters 1500   # v1
"""

import argparse
import copy
import json
import time
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
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


def tree_advantage(ep_ids, actions, rewards, group):
    """Per-move advantage V(after) - V(before) over the tree of `group` attempts per puzzle.
    ep_ids/actions are in recording order (each episode's moves appear in move order); episode e belongs
    to puzzle e // group."""
    hist = defaultdict(tuple)
    before, after = [], []
    for e, a in zip(ep_ids.tolist(), actions.tolist()):
        kb = (e // group,) + hist[e]
        hist[e] = hist[e] + (a,)
        before.append(kb)
        after.append(kb + (a,))
    total, count = defaultdict(float), defaultdict(int)
    for e, r in enumerate(rewards.tolist()):
        node = (e // group,)
        total[node] += r; count[node] += 1
        for a in hist[e]:
            node = node + (a,)
            total[node] += r; count[node] += 1
    return np.array([total[ka] / count[ka] - total[kb] / count[kb] for kb, ka in zip(before, after)],
                    dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="checkpoints/chess_supervised.pt")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--puzzles", type=int, default=256, help="puzzles per iteration")
    ap.add_argument("--group", type=int, default=8, help="attempts per puzzle (1 = plain REINFORCE with a value baseline)")
    ap.add_argument("--min-rating", type=int, default=1300, help="only train on puzzles rated at least this")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (also used in the loss)")
    ap.add_argument("--c-value", type=float, default=0.5)
    ap.add_argument("--c-kl", type=float, default=0.05)
    ap.add_argument("--c-ent", type=float, default=0.001)
    ap.add_argument("--value-warmup", type=int, default=0,
                    help="first iterations train only the value head (needed for --group 1, where it is the baseline)")
    ap.add_argument("--chunk", type=int, default=512, help="positions per forward/backward chunk (memory bound)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-size", type=int, default=20000)
    ap.add_argument("--save", default="checkpoints/chess_rl.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pq.read_table(PUZZLES, columns=["FEN", "Moves", "Rating", "split"],
                       filters=[("split", "in", ["rl", "val"])]).to_pandas()
    pool = df[(df["split"] == "rl") & (df["Rating"] >= args.min_rating)].reset_index(drop=True)
    val = df[df["split"] == "val"].sample(n=args.val_size, random_state=args.seed)
    model, meta = load_checkpoint(args.init, dev)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    model.eval()   # BatchNorm statistics stay frozen; gradients still flow
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log = open(args.save + ".log.jsonl", "w")
    best = solve_rate(model, val, dev)
    print(f"RL pool {len(pool):,} puzzles (rating >= {args.min_rating}) | warm start val solve rate {best:.4f}", flush=True)
    log.write(json.dumps({"iter": 0, "val_solve_rate": best, "args": vars(args)}) + "\n")
    save_checkpoint(args.save, model, {**meta, "stage": "rl", "iter": 0, "val_solve_rate": best})
    t0, hist = time.time(), defaultdict(list)
    for it in range(1, args.iters + 1):
        batch = pool.iloc[rng.integers(len(pool), size=args.puzzles)]
        eps = [Episode.from_row(f, m) for f, m in zip(batch["FEN"], batch["Moves"]) for _ in range(args.group)]
        traj = play_puzzles(model, eps, dev, greedy=False, temperature=args.temperature, record=True)
        reward_np = np.array([float(e.solved) for e in eps], dtype=np.float32)
        reward = torch.from_numpy(reward_np).to(dev)
        R = reward[traj["episode"]]
        if args.group > 1:
            adv_all = torch.from_numpy(tree_advantage(traj["episode"].cpu().numpy(), traj["action"].cpu().numpy(),
                                                      reward_np, args.group)).to(dev)
            per_puzzle = reward_np.reshape(-1, args.group).mean(1)
            hist["mixed"].append(float(((per_puzzle > 0) & (per_puzzle < 1)).mean()))

        # one gradient step over all recorded positions, accumulated in chunks to bound GPU memory
        # (every chunk adds sum/N, so the gradient equals the full-batch mean)
        n = len(R)
        opt.zero_grad(set_to_none=True)
        stats = defaultdict(float)
        for s in range(0, n, args.chunk):
            sl = slice(s, s + args.chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                planes = planes_from_encoded(traj["enc"][sl])
                logits, v = model(planes, detach_value=True)
                with torch.no_grad():
                    ref_logits, _ = ref(planes)
            mask, r = traj["mask"][sl], R[sl]
            logits = logits.float().masked_fill(~mask, -1e9)
            logp = F.log_softmax(logits, dim=1)
            ref_logp = F.log_softmax(ref_logits.float().masked_fill(~mask, -1e9), dim=1)
            p = logp.exp()
            act_logp = F.log_softmax(logits / args.temperature, dim=1).gather(1, traj["action"][sl, None]).squeeze(1)
            adv = adv_all[sl] if args.group > 1 else r - torch.sigmoid(v.float()).detach()
            loss_pi = -(adv * act_logp).sum() / n
            loss_v = F.binary_cross_entropy_with_logits(v.float(), r, reduction="sum") / n
            kl = (p * (logp - ref_logp)).masked_fill(~mask, 0).sum() / n
            ent = -(p * logp).masked_fill(~mask, 0).sum() / n
            if it <= args.value_warmup:
                loss = loss_v                 # calibrate the baseline first; the policy is not changed yet
            else:
                loss = loss_pi + args.c_value * loss_v + args.c_kl * kl - args.c_ent * ent
            loss.backward()
            stats["loss_v"] += loss_v.item(); stats["kl"] += kl.item(); stats["ent"] += ent.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist["reward"].append(float(reward_np.mean()))
        for k, x in stats.items():
            hist[k].append(x)

        if it % args.eval_every == 0 or it == args.iters:
            rate = solve_rate(model, val, dev)
            w = args.eval_every
            rec = {"iter": it, "phase": "value-warmup" if it <= args.value_warmup else "policy-gradient",
                   "train_reward": round(float(np.mean(hist["reward"][-w:])), 4),
                   "value_loss": round(float(np.mean(hist["loss_v"][-w:])), 4),
                   "val_solve_rate": round(rate, 4), "kl": round(float(np.mean(hist["kl"][-w:])), 4),
                   "entropy": round(float(np.mean(hist["ent"][-w:])), 3),
                   "minutes": round((time.time() - t0) / 60, 1)}
            if hist["mixed"]:
                rec["mixed_puzzles"] = round(float(np.mean(hist["mixed"][-w:])), 3)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if rate > best:
                best = rate
                save_checkpoint(args.save, model, {**meta, "stage": "rl", "iter": it, "val_solve_rate": rate,
                                                   "rl_args": vars(args)})
    print(f"best validation solve rate {best:.4f} -> {args.save}")


if __name__ == "__main__":
    main()
