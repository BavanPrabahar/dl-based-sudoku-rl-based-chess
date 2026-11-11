"""
chess_rl/train_games.py
Stage 3: reinforcement learning in full games against Stockfish at Skill Level 0.

Puzzles only ever show the agent short tactics from fixed positions, and puzzle RL ends up relearning the
solution moves the supervised stage already saw. In full games the agent meets positions that are the result
of its own earlier moves, which no dataset covers, so this is where reinforcement learning can teach it
something new (not hanging pieces, converting a won position, ...).

Every iteration:
  1. play --games games against Stockfish (Skill Level 0, --sf-ms ms per move; half as White, half as
     Black), the agent SAMPLING its moves; a game ends by the rules or as a draw after --max-plies plies;
  2. reward: the game score for the agent (win 1, draw 0.5, loss 0), plus potential-based shaping with the
     material balance Phi(s) = --c-material * (own material - opponent material) in pawns
     (Ng et al. 1999: shaping of this form does not change which policy is optimal; it only tells the agent
     sooner which move lost the piece);
  3. advantage with GAE(lambda) along each game, from U(s) = V(s) + Phi(s), U(end of game) = score:
     delta_t = U(s_{t+1}) - U(s_t),  A_t = sum_k lambda^k delta_{t+k};
  4. update  L = -A * log pi(a|s) + c_v * BCE(V(s), score) + c_kl * KL(pi || pi_start) - c_ent * H(pi).
The value head (expected game score) is trained on detached trunk features, so it never changes the policy;
the first --value-warmup iterations train only the value head, because before this stage it predicted
puzzle success, not game results.

Every --eval-every iterations the greedy policy plays --eval-games fresh games against the same opponent and
a sample of validation puzzles (to check that tactics are not forgotten).

Example:
  python -m chess_rl.train_games --init checkpoints/chess_rl.pt --stockfish /path/to/stockfish \
         --save checkpoints/chess_rl_games.pt
"""

import argparse
import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from chess_rl.agent import policy_step
from chess_rl.encoding import planes_from_encoded
from chess_rl.model import load_checkpoint, save_checkpoint
from chess_rl.train_rl import PUZZLES, solve_rate

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board, colour):
    return sum(v * (len(board.pieces(p, colour)) - len(board.pieces(p, not colour))) for p, v in VALUES.items())


def game_score(board, white):
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0 if white else 0.0
    if res == "0-1":
        return 0.0 if white else 1.0
    return 0.5          # draw by the rules, or stopped at --max-plies


class StockfishPool:
    """K Stockfish processes; one call moves a whole batch of boards, the engines working in parallel."""

    def __init__(self, path, k, skill, ms):
        self.engines = [chess.engine.SimpleEngine.popen_uci(path) for _ in range(k)]
        for e in self.engines:
            e.configure({"Skill Level": skill, "Threads": 1})
        self.limit = chess.engine.Limit(time=ms / 1000)
        self.pool = ThreadPoolExecutor(max_workers=k)

    def moves(self, boards):
        k = len(self.engines)
        groups = [boards[i::k] for i in range(k)]
        res = self.pool.map(lambda i: [self.engines[i].play(b, self.limit).move for b in groups[i]], range(k))
        out = [None] * len(boards)
        for i, ms in enumerate(res):
            out[i::k] = ms
        return out

    def close(self):
        for e in self.engines:
            e.quit()
        self.pool.shutdown()


def play_games(model, sf, n, device, greedy, temperature=1.0, max_plies=300, record=False):
    """n games vs Stockfish (agent White in even games). Returns scores (and per-move records)."""
    boards = [chess.Board() for _ in range(n)]
    colour = [chess.WHITE if g % 2 == 0 else chess.BLACK for g in range(n)]
    rec = {"enc": [], "mask": [], "action": [], "value": [], "game": [], "material": []}

    def running(g):
        return not boards[g].is_game_over(claim_draw=True) and boards[g].ply() < max_plies

    active = list(range(n))
    while active:
        sf_turn = [g for g in active if boards[g].turn != colour[g]]
        if sf_turn:
            for g, m in zip(sf_turn, sf.moves([boards[g] for g in sf_turn])):
                boards[g].push(m)
        active = [g for g in active if running(g)]
        ag = [g for g in active if boards[g].turn == colour[g]]
        if ag:
            moves, enc, mask, actions, value = policy_step(model, [boards[g] for g in ag], device, greedy, temperature)
            if record:
                rec["enc"].append(enc); rec["mask"].append(mask); rec["action"].append(actions)
                rec["value"].append(torch.sigmoid(value).cpu())
                rec["game"].extend(ag)
                rec["material"].extend(material(boards[g], colour[g]) for g in ag)
            for g, m in zip(ag, moves):
                boards[g].push(m)
        active = [g for g in active if running(g)]
    scores = np.array([game_score(boards[g], colour[g] == chess.WHITE) for g in range(n)], dtype=np.float32)
    if not record:
        return scores
    rec = {"enc": torch.cat(rec["enc"]), "mask": torch.cat(rec["mask"]), "action": torch.cat(rec["action"]),
           "value": torch.cat(rec["value"]).numpy(), "game": np.array(rec["game"]),
           "material": np.array(rec["material"], dtype=np.float32)}
    return scores, rec


def gae(rec, scores, c_material, lam):
    """Per-move advantages along each game (records are in move order within a game)."""
    U = rec["value"] + c_material * rec["material"]
    adv = np.zeros(len(U), dtype=np.float32)
    for g in np.unique(rec["game"]):
        idx = np.flatnonzero(rec["game"] == g)
        nxt = np.append(U[idx[1:]], scores[g])          # U after the move (and the reply); game end -> score
        delta = nxt - U[idx]
        run = 0.0
        for k in range(len(idx) - 1, -1, -1):
            run = delta[k] + lam * run
            adv[idx[k]] = run
    return adv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="checkpoints/chess_rl.pt")
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--skill", type=int, default=0)
    ap.add_argument("--sf-ms", type=float, default=10.0, help="Stockfish thinking time per move")
    ap.add_argument("--engines", type=int, default=12, help="Stockfish processes running in parallel")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--games", type=int, default=128, help="training games per iteration")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.9, help="GAE lambda")
    ap.add_argument("--c-material", type=float, default=0.02, help="shaping: expected-score units per pawn")
    ap.add_argument("--c-value", type=float, default=0.5)
    ap.add_argument("--c-kl", type=float, default=0.05)
    ap.add_argument("--c-ent", type=float, default=0.001)
    ap.add_argument("--value-warmup", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--val-puzzles", type=int, default=5000)
    ap.add_argument("--save", default="checkpoints/chess_rl_games.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val = pq.read_table(PUZZLES, columns=["FEN", "Moves", "split"], filters=[("split", "=", "val")]).to_pandas()
    val = val.sample(n=args.val_puzzles, random_state=args.seed)
    model, meta = load_checkpoint(args.init, dev)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    model.eval()   # BatchNorm statistics stay frozen; gradients still flow
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sf = StockfishPool(args.stockfish, args.engines, args.skill, args.sf_ms)
    log = open(args.save + ".log.jsonl", "w")

    def evaluate(it):
        s = play_games(model, sf, args.eval_games, dev, greedy=True, max_plies=args.max_plies)
        r = {"iter": it, "eval_score": round(float(s.mean()), 4), "eval_wins": int((s == 1).sum()),
             "eval_draws": int((s == 0.5).sum()), "eval_losses": int((s == 0).sum()),
             "val_puzzle_solve_rate": round(solve_rate(model, val, dev), 4)}
        return r

    r0 = evaluate(0)
    r0["args"] = vars(args)
    print(json.dumps(r0), flush=True)
    log.write(json.dumps(r0) + "\n"); log.flush()
    best = r0["eval_score"]
    save_checkpoint(args.save, model, {**meta, "stage": "rl-games", "iter": 0, "eval_score": best})
    t0, hist = time.time(), {"score": [], "plies": [], "kl": [], "loss_v": []}
    for it in range(1, args.iters + 1):
        model.eval()
        scores, rec = play_games(model, sf, args.games, dev, greedy=False, temperature=args.temperature,
                                 max_plies=args.max_plies, record=True)
        adv = torch.from_numpy(gae(rec, scores, args.c_material, args.lam)).to(dev)
        adv = adv / (adv.std() + 1e-6)
        target = torch.from_numpy(scores[rec["game"]]).to(dev)
        n = len(adv)
        opt.zero_grad(set_to_none=True)
        kl_sum = lv_sum = 0.0
        for s in range(0, n, args.chunk):
            sl = slice(s, s + args.chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                planes = planes_from_encoded(rec["enc"][sl])
                logits, v = model(planes, detach_value=True)
                with torch.no_grad():
                    ref_logits, _ = ref(planes)
            mask = rec["mask"][sl]
            logits = logits.float().masked_fill(~mask, -1e9)
            logp = F.log_softmax(logits, dim=1)
            ref_logp = F.log_softmax(ref_logits.float().masked_fill(~mask, -1e9), dim=1)
            p = logp.exp()
            act_logp = F.log_softmax(logits / args.temperature, dim=1).gather(1, rec["action"][sl, None]).squeeze(1)
            loss_pi = -(adv[sl] * act_logp).sum() / n
            loss_v = F.binary_cross_entropy_with_logits(v.float(), target[sl], reduction="sum") / n
            kl = (p * (logp - ref_logp)).masked_fill(~mask, 0).sum() / n
            ent = -(p * logp).masked_fill(~mask, 0).sum() / n
            if it <= args.value_warmup:
                loss = loss_v
            else:
                loss = loss_pi + args.c_value * loss_v + args.c_kl * kl - args.c_ent * ent
            loss.backward()
            kl_sum += kl.item(); lv_sum += loss_v.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist["score"].append(float(scores.mean())); hist["plies"].append(n / args.games)
        hist["kl"].append(kl_sum); hist["loss_v"].append(lv_sum)

        if it % args.eval_every == 0 or it == args.iters:
            w = args.eval_every
            r = evaluate(it)
            r.update({"phase": "value-warmup" if it <= args.value_warmup else "policy-gradient",
                      "train_score": round(float(np.mean(hist["score"][-w:])), 4),
                      "agent_moves_per_game": round(float(np.mean(hist["plies"][-w:])), 1),
                      "value_loss": round(float(np.mean(hist["loss_v"][-w:])), 4),
                      "kl": round(float(np.mean(hist["kl"][-w:])), 4),
                      "minutes": round((time.time() - t0) / 60, 1)})
            print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n"); log.flush()
            save_checkpoint(args.save.replace(".pt", "_last.pt"), model, {**meta, "stage": "rl-games", "iter": it})
            if r["eval_score"] > best:
                best = r["eval_score"]
                save_checkpoint(args.save, model, {**meta, "stage": "rl-games", "iter": it, "eval_score": best,
                                                   "val_puzzle_solve_rate": r["val_puzzle_solve_rate"],
                                                   "games_args": vars(args)})
    sf.close()
    print(f"best eval score {best:.3f} -> {args.save}")


if __name__ == "__main__":
    main()
