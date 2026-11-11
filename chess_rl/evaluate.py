"""
chess_rl/evaluate.py
Evaluate chess policies on the held-out TEST puzzles (never used for training or model selection) and in games.

Puzzle metrics (greedy policy, no search):
  first_move_acc  the first solver move is the Lichess solution (or mates)
  solve_rate      the whole solution line is played correctly ("puzzle accuracy" on Lichess)
  by rating bucket, and an estimated puzzle rating: the player rating R that maximises the likelihood
  of the observed successes under the Elo model P(solve) = 1 / (1 + 10^((puzzle_rating - R) / 400)).
Games: the greedy policy against a uniformly random mover, and optionally against Stockfish (--stockfish PATH)
at each Skill Level in --skills, 10 ms per move. Colours alternate, a game is a draw after 300 plies.

Example:
  python -m chess_rl.evaluate --models random checkpoints/chess_supervised.pt checkpoints/chess_rl.pt
"""

import argparse
import json
import os

import chess
import numpy as np
import pandas as pd
import torch

from chess_rl.agent import play_puzzles, policy_step, random_move
from chess_rl.model import load_checkpoint
from chess_rl.puzzles import Episode

BUCKETS = [(0, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 4000)]


class RandomPolicy:
    """Stand-in 'model' that plays uniformly random legal moves (lower baseline)."""


def run_puzzles_random(episodes, rng):
    active = list(range(len(episodes)))
    while active:
        for i in active:
            episodes[i].step(random_move(episodes[i].board, rng))
        active = [i for i in active if not episodes[i].done]


def puzzle_rating(ratings, solved):
    grid = np.arange(400, 3200, 5)
    ll = [np.sum(np.where(solved, -np.log1p(10 ** ((ratings - r) / 400)), -np.log1p(10 ** ((r - ratings) / 400))))
          for r in grid]
    return int(grid[int(np.argmax(ll))])


def evaluate_puzzles(name, model, df, device, rng):
    eps = [Episode.from_row(f, m) for f, m in zip(df["FEN"], df["Moves"])]
    if model is None:
        run_puzzles_random(eps, rng)
    else:
        play_puzzles(model, eps, device, greedy=True)
    solved = np.array([e.solved for e in eps])
    first = np.array([bool(e.first_correct) for e in eps])
    ratings = df["Rating"].to_numpy()
    res = {"model": name, "puzzles": len(eps), "first_move_acc": float(first.mean()),
           "solve_rate": float(solved.mean()), "estimated_puzzle_rating": puzzle_rating(ratings, solved),
           "by_rating": {}}
    for lo, hi in BUCKETS:
        m = (ratings >= lo) & (ratings < hi)
        if m.any():
            res["by_rating"][f"{lo}-{hi}"] = {"n": int(m.sum()), "first_move_acc": float(first[m].mean()),
                                              "solve_rate": float(solved[m].mean())}
    return res


def play_game(model, device, white: bool, opponent, max_plies=300):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if (board.turn == chess.WHITE) == white:
            move = policy_step(model, [board], device, greedy=True)[0][0]
        else:
            move = opponent(board)
        board.push(move)
    result = board.result(claim_draw=True)
    if result == "*":
        return 0.5
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result] if white else {"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}[result]


def evaluate_games(model, device, opponent, n_games, label):
    scores = [play_game(model, device, white=(g % 2 == 0), opponent=opponent) for g in range(n_games)]
    wins = sum(s == 1.0 for s in scores)
    draws = sum(s == 0.5 for s in scores)
    return {"opponent": label, "games": n_games, "wins": wins, "draws": draws, "losses": n_games - wins - draws,
            "score": float(np.mean(scores))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["random", "checkpoints/chess_supervised.pt", "checkpoints/chess_rl.pt"])
    ap.add_argument("--puzzles", default="data/processed/lichess_puzzles.parquet")
    ap.add_argument("--max-puzzles", type=int, default=0, help="0 = the whole test split")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--stockfish", default=None, help="path to a Stockfish binary for games against Stockfish")
    ap.add_argument("--skills", type=int, nargs="+", default=[0], help="Stockfish Skill Levels to play against")
    ap.add_argument("--engines", type=int, default=12, help="Stockfish processes (games are played in parallel)")
    ap.add_argument("--out", default="results/chess_results.json")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    df = pd.read_parquet(args.puzzles, columns=["PuzzleId", "FEN", "Moves", "Rating", "split"])
    test = df[df["split"] == "test"].reset_index(drop=True)
    if args.max_puzzles:
        test = test.sample(n=args.max_puzzles, random_state=0).reset_index(drop=True)
    out = {"test_puzzles": len(test), "puzzles": [], "games": []}
    pools = {}
    if args.stockfish:
        from chess_rl.train_games import StockfishPool
        pools = {k: StockfishPool(args.stockfish, args.engines, k, 10.0) for k in args.skills}
    for spec in args.models:
        if spec == "random":
            res = evaluate_puzzles("random legal moves", None, test, dev, rng)
            out["puzzles"].append(res)
        else:
            model, meta = load_checkpoint(spec, dev)
            name = f"{os.path.basename(spec)} ({meta.get('stage', '?')})"
            res = evaluate_puzzles(name, model, test, dev, rng)
            out["puzzles"].append(res)
            g = evaluate_games(model, dev, lambda b: random_move(b, rng), args.games, "random mover")
            g["model"] = name
            out["games"].append(g)
            for skill, pool in pools.items():
                from chess_rl.train_games import play_games
                s = play_games(model, pool, args.games, dev, greedy=True, max_plies=300)
                out["games"].append({"opponent": f"Stockfish Skill Level {skill}, 10 ms/move", "games": args.games,
                                     "wins": int((s == 1).sum()), "draws": int((s == 0.5).sum()),
                                     "losses": int((s == 0).sum()), "score": float(s.mean()), "model": name})
                print(json.dumps(out["games"][-1]), flush=True)
        print(json.dumps(out["puzzles"][-1]), flush=True)
    for pool in pools.values():
        pool.close()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    for g in out["games"]:
        print(json.dumps(g))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
