"""
chess_rl/play.py
Play against the trained agent in the terminal (moves in UCI, e.g. e2e4), or let it solve one puzzle.

  python -m chess_rl.play                       # you play White
  python -m chess_rl.play --black               # you play Black
  python -m chess_rl.play --fen "<FEN>"         # start from a position
"""

import argparse

import chess
import torch

from chess_rl.agent import policy_step
from chess_rl.model import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description="Play chess against the RL agent")
    ap.add_argument("--checkpoint", default="checkpoints/chess_rl.pt")
    ap.add_argument("--black", action="store_true", help="you play Black")
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.checkpoint, dev)
    board = chess.Board(args.fen)
    human = chess.BLACK if args.black else chess.WHITE
    print(f"Loaded {args.checkpoint} ({meta.get('stage')}). Type moves in UCI (e2e4), 'quit' to stop.")
    while not board.is_game_over(claim_draw=True):
        print(board, "\n")
        if board.turn == human:
            text = input("your move: ").strip()
            if text == "quit":
                return
            try:
                move = chess.Move.from_uci(text)
                if move not in board.legal_moves:
                    raise ValueError
            except ValueError:
                print("illegal move, try again")
                continue
        else:
            move = policy_step(model, [board], dev, greedy=True)[0][0]
            print(f"agent plays {board.san(move)}")
        board.push(move)
    print(board, "\nresult:", board.result(claim_draw=True))


if __name__ == "__main__":
    main()
