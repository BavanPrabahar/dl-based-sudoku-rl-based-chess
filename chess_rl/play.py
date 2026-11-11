"""
chess_rl/play.py
Play against the trained agent in the terminal, or watch it play itself.

  python -m chess_rl.play                       # you play White
  python -m chess_rl.play --black               # you play Black (board shown from Black's side)
  python -m chess_rl.play --fen "<FEN>"         # start from a position
  python -m chess_rl.play --selfplay            # the agent plays both sides

Moves can be typed in standard notation (e4, Nf3, O-O, exd5, e8=Q) or UCI (e2e4, e7e8q).
Commands: undo (take back your last move), hint (the agent's suggestion for you), quit.
"""

import argparse

import chess
import torch

from chess_rl.agent import policy_step
from chess_rl.model import load_checkpoint

PIECES = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
          "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}


def show(board, orientation=chess.WHITE, ascii_only=False):
    """Board with rank/file labels, from `orientation`'s side; the last move's squares are marked with ()."""
    last = board.peek() if board.move_stack else None
    marked = {last.from_square, last.to_square} if last else set()
    ranks = range(7, -1, -1) if orientation == chess.WHITE else range(8)
    files = range(8) if orientation == chess.WHITE else range(7, -1, -1)
    lines = []
    for r in ranks:
        cells = []
        for f in files:
            sq = chess.square(f, r)
            piece = board.piece_at(sq)
            ch = "." if piece is None else (piece.symbol() if ascii_only else PIECES[piece.symbol()])
            cells.append(f"({ch})" if sq in marked else f" {ch} ")
        lines.append(f" {r + 1} " + "".join(cells))
    lines.append("   " + "".join(f" {chess.FILE_NAMES[f]} " for f in files))
    return "\n".join(lines)


def parse_move(board, text):
    """Standard notation first (e4, Nf3, O-O), then UCI (e2e4)."""
    try:
        return board.parse_san(text)
    except ValueError:
        pass
    try:
        move = chess.Move.from_uci(text)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def agent_move(model, board, dev):
    return policy_step(model, [board], dev, greedy=True)[0][0]


def main():
    ap = argparse.ArgumentParser(description="Play chess against the RL agent")
    ap.add_argument("--checkpoint", default="checkpoints/chess_rl_games.pt")
    ap.add_argument("--black", action="store_true", help="you play Black")
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    ap.add_argument("--selfplay", action="store_true", help="the agent plays both sides")
    ap.add_argument("--max-plies", type=int, default=300, help="self-play: stop after this many plies")
    ap.add_argument("--ascii", action="store_true", help="letters instead of chess symbols")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(args.checkpoint, dev)
    board = chess.Board(args.fen)
    human = None if args.selfplay else (chess.BLACK if args.black else chess.WHITE)
    view = chess.BLACK if human == chess.BLACK else chess.WHITE
    print(f"Loaded {args.checkpoint} ({meta.get('stage')}).")
    if human is not None:
        print("Type moves like e4, Nf3, O-O or e2e4.  Commands: undo, hint, quit.")

    while not board.is_game_over(claim_draw=True):
        if args.selfplay and len(board.move_stack) >= args.max_plies:
            print(f"stopped after {args.max_plies} plies")
            break
        print("\n" + show(board, view, args.ascii) + "\n")
        if board.turn == human:
            if board.is_check():
                print("check!")
            try:
                text = input("your move: ").strip()
            except EOFError:
                return
            if text == "quit":
                return
            if text == "undo":
                if len(board.move_stack) >= 2:
                    board.pop(); board.pop()
                else:
                    print("nothing to take back")
                continue
            if text == "hint":
                print("agent suggests:", board.san(agent_move(model, board, dev)))
                continue
            move = parse_move(board, text)
            if move is None:
                print("illegal or unreadable move, try again (e.g. e4, Nf3, O-O, e2e4)")
                continue
        else:
            move = agent_move(model, board, dev)
            side = "White" if board.turn == chess.WHITE else "Black"
            print(f"agent ({side}) plays {board.san(move)}")
        board.push(move)

    print("\n" + show(board, view, args.ascii))
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        reason = outcome.termination.name.replace("_", " ").lower()
        print(f"\nresult: {board.result(claim_draw=True)} ({reason})")


if __name__ == "__main__":
    main()
