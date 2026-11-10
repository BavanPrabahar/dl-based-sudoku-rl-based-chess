"""
chess_rl/puzzles.py
Lichess puzzle database: preparation, train/val/test split and the reinforcement-learning environment.

Lichess puzzle format (https://database.lichess.org/#puzzles):
  FEN    position BEFORE the opponent's move
  Moves  UCI moves; Moves[0] is the opponent's move that creates the puzzle, then the solver's and the
         opponent's moves alternate, ending with a solver move. Like on Lichess, any move that gives
         checkmate is accepted as correct even if it is not the listed one.

Split: by a hash of PuzzleId (stable across dataset updates): 1% test, 1% validation, 98% train.
The train part is split again (also by hash) into a supervised pool (80%) and an RL pool (20%) so
the reinforcement-learning stage practises on puzzles the warm-start never saw.
"""

import zlib
from dataclasses import dataclass, field
from typing import List

import chess
import numpy as np

from chess_rl.encoding import encode_board, move_to_index


def split_of(puzzle_id: str) -> str:
    h = zlib.crc32(puzzle_id.encode()) % 1000
    if h < 10:
        return "test"
    if h < 20:
        return "val"
    return "rl" if h < 216 else "sup"     # 196/980 = 20% of train


def prepare_puzzles(src, out_path: str) -> int:
    """Read the Lichess puzzle parquet shards (directory) or CSV(.zst) and save a slim parquet with a split column."""
    import glob
    import os

    import pandas as pd
    cols = ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation", "Popularity", "NbPlays", "Themes"]
    if os.path.isdir(src):
        df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(os.path.join(src, "*.parquet")))],
                       ignore_index=True)
        df["Themes"] = df["Themes"].map(lambda t: " ".join(t) if t is not None else "")
    else:
        df = pd.read_csv(src, usecols=cols)
    df["split"] = df["PuzzleId"].map(split_of)
    df.to_parquet(out_path, index=False)
    return len(df)


@dataclass
class Episode:
    """One puzzle as an RL episode: the agent moves, the environment answers with the puzzle line."""
    board: chess.Board
    line: List[str]                 # full UCI line after the opponent's first move: agent, opp, agent, ...
    ptr: int = 0                    # index of the next agent move in `line`
    done: bool = False
    solved: bool = False
    first_correct: bool = None
    agent_moves: int = field(default=0)

    @classmethod
    def from_row(cls, fen: str, moves: str) -> "Episode":
        uci = moves.split()
        board = chess.Board(fen)
        board.push_uci(uci[0])
        return cls(board=board, line=uci[1:])

    def step(self, move: chess.Move) -> bool:
        """Apply the agent's move. Returns True if it was accepted (correct or checkmate)."""
        expected = self.line[self.ptr]
        self.board.push(move)
        correct = move.uci() == expected or self.board.is_checkmate()
        if self.first_correct is None:
            self.first_correct = correct
        self.agent_moves += 1
        if not correct:
            self.done = True
            return False
        if self.board.is_checkmate() or self.ptr + 1 >= len(self.line):
            self.done, self.solved = True, True
            return True
        self.board.push_uci(self.line[self.ptr + 1])    # opponent's reply
        self.ptr += 2
        return True


def supervised_samples(fen: str, moves: str):
    """All (encoded board, target move index) pairs of the solver's moves in one puzzle."""
    uci = moves.split()
    board = chess.Board(fen)
    board.push_uci(uci[0])
    out = []
    for k in range(1, len(uci)):
        move = chess.Move.from_uci(uci[k])
        if k % 2 == 1:   # solver's move
            out.append((encode_board(board), move_to_index(board, move)))
        board.push(move)
    return out


def _worker(rows):
    enc, idx = [], []
    for fen, moves in rows:
        for e, i in supervised_samples(fen, moves):
            enc.append(e)
            idx.append(i)
    return np.array(enc, dtype=np.int64).reshape(-1, 14), np.array(idx, dtype=np.int64)


def build_supervised_arrays(df, out_path: str, processes: int = 16, chunk: int = 20000):
    """Encode every solver move of the puzzles in `df` (multiprocessing) into an .npz (enc, target)."""
    from multiprocessing import Pool
    rows = list(zip(df["FEN"].tolist(), df["Moves"].tolist()))
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    with Pool(processes) as pool:
        parts = pool.map(_worker, chunks)
    enc = np.concatenate([p[0] for p in parts])
    tgt = np.concatenate([p[1] for p in parts])
    np.savez(out_path, enc=enc, target=tgt)
    return len(tgt)
