#!/usr/bin/env python3
"""
Download the datasets used in this repository and convert them for training.

Sudoku: Sudoku-Extreme (Hugging Face `sapientinc/sudoku-extreme`): ~3.8M training and ~423k held-out
        test puzzles with unique solutions, collected from the Sudoku community (Kaggle 1M, unbiased
        random puzzles, 17-clue puzzles, forum "hardest" lists). `rating` = backtracks a solver needs.
Chess:  Lichess puzzle database (https://database.lichess.org/#puzzles, CC0, Hugging Face mirror
        Lichess/chess-puzzles): ~6.2M rated puzzles taken
        from real games, each with the position, the forced solution line and a Glicko-2 rating.

Usage:  python scripts/download_data.py [--only sudoku|chess]
"""
import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FILES = {
    "sudoku": [
        ("https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/train.csv", "sudoku_extreme_train.csv"),
        ("https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/test.csv", "sudoku_extreme_test.csv"),
    ],
    # Lichess publishes the same database on Hugging Face (Lichess/chess-puzzles, parquet, updated monthly);
    # it downloads much faster than database.lichess.org/lichess_db_puzzle.csv.zst (also supported by prepare_puzzles).
    "chess": [
        (f"https://huggingface.co/datasets/Lichess/chess-puzzles/resolve/main/data/train-0000{i}-of-00003.parquet",
         f"lichess_puzzles/train-0000{i}-of-00003.parquet") for i in range(3)
    ],
}


def download(url, path):
    if os.path.exists(path):
        print(f"  have {path}")
        return
    print(f"  downloading {url}")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["sudoku", "chess"])
    args = ap.parse_args()
    raw, processed = os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "processed")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(processed, exist_ok=True)

    if args.only in (None, "sudoku"):
        print("Sudoku-Extreme")
        from sudoku.dataset import prepare_sudoku_extreme
        for url, name in FILES["sudoku"]:
            download(url, os.path.join(raw, name))
            out = os.path.join(processed, name.replace(".csv", ".npz"))
            if not os.path.exists(out):
                d = prepare_sudoku_extreme(os.path.join(raw, name), out)
                print(f"  {out}: {len(d['puzzles']):,} puzzles")

    if args.only in (None, "chess"):
        print("Lichess puzzles")
        from chess_rl.puzzles import prepare_puzzles
        os.makedirs(os.path.join(raw, "lichess_puzzles"), exist_ok=True)
        for url, name in FILES["chess"]:
            download(url, os.path.join(raw, name))
        out = os.path.join(processed, "lichess_puzzles.parquet")
        if not os.path.exists(out):
            n = prepare_puzzles(os.path.join(raw, "lichess_puzzles"), out)
            print(f"  {out}: {n:,} puzzles")


if __name__ == "__main__":
    main()
