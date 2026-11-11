"""
Plot the game-RL training curve from checkpoints/chess_rl_games.pt.log.jsonl -> docs/game_rl_curve.png

  python scripts/plot_game_rl.py      (needs matplotlib)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = "checkpoints/chess_rl_games.pt.log.jsonl"
OUT = "docs/game_rl_curve.png"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

rows = [json.loads(line) for line in open(LOG)]
it = [r["iter"] for r in rows]
ev = [r["eval_score"] for r in rows]
tr_it = [r["iter"] for r in rows if "train_score" in r]
tr = [r["train_score"] for r in rows if "train_score" in r]
pz = [100 * r["val_puzzle_solve_rate"] for r in rows]

plt.rcParams.update({"font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.titlesize": 12})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
for a in ax:
    a.set_facecolor(SURFACE)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.spines[["top", "right"]].set_visible(False)
    a.set_xlabel("RL iteration (128 games each)")
    a.axvspan(0, 20, color=GRID, alpha=0.5, lw=0)

a = ax[0]
a.plot(tr_it, tr, color=BLUE, lw=2, marker="o", ms=5, label="training games (sampled moves, 3,200 games per point)")
a.plot(it, ev, color=ORANGE, lw=0, marker="o", ms=7, mec=SURFACE, mew=1.5,
       label="evaluation (greedy, 100 fresh games per point)")
a.axhline(ev[0], color=MUTED, lw=1, ls=(0, (4, 3)))
a.text(300, ev[0] - 0.012, "start", color=MUTED, ha="right", va="top", fontsize=10)
a.set_ylim(0.25, 0.62)
a.set_ylabel("score vs Stockfish Skill 0 (win 1, draw ½)")
a.set_title("Game score while training against Stockfish")
a.text(10, 0.605, "value-head\nwarm-up", color=MUTED, ha="center", va="top", fontsize=9)
a.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK2)

a = ax[1]
a.plot(it, pz, color=BLUE, lw=2, marker="o", ms=5)
a.set_ylim(74, 79)
a.set_ylabel("validation puzzles solved (%)")
a.set_title("Puzzle skill (5,000 validation puzzles)")

fig.tight_layout()
fig.savefig(OUT, dpi=120, facecolor=SURFACE)
print("->", OUT)
