"""
chess_rl/agent.py
Batched move selection with the policy network, and batched puzzle play.
"""

from typing import List

import chess
import numpy as np
import torch

from chess_rl.encoding import NUM_ACTIONS, encode_board, index_to_move, legal_mask, planes_from_encoded


@torch.no_grad()
def policy_step(model, boards: List[chess.Board], device, greedy=True, temperature=1.0):
    """Choose one move per board. Returns (moves, encoded boards, masks, action indices, values)."""
    enc = torch.from_numpy(np.stack([encode_board(b) for b in boards])).to(device)
    mask = legal_mask(boards).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, value = model(planes_from_encoded(enc))
    logits = logits.float().masked_fill(~mask, float("-inf"))
    if greedy:
        actions = logits.argmax(dim=1)
    else:
        actions = torch.distributions.Categorical(logits=logits / temperature).sample()
    moves = [index_to_move(b, int(a)) for b, a in zip(boards, actions.tolist())]
    return moves, enc, mask, actions, value.float()


def play_puzzles(model, episodes, device, greedy=True, temperature=1.0, batch=2048, record=False):
    """Run every Episode to the end. With record=True also returns the per-move trajectory tensors."""
    traj = {"enc": [], "mask": [], "action": [], "episode": []}
    active = list(range(len(episodes)))
    while active:
        for start in range(0, len(active), batch):
            ids = active[start:start + batch]
            moves, enc, mask, actions, _ = policy_step(model, [episodes[i].board for i in ids], device, greedy, temperature)
            if record:
                traj["enc"].append(enc); traj["mask"].append(mask); traj["action"].append(actions)
                traj["episode"].append(torch.tensor(ids, device=device))
            for i, m in zip(ids, moves):
                episodes[i].step(m)
        active = [i for i in active if not episodes[i].done]
    if record:
        return {k: torch.cat(v) for k, v in traj.items()}
    return None


def random_move(board: chess.Board, rng) -> chess.Move:
    moves = list(board.legal_moves)
    return moves[rng.integers(len(moves))]


__all__ = ["policy_step", "play_puzzles", "random_move", "NUM_ACTIONS"]
