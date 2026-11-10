"""
chess_rl/encoding.py
Board and move encodings for the chess policy/value network.

Every position is seen from the side to move: if Black is to move, the board is mirrored
(ranks flipped, colours swapped) so the network always plays "up the board" as White.

Board  -> 12 piece bitboards (own P N B R Q K, opponent P N B R Q K) + castling bits + en-passant square,
          stored compactly (int64 bitboards) and expanded to 17 input planes of 8x8 on the GPU:
          12 piece planes, 4 castling planes (own K/Q side, opponent K/Q side), 1 en-passant plane.
Move   -> one of NUM_ACTIONS = 4096 + 72 indices (in the mirrored frame):
          from*64 + to                                  for every move (promotions to a queen included),
          4096 + (from_file*3 + (to_file-from_file+1))*3 + k   for under-promotions to knight/bishop/rook (k=0,1,2).
"""

import chess
import numpy as np
import torch

NUM_ACTIONS = 4096 + 72
NUM_PLANES = 17
_PIECES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_UNDER = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
_UNDER_INV = {v: k for k, v in _UNDER.items()}


def oriented(board: chess.Board) -> chess.Board:
    """The position from the side to move's point of view (White to move)."""
    return board if board.turn == chess.WHITE else board.mirror()


def encode_board(board: chess.Board) -> np.ndarray:
    """-> int64[14]: 12 piece bitboards, castling bits (own K, own Q, opp K, opp Q), ep square (-1 = none)."""
    b = oriented(board)
    out = np.empty(14, dtype=np.int64)
    for i, colour in enumerate((chess.WHITE, chess.BLACK)):
        for j, pt in enumerate(_PIECES):
            out[i * 6 + j] = np.int64(np.uint64(b.pieces_mask(pt, colour)).view(np.int64))
    out[12] = (int(b.has_kingside_castling_rights(chess.WHITE)) | int(b.has_queenside_castling_rights(chess.WHITE)) << 1
               | int(b.has_kingside_castling_rights(chess.BLACK)) << 2 | int(b.has_queenside_castling_rights(chess.BLACK)) << 3)
    out[13] = b.ep_square if b.ep_square is not None and b.has_legal_en_passant() else -1
    return out


_SHIFTS = torch.arange(64)


def planes_from_encoded(enc: torch.Tensor) -> torch.Tensor:
    """(B, 14) int64 -> (B, 17, 8, 8) float planes (square index = rank*8 + file)."""
    shifts = _SHIFTS.to(enc.device)
    bits = (enc[:, :12, None] >> shifts) & 1                                   # (B, 12, 64)
    castle = ((enc[:, 12, None] >> torch.arange(4, device=enc.device)) & 1)   # (B, 4)
    castle = castle[:, :, None].expand(-1, 4, 64)
    ep = torch.zeros(enc.shape[0], 1, 64, device=enc.device, dtype=bits.dtype)
    has_ep = enc[:, 13] >= 0
    ep[has_ep, 0, enc[has_ep, 13]] = 1
    return torch.cat([bits, castle, ep], dim=1).float().view(-1, NUM_PLANES, 8, 8)


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    """Index of `move` (legal in `board`) in the side-to-move frame."""
    if board.turn == chess.BLACK:
        frm, to = chess.square_mirror(move.from_square), chess.square_mirror(move.to_square)
    else:
        frm, to = move.from_square, move.to_square
    if move.promotion and move.promotion != chess.QUEEN:
        ff, tf = chess.square_file(frm), chess.square_file(to)
        return 4096 + (ff * 3 + (tf - ff + 1)) * 3 + _UNDER[move.promotion]
    return frm * 64 + to


def index_to_move(board: chess.Board, index: int) -> chess.Move:
    """Inverse of move_to_index for the given board (queen promotion assumed for plain from-to pawn moves)."""
    if index >= 4096:
        k = index - 4096
        piece = _UNDER_INV[k % 3]
        ff, d = divmod(k // 3, 3)
        frm = chess.square(ff, 6)
        to = chess.square(ff + d - 1, 7)
    else:
        frm, to = divmod(index, 64)
        piece = None
    if board.turn == chess.BLACK:
        frm, to = chess.square_mirror(frm), chess.square_mirror(to)
    if piece is None:
        p = board.piece_at(frm)
        if p is not None and p.piece_type == chess.PAWN and chess.square_rank(to) in (0, 7):
            piece = chess.QUEEN
    return chess.Move(frm, to, promotion=piece)


def legal_indices(board: chess.Board) -> np.ndarray:
    return np.fromiter((move_to_index(board, m) for m in board.legal_moves), dtype=np.int64)


def legal_mask(boards) -> torch.Tensor:
    """(B, NUM_ACTIONS) bool mask of legal moves for a list of boards."""
    mask = torch.zeros(len(boards), NUM_ACTIONS, dtype=torch.bool)
    for i, b in enumerate(boards):
        mask[i, torch.from_numpy(legal_indices(b))] = True
    return mask


if __name__ == "__main__":
    # Round-trip self-test on random games: every legal move maps to a unique index and back.
    import random
    random.seed(0)
    n_moves = 0
    for _ in range(200):
        b = chess.Board()
        for _ in range(random.randint(0, 120)):
            moves = list(b.legal_moves)
            if not moves:
                break
            idx = [move_to_index(b, m) for m in moves]
            assert len(set(idx)) == len(idx), "index collision"
            for m, i in zip(moves, idx):
                assert index_to_move(b, i) == m, (b.fen(), m, i)
            n_moves += len(moves)
            b.push(random.choice(moves))
    # promotion corner cases
    b = chess.Board("8/P6k/8/8/8/8/6Kp/8 w - - 0 1")
    for m in b.legal_moves:
        assert index_to_move(b, move_to_index(b, m)) == m
    b.turn = chess.BLACK
    for m in b.legal_moves:
        assert index_to_move(b, move_to_index(b, m)) == m
    print(f"encoding self-test passed on {n_moves:,} legal moves")
