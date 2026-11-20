"""Kiểm thử board_tensor, mạng Dual-Head ResNet và loss (Cấp 5-8)."""
import random

import chess
import numpy as np
import torch

from chess_ai.core.board_tensor import (
    ACTION_SPACE, action_index_to_move, board_to_tensor, board_to_uint8_planes,
    canonical_square, get_legal_action_mask, move_to_action_index, uint8_planes_to_tensor,
)
from chess_ai.models.dual_resnet import DualHeadResNet
from chess_ai.models.loss import DualHeadLoss


def _random_positions(n=60, seed=0):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(0, 60)):
            if b.is_game_over():
                break
            b.push(rng.choice(list(b.legal_moves)))
        if not b.is_game_over():
            out.append(b)
    return out


def test_shapes_and_dtypes():
    b = chess.Board()
    t = board_to_tensor(b)
    assert t.shape == (18, 8, 8) and t.dtype == np.float32
    u = board_to_uint8_planes(b)
    assert u.shape == (18, 8, 8) and u.dtype == np.uint8
    assert np.allclose(uint8_planes_to_tensor(u), t)


def test_start_position_planes():
    t = board_to_tensor(chess.Board())
    assert t[0].sum() == 8 and t[5].sum() == 1 and t[6].sum() == 8
    assert t[12:16].min() == 1  # đủ 4 quyền nhập thành
    assert t[16].sum() == 0


def test_color_flip_invariance():
    """Thế gương với đổi màu cho cùng tensor."""
    for b in _random_positions(40, seed=1):
        m = b.mirror()
        assert np.array_equal(board_to_tensor(b), board_to_tensor(m))


def test_roundtrip_all_legal_moves():
    for b in _random_positions(60, seed=2):
        for mv in b.legal_moves:
            idx = move_to_action_index(mv, b.turn)
            dec = action_index_to_move(idx, b)
            if mv.promotion in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
                assert (dec.from_square, dec.to_square) == (mv.from_square, mv.to_square)
            else:
                assert dec == mv


def test_legal_mask_matches_legal_moves():
    for b in _random_positions(40, seed=3):
        mask = get_legal_action_mask(b)
        assert mask.shape == (ACTION_SPACE,)
        idxs = {move_to_action_index(m, b.turn) for m in b.legal_moves}
        assert set(np.nonzero(mask)[0].tolist()) == idxs


def test_en_passant_castling_promotion():
    b = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    assert board_to_tensor(b)[16].sum() == 1
    ep = chess.Move.from_uci("e5f6")
    assert action_index_to_move(move_to_action_index(ep, b.turn), b) == ep
    c = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    for u in ("e1g1", "e1c1"):
        mv = chess.Move.from_uci(u)
        assert action_index_to_move(move_to_action_index(mv, c.turn), c) == mv
    p = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
    q = chess.Move.from_uci("a7a8q")
    assert action_index_to_move(move_to_action_index(q, p.turn), p) == q


def test_canonical_square_involution():
    for sq in range(64):
        assert canonical_square(canonical_square(sq, chess.BLACK), chess.BLACK) == sq
        assert canonical_square(sq, chess.WHITE) == sq


def test_model_forward_shapes_and_range():
    net = DualHeadResNet()
    x = torch.from_numpy(np.stack([board_to_tensor(b) for b in _random_positions(4)]))
    net.eval()
    p, v = net(x)
    assert p.shape == (4, 4096) and v.shape == (4, 1)
    assert float(v.abs().max()) <= 1.0


def test_loss_hard_and_soft_targets():
    crit = DualHeadLoss()
    logits = torch.randn(5, 4096, requires_grad=True)
    value = torch.tanh(torch.randn(5, 1, requires_grad=True))
    z = torch.tensor([1.0, -1.0, 0.0, 1.0, 0.0])
    hard = torch.randint(0, 4096, (5,))
    l1, d1 = crit(logits, value, hard, z)
    soft = torch.softmax(torch.randn(5, 4096), dim=1)
    l2, d2 = crit(logits, value, soft, z)
    l1.backward(retain_graph=True)
    l2.backward()
    assert set(d1) == {"policy", "value", "total"} and d2["total"] > 0


def test_one_overfit_step_reduces_loss():
    torch.manual_seed(0)
    net = DualHeadResNet(num_res_blocks=1, filters=16)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    boards = _random_positions(16, seed=4)
    x = torch.from_numpy(np.stack([board_to_tensor(b) for b in boards]))
    y = torch.tensor([move_to_action_index(next(iter(b.legal_moves)), b.turn) for b in boards])
    z = torch.zeros(16)
    crit = DualHeadLoss()
    first = None
    for _ in range(30):
        p, v = net(x)
        loss, _ = crit(p, v, y, z)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < first
