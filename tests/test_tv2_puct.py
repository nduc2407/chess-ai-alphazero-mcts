"""Smoke test PUCT / Cấp 7 với mạng khởi tạo ngẫu nhiên (chỉ kiểm tra mã, không phải kết quả thực nghiệm)."""
import chess
import numpy as np

from chess_ai.agents.level7_guided_mcts import Level7GuidedMCTSAgent
from chess_ai.core.chess_env import ChessEnv
from chess_ai.models.dual_resnet import DualHeadResNet
from chess_ai.search.mcts_puct import search_puct


def _net():
    return DualHeadResNet(num_res_blocks=1, filters=16).eval()


def test_search_puct_returns_legal_move_and_pi():
    b = chess.Board()
    mv, st = search_puct(b, _net(), simulations=20)
    assert mv in b.legal_moves
    assert abs(st["pi"].sum() - 1.0) < 1e-4 and st["simulations"] == 20


def test_selfplay_mode_and_mate_in_one():
    b = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    mv, st = search_puct(b, _net(), simulations=60, eval_mode=False)
    assert mv in b.legal_moves


def test_agent_plays_with_subtree_reuse():
    ag = Level7GuidedMCTSAgent(model=_net(), simulations=15)
    env = ChessEnv()
    for _ in range(4):
        mv = ag.select_move(env)
        assert mv in env.board.legal_moves
        env.step(mv)
    assert ag.last_search_stats["net_evals"] >= 1


def test_ablation_flags():
    ag = Level7GuidedMCTSAgent(model=_net(), simulations=10, use_prior=False, use_value=False)
    assert ag.select_move(ChessEnv()) in chess.Board().legal_moves
