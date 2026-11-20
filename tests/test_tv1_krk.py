"""Smoke test RL KRK (ít tập; chỉ kiểm tra mã, không phải kết quả thực nghiệm)."""
import chess
from chess_ai.agents.level6_reinforcement import Level6ReinforcementAgent
from chess_ai.core.chess_env import ChessEnv
from chess_ai.training.train_rl_krk import random_krk_board, train_krk_rl
import random


def test_krk_train_and_agent(tmp_path):
    out = train_krk_rl(episodes=40, output_qtable=str(tmp_path / "q.pkl"),
                       output_metrics=str(tmp_path / "c.json"), eval_every=20, eval_episodes=5)
    assert out["curve"] and (tmp_path / "q.pkl").exists()
    ag = Level6ReinforcementAgent(q_table_path=str(tmp_path / "q.pkl"))
    env = ChessEnv(random_krk_board(random.Random(1)).fen())
    assert ag.select_move(env) in env.board.legal_moves


def test_mate_in_one_is_played():
    ag = Level6ReinforcementAgent(table={})
    env = ChessEnv("6k1/8/6K1/8/8/8/8/R7 w - - 0 1")
    mv = ag.select_move(env)
    env.step(mv)
    assert env.board.is_checkmate()
