"""Smoke test replay buffer, self-play, gating, Cấp 8 (mạng nhỏ ngẫu nhiên; chỉ kiểm tra mã)."""
import numpy as np
import chess

from chess_ai.agents.level8_alphazero import Level8AlphaZeroAgent
from chess_ai.core.chess_env import ChessEnv
from chess_ai.models.dual_resnet import DualHeadResNet
from chess_ai.training.replay_buffer import SelfPlayReplayBuffer
from chess_ai.training.selfplay import play_selfplay_game, run_selfplay_pipeline


def test_replay_buffer_circular():
    b = SelfPlayReplayBuffer(capacity=10)
    b.push_game_samples(np.ones((7, 18, 8, 8)), np.ones((7, 4096)), np.ones(7))
    b.push_game_samples(np.zeros((6, 18, 8, 8)), np.zeros((6, 4096)), np.zeros(6))
    assert len(b) == 10
    s, p, v = b.sample_batch(4)
    assert s.shape == (4, 18, 8, 8) and p.shape == (4, 4096) and v.shape == (4,)


def test_selfplay_game_shapes():
    net = DualHeadResNet(num_res_blocks=1, filters=16).eval()
    s, p, z, r = play_selfplay_game(net, simulations=4, max_plies=12)
    assert len(s) == len(p) == len(z) <= 12 and np.allclose(p.sum(1), 1, atol=1e-4)
    assert set(np.unique(z)) <= {-1.0, 0.0, 1.0}


def test_pipeline_and_agent(tmp_path):
    out = run_selfplay_pipeline(
        warm_start_checkpoint=str(tmp_path / "none.pt"), output_dir=str(tmp_path), iterations=1,
        device="cpu", champion_path=str(tmp_path / "az.pt"),
        model_config=dict(num_res_blocks=1, filters=16),
        overrides=dict(selfplay_games_per_iter=2, mcts_sims_per_move=4, arena_eval_games=2,
                       max_plies=10, train_steps_per_iter=2, batch_size=8, arena_win_threshold=0.0))
    assert out["iterations"][0]["promoted"] and (tmp_path / "az.pt").exists()
    ag = Level8AlphaZeroAgent(checkpoint_path=str(tmp_path / "az.pt"), simulations=5)
    env = ChessEnv()
    assert ag.select_move(env) in env.board.legal_moves
