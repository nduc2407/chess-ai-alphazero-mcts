"""
chess_ai/experiments/eval_krk.py
Đánh giá Cấp 6 trên cờ tàn KRK: tỷ lệ chiếu hết và độ dài trung bình, so với Cấp 1 (ngẫu nhiên)
và Cấp 3 (Alpha-Beta) cùng bộ thế khởi đầu có seed. Đối thủ (Vua Đen) đi ngẫu nhiên.
"""
import json
import random
from pathlib import Path
from typing import Any, Dict

import chess

from chess_ai.agents.level1_random import Level1RandomAgent
from chess_ai.agents.level3_alphabeta import Level3AlphaBetaAgent
from chess_ai.agents.level6_reinforcement import Level6ReinforcementAgent
from chess_ai.core.chess_env import ChessEnv
from chess_ai.training.train_rl_krk import random_krk_board


def evaluate_krk_agent(agent, positions: int = 200, max_moves: int = 50, seed: int = 123,
                       **budget: Any) -> Dict[str, Any]:
    rng = random.Random(seed)
    wins, plies = 0, []
    for _ in range(positions):
        env = ChessEnv(random_krk_board(rng).fen())
        agent.reset()
        for _ in range(max_moves):
            env.step(agent.select_move(env, **budget))
            if env.is_terminal():
                break
            env.step(rng.choice(env.legal_actions()))
            if env.is_terminal():
                break
        if env.board.is_checkmate() and env.board.turn == chess.BLACK:
            wins += 1; plies.append(len(env.board.move_stack))
    return dict(agent=agent.name, positions=positions, mate_rate=wins / positions,
                mean_mate_plies=(sum(plies) / len(plies)) if plies else None)


def run_krk_evaluation(q_table_path: str = "checkpoints/l6_krk_qtable.pkl",
                       output_path: str = "results/training/krk_eval.json",
                       positions: int = 200, alphabeta_nodes: int = 2000) -> Dict[str, Any]:
    rows = [
        evaluate_krk_agent(Level1RandomAgent(seed=1), positions),
        evaluate_krk_agent(Level3AlphaBetaAgent(), positions, max_workload=alphabeta_nodes),
        evaluate_krk_agent(Level6ReinforcementAgent(q_table_path=q_table_path), positions),
    ]
    out = dict(positions=positions, results=rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    run_krk_evaluation()
