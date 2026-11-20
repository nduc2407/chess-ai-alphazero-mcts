"""
chess_ai/agents/level6_reinforcement.py
Cấp 6 [MỞ RỘNG]: Tác tử Học tăng cường (Tabular Q-Learning & Policy Gradient REINFORCE)
trên không gian trạng thái cờ tàn chuẩn FIDE KRK (Vua + Xe đấu Vua).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""

import pickle
import random
from typing import Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv
from chess_ai.training.train_rl_krk import krk_state_key


class Level6ReinforcementAgent(BaseAgent):
    """Tác tử Cấp 6: Reinforcement Learning trên cờ tàn KRK (chỉ dùng được ở thế KRK)."""

    def __init__(self, q_table_path: Optional[str] = None, table: Optional[dict] = None,
                 seed: int = 42) -> None:
        super().__init__(name="Level6_RL_KRK")
        self.q_table_path = q_table_path
        if table is None:
            if not q_table_path:
                raise ValueError("Cần q_table_path hoặc table đã huấn luyện cho Cấp 6.")
            with open(q_table_path, "rb") as f:
                table = pickle.load(f)["q"]
        self.q = table
        self.seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        super().reset()
        self._rng = random.Random(self.seed)

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi từ chính sách RL đã học; thế chưa gặp: ưu tiên chiếu hết rồi ngẫu nhiên."""
        board = env.board
        moves = list(board.legal_moves)
        key = krk_state_key(board)
        for m in moves:  # chiếu hết 1 nước luôn được chọn
            board.push(m); mate = board.is_checkmate(); board.pop()
            if mate:
                self.last_search_stats = dict(nodes=len(moves), table_hit=True)
                return m
        vals = [self.q.get((key, m.uci()), 0.0) for m in moves]
        best = max(vals)
        self.last_search_stats = dict(nodes=len(moves), table_hit=any(v != 0.0 for v in vals))
        return self._rng.choice([m for m, v in zip(moves, vals) if v == best])
