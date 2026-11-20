"""
chess_ai/agents/level8_alphazero.py
Cấp 8 [MỞ RỘNG]: Tác tử AlphaZero-inspired với vòng lặp tự chơi (Self-play Loop),
Replay Buffer và cơ chế đấu loại trực tiếp Checkpoint Arena Gating (ngưỡng thắng 55%).
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

from typing import Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.agents.level7_guided_mcts import Level7GuidedMCTSAgent
from chess_ai.core.chess_env import ChessEnv


class Level8AlphaZeroAgent(BaseAgent):
    """Tác tử Cấp 8: dùng champion sau gating (PUCT eval_mode) làm bộ ra quyết định."""

    def __init__(self, checkpoint_path: Optional[str] = None, simulations: int = 200,
                 model=None, device: str = "cpu") -> None:
        super().__init__(name="Level8_AlphaZero")
        self.checkpoint_path = checkpoint_path
        self.simulations = simulations
        self._inner = Level7GuidedMCTSAgent(checkpoint_path=checkpoint_path, model=model,
                                            simulations=simulations, eval_mode=True, device=device,
                                            name=self.name)

    def reset(self) -> None:
        super().reset()
        self._inner.reset()

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi bằng mô hình tốt nhất từ vòng lặp Self-play."""
        mv = self._inner.select_move(env, time_limit, max_workload)
        self.last_search_stats = dict(self._inner.last_search_stats)
        return mv
