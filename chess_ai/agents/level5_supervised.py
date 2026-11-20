"""
chess_ai/agents/level5_supervised.py
Cấp 5: tác tử mạng nơ-ron học có giám sát (Dual-Head ResNet), ra quyết định
bằng Policy Head có áp Legal Action Masking (chọn nước hợp lệ có logit cao nhất).
"""
from typing import Optional

import chess
import numpy as np
import torch

from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.board_tensor import (
    action_index_to_move, board_to_tensor, get_legal_action_mask, move_to_action_index,
)
from chess_ai.core.chess_env import ChessEnv
from chess_ai.models.dual_resnet import DualHeadResNet
from chess_ai.models.io import load_model


class Level5SupervisedAgent(BaseAgent):
    """Tác tử Cấp 5: Supervised Dual-Head ResNet."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[DualHeadResNet] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(name="Level5_Supervised")
        self.checkpoint_path = checkpoint_path
        self.device = device
        if model is not None:
            self.model = model.to(device).eval()
        elif checkpoint_path:
            self.model = load_model(checkpoint_path, device)
        else:
            raise ValueError("Cần checkpoint_path hoặc model đã huấn luyện cho Cấp 5.")

    @torch.no_grad()
    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Chọn nước đi hợp lệ có xác suất Policy Head cao nhất (một lần suy luận mạng)."""
        board = env.board
        x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(self.device)
        logits, value = self.model(x)
        mask = torch.from_numpy(get_legal_action_mask(board)).to(self.device)
        masked = logits[0].masked_fill(mask == 0, float("-inf"))
        best_idx = int(masked.argmax())
        move = action_index_to_move(best_idx, board)
        self.last_search_stats = dict(nodes=1, value=float(value[0, 0]), net_evals=1)
        return move
