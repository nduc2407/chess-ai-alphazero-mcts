"""
chess_ai/agents/level7_guided_mcts.py
Cấp 7: AlphaGo-inspired Guided MCTS (PUCT) với Policy Prior và Value Head của mạng Dual-Head ResNet.
Hỗ trợ cờ hiệu use_prior / use_value để chạy ablation (prior đều, giá trị 0).
"""
import random
from typing import Optional

import chess

from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv
from chess_ai.core.zobrist import compute_zobrist_hash
from chess_ai.models.dual_resnet import DualHeadResNet
from chess_ai.models.io import load_model
from chess_ai.search.mcts_node import MCTSNode
from chess_ai.search.mcts_puct import run_puct_search


class _AblatedModel:
    """Bao mạng để tắt prior (logit bằng 0 -> prior đều) hoặc tắt value (giá trị 0) trong ablation."""

    def __init__(self, model, use_prior: bool, use_value: bool) -> None:
        self.model, self.use_prior, self.use_value = model, use_prior, use_value
        self.training = model.training

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval(); return self

    def train(self, mode=True):
        self.model.train(mode); return self

    def __call__(self, x):
        logits, value = self.model(x)
        if not self.use_prior:
            logits = logits * 0.0
        if not self.use_value:
            value = value * 0.0
        return logits, value


class Level7GuidedMCTSAgent(BaseAgent):
    """Tác tử Cấp 7: Guided MCTS (ResNet Prior + Value + PUCT)."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        simulations: int = 400,
        c_puct: float = 1.414,
        eval_mode: bool = True,
        model: Optional[DualHeadResNet] = None,
        use_prior: bool = True,
        use_value: bool = True,
        subtree_reuse: bool = True,
        seed: Optional[int] = 42,
        device: str = "cpu",
        name: str = "Level7_GuidedMCTS",
    ) -> None:
        super().__init__(name=name)
        self.checkpoint_path = checkpoint_path
        self.simulations = simulations
        self.c_puct = c_puct
        self.eval_mode = eval_mode
        self.subtree_reuse = subtree_reuse
        self.seed = seed
        if model is not None:
            net = model.to(device).eval()
        elif checkpoint_path:
            net = load_model(checkpoint_path, device)
        else:
            raise ValueError("Cần checkpoint_path hoặc model cho Cấp 7.")
        self.model = net if (use_prior and use_value) else _AblatedModel(net, use_prior, use_value)
        self._rng = random.Random(seed)
        self._root: Optional[MCTSNode] = None

    def reset(self) -> None:
        super().reset()
        self._root = None
        self._rng = random.Random(self.seed)

    def select_move(
        self,
        env: ChessEnv,
        time_limit: Optional[float] = None,
        max_workload: Optional[int] = None,
    ) -> chess.Move:
        """Thực thi tìm kiếm PUCT dẫn hướng bởi mạng Dual-Head ResNet."""
        sims = max_workload if max_workload is not None else (100_000 if time_limit is not None else self.simulations)
        reusable: Optional[MCTSNode] = None
        if self.subtree_reuse and self._root is not None:
            reusable = self._root.find_reusable_root(compute_zobrist_hash(env.board))
        move, stats, root = run_puct_search(
            env.board, self.model, simulations=sims, time_limit=time_limit, c_puct=self.c_puct,
            eval_mode=self.eval_mode, root=reusable, rng=self._rng,
        )
        self._root = root if self.subtree_reuse else None
        self.last_search_stats = {k: v for k, v in stats.items() if k != "pi"}
        return move
