"""
chess_ai/training/replay_buffer.py
Bộ đệm kinh nghiệm vòng (Circular Experience Replay Buffer) lưu trữ các bộ ba
(state_tensor [18, 8, 8], mcts_policy [4096], outcome_z) sinh ra từ quá trình tự chơi (Self-play).
Phụ trách: TV2 (DL, ETL, PUCT & Self-Play Lead)
"""

from typing import Tuple
import numpy as np


class SelfPlayReplayBuffer:
    """Replay Buffer phục vụ huấn luyện lặp AlphaZero-inspired (Cấp 8)."""

    def __init__(self, capacity: int = 50000, seed: int = 42) -> None:
        self.capacity = capacity
        self.states = np.zeros((capacity, 18, 8, 8), dtype=np.float32)
        self.policies = np.zeros((capacity, 4096), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self._ptr = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def push_game_samples(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Thêm toàn bộ mẫu từ một ván tự chơi vào bộ đệm (ghi đè mẫu cũ nhất khi đầy)."""
        n = len(states)
        assert len(policies) == n and len(values) == n
        for i in range(n):
            self.states[self._ptr] = states[i]
            self.policies[self._ptr] = policies[i]
            self.values[self._ptr] = values[i]
            self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample_batch(self, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lấy mẫu ngẫu nhiên một batch (states, policies, values) để cập nhật mạng."""
        if self._size == 0:
            raise ValueError("Replay buffer rỗng")
        idx = self._rng.integers(0, self._size, size=min(batch_size, self._size))
        return self.states[idx].copy(), self.policies[idx].copy(), self.values[idx].copy()
