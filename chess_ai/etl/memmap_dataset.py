"""
chess_ai/etl/memmap_dataset.py
PyTorch Dataset nạp trực tiếp từ các tệp nhị phân do pgn_etl.py ghi
({train,val,test}_{states,actions,values}.dat) bằng np.memmap, giữ RAM thấp.
"""
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from chess_ai.core.board_tensor import uint8_planes_to_tensor


class ChessMemmapDataset(Dataset):
    """Dataset ánh xạ bộ nhớ cho các tập train, val, test."""

    def __init__(self, data_dir: str, split: str = "train") -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        meta = json.loads((self.data_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
        self.n = int(meta["counts"][split])
        if self.n == 0:
            self.states = self.actions = self.values = None
            return
        self.states = np.memmap(self.data_dir / f"{split}_states.dat", dtype=np.uint8, mode="r", shape=(self.n, 18, 8, 8))
        self.actions = np.memmap(self.data_dir / f"{split}_actions.dat", dtype=np.int16, mode="r", shape=(self.n,))
        self.values = np.memmap(self.data_dir / f"{split}_values.dat", dtype=np.int8, mode="r", shape=(self.n,))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trả về (state_float32 [18, 8, 8], action_int64, value_float32)."""
        state = torch.from_numpy(uint8_planes_to_tensor(np.asarray(self.states[idx])))
        action = torch.tensor(int(self.actions[idx]), dtype=torch.int64)
        value = torch.tensor(float(self.values[idx]), dtype=torch.float32)
        return state, action, value
