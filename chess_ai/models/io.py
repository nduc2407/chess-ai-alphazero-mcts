"""
chess_ai/models/io.py
Nạp/lưu checkpoint mô hình dùng chung cho Cấp 5, 7, 8 và các kịch bản huấn luyện.
Checkpoint là dict chứa tối thiểu: model_state_dict, model_config (xem configs/model_resnet.yaml).
"""
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from chess_ai.models.dual_resnet import DualHeadResNet


def build_model_from_config(model_config: Optional[Dict[str, Any]] = None) -> DualHeadResNet:
    return DualHeadResNet(**(model_config or {}))


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    Path(tmp).replace(path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model(path: str, device: str = "cpu") -> DualHeadResNet:
    """Dựng mạng theo model_config trong checkpoint và nạp trọng số; trả về mạng ở chế độ eval."""
    ckpt = load_checkpoint(path, map_location=device)
    model = build_model_from_config(ckpt.get("model_config"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model
