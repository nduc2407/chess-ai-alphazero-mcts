"""
chess_ai/models/loss.py
Hàm mất mát đa nhiệm: Cross-Entropy (Policy) + MSE (Value).
L2 do weight_decay của AdamW đảm nhiệm, không cộng thêm vào loss.

Hỗ trợ hai dạng nhãn policy:
- target_actions có shape (B,) kiểu int  -> nhãn cứng (huấn luyện có giám sát, Cấp 5).
- target_actions có shape (B, 4096) float -> nhãn mềm là phân phối pi (tự chơi, Cấp 8).
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHeadLoss(nn.Module):
    """Hàm mất mát kết hợp cho Dual-Head ResNet."""

    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0) -> None:
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight

    def forward(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        target_actions: torch.Tensor,
        target_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Trả về (total_loss, {'policy': ..., 'value': ..., 'total': ...})."""
        if target_actions.dtype in (torch.int64, torch.int32) and target_actions.dim() == 1:
            policy_loss = F.cross_entropy(policy_logits, target_actions.long())
        else:
            log_p = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(target_actions * log_p).sum(dim=1).mean()
        value_loss = F.mse_loss(value_pred.view(-1), target_values.view(-1).float())
        total = self.policy_weight * policy_loss + self.value_weight * value_loss
        return total, {
            "policy": float(policy_loss.detach()),
            "value": float(value_loss.detach()),
            "total": float(total.detach()),
        }
