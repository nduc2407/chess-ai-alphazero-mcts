"""
chess_ai/models/dual_resnet.py
Dual-Head ResNet (thân 6x64, Policy Head 16 filters -> 4096,
Value Head 1 filter -> 64 -> 1 -> Tanh), đầu vào (B, 18, 8, 8) góc nhìn bên đi.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResBlock(nn.Module):
    def __init__(self, filters: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(filters)
        self.c2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class DualHeadResNet(nn.Module):
    """Mạng Dual-Head ResNet dẫn hướng cho Cấp 5, Cấp 7 và Cấp 8."""

    def __init__(
        self,
        in_channels: int = 18,
        num_res_blocks: int = 6,
        filters: int = 64,
        policy_conv_filters: int = 16,
        value_conv_filters: int = 1,
        value_hidden_units: int = 64,
        action_space: int = 4096,
    ) -> None:
        super().__init__()
        self.config = dict(
            in_channels=in_channels, num_res_blocks=num_res_blocks, filters=filters,
            policy_conv_filters=policy_conv_filters, value_conv_filters=value_conv_filters,
            value_hidden_units=value_hidden_units, action_space=action_space,
        )
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters), nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[_ResBlock(filters) for _ in range(num_res_blocks)])
        self.p_conv = nn.Conv2d(filters, policy_conv_filters, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(policy_conv_filters)
        self.p_fc = nn.Linear(policy_conv_filters * 64, action_space)
        self.v_conv = nn.Conv2d(filters, value_conv_filters, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(value_conv_filters)
        self.v_fc1 = nn.Linear(value_conv_filters * 64, value_hidden_units)
        self.v_fc2 = nn.Linear(value_hidden_units, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trả về (policy_logits [B, 4096], value [B, 1] trong [-1, 1])."""
        h = self.trunk(self.stem(x))
        p = F.relu(self.p_bn(self.p_conv(h))).flatten(1)
        policy_logits = self.p_fc(p)
        v = F.relu(self.v_bn(self.v_conv(h))).flatten(1)
        value = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v))))
        return policy_logits, value
