"""
chess_ai/arena/metrics.py
Module tính toán thống kê khoa học:
- Điểm số chuẩn hóa S = (W + 0.5 * D) / N.
- Khoảng tin cậy Wilson Score 95%.
- Khoảng tin cậy Paired-Opening Bootstrap 95% (10,000 resamples).
- Độ trễ p95 (ms), tốc độ NPS / SPS, và các chỉ số phong cách thi đấu (RQ4).
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

import math
import random
from typing import Sequence, Tuple


def compute_normalized_score(wins: int, draws: int, losses: int) -> float:
    """Tính tỷ lệ điểm chuẩn hóa S = (W + 0.5 * D) / (W + D + L)."""
    n = wins + draws + losses
    if n <= 0:
        raise ValueError("Tổng số ván phải > 0")
    return (wins + 0.5 * draws) / n


def wilson_confidence_interval(
    score: float,
    total_games: int,
    z: float = 1.96,
) -> Tuple[float, float]:
    """Tính khoảng tin cậy Wilson Score 95% [ci_low, ci_high] trong [0.0, 1.0]."""
    if total_games <= 0:
        return (0.0, 1.0)
    n, p = total_games, score
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap_confidence_interval(
    pair_scores: Sequence[float],
    num_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Tính khoảng tin cậy Bootstrap 95% trên 50 cặp ván khai cuộc đối xứng."""
    scores = list(pair_scores)
    if not scores:
        raise ValueError("Cần ít nhất một cặp ván")
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(sum(rng.choice(scores) for _ in range(n)) / n for _ in range(num_resamples))
    a = (1 - confidence) / 2
    return (means[int(a * num_resamples)], means[min(num_resamples - 1, int((1 - a) * num_resamples))])

