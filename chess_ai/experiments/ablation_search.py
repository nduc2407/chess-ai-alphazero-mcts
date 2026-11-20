"""
chess_ai/experiments/ablation_search.py
Ablation 1: Đánh giá 5 cấu hình tìm kiếm cổ điển lũy tiến (Config_A -> Config_D)
và khảo sát độ nhạy theo độ sâu d in {2, 3, 4} trên bộ 50 thế cờ WAC (resources/benchmark_positions.epd).
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import chess

from chess_ai.search.alphabeta import search_alphabeta
from chess_ai.utils.epd import load_epd_file

CONFIGS: Dict[str, Dict[str, bool]] = {
    "Config_A Minimax thuần": dict(use_alpha_beta=False, use_move_ordering=False, use_tt=False, use_quiescence=False),
    "Config_B + Alpha-Beta": dict(use_alpha_beta=True, use_move_ordering=False, use_tt=False, use_quiescence=False),
    "Config_C + Move Ordering": dict(use_alpha_beta=True, use_move_ordering=True, use_tt=False, use_quiescence=False),
    "Config_D + TT": dict(use_alpha_beta=True, use_move_ordering=True, use_tt=True, use_quiescence=False),
    "Config_E + Quiescence (đầy đủ)": dict(use_alpha_beta=True, use_move_ordering=True, use_tt=True, use_quiescence=True),
}


def run_ablation_search(
    epd_path: str = "resources/benchmark_positions.epd",
    output_path: str = "results/ablation1/ablation1_results.json",
    depths: Sequence[int] = (2, 3, 4),
) -> List[Dict[str, Any]]:
    """Chạy thực nghiệm Ablation 1 và lưu kết quả JSON (số nút, thời gian, số thế giải đúng)."""
    records = load_epd_file(epd_path)
    rows: List[Dict[str, Any]] = []
    for depth in depths:
        for name, kw in CONFIGS.items():
            nodes = solved = 0
            t = 0.0
            for rec in records:
                mv, st = search_alphabeta(chess.Board(rec.fen), max_depth=depth, **kw)
                nodes += int(st["nodes"]); t += float(st["time_sec"]); solved += int(mv in rec.best_moves)
            rows.append(dict(depth=depth, config=name, total_nodes=nodes, mean_nodes=nodes / len(records),
                             total_time_sec=t, solved=solved, n=len(records)))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return rows


if __name__ == "__main__":
    run_ablation_search()
