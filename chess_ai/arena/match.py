"""
chess_ai/arena/match.py
Hàm điều phối một ván đấu đơn lẻ giữa hai tác tử (Single-Game Runner),
dùng chung cho:
1. Giải đấu Round-Robin (chess_ai/arena/tournament.py).
2. Thực nghiệm Ablation 2 (chess_ai/experiments/ablation_mcts.py).
3. Đánh giá Checkpoint Arena Gating của Cấp 8 (chess_ai/training/selfplay.py).
4. Chế độ Máy đấu Máy trên giao diện Pygame (chess_ai/ui/desktop_pygame.py).
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

import time
from typing import Any, Dict, List, Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.core.chess_env import ChessEnv
from chess_ai.utils.seeding import set_global_seed


def play_single_match(
    white_agent: BaseAgent,
    black_agent: BaseAgent,
    initial_fen: str = chess.STARTING_FEN,
    time_limit_sec: Optional[float] = None,
    max_workload: Optional[int] = None,
    max_plies: int = 400,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Điều phối một ván đấu hoàn chỉnh từ initial_fen đến khi kết thúc hoặc chạm max_plies.
    Trả về dictionary chứa kết quả, danh sách nước đi, độ trễ từng nước và thống kê tìm kiếm.
    """
    set_global_seed(seed)
    env = ChessEnv(initial_fen)
    white_agent.reset()
    black_agent.reset()
    moves: List[str] = []
    latencies: List[float] = []
    stats: List[Dict[str, Any]] = []
    while not env.is_terminal() and len(moves) < max_plies:
        agent = white_agent if env.board.turn == chess.WHITE else black_agent
        t0 = time.perf_counter()
        mv = agent.select_move(env, time_limit=time_limit_sec, max_workload=max_workload)
        latencies.append(time.perf_counter() - t0)
        if mv not in env.board.legal_moves:
            raise ValueError(f"{agent.name} trả về nước bất hợp lệ {mv}")
        stats.append(dict(agent.last_search_stats))
        moves.append(mv.uci())
        env.step(mv)
    res = env.result()
    return {
        "white": white_agent.name, "black": black_agent.name, "initial_fen": initial_fen,
        "result": res if res is not None else 0.0,  # góc nhìn Trắng; chạm max_plies tính hòa
        "terminated": res is not None, "num_plies": len(moves), "moves": moves,
        "latencies": latencies, "search_stats": stats, "final_fen": env.board.fen(),
    }

