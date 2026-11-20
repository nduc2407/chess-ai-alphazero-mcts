"""
chess_ai/training/train_rl_krk.py
Kịch bản huấn luyện tác tử Học tăng cường Cấp 6 (Tabular Q-Learning & Policy Gradient)
trên bài toán cờ tàn chuẩn FIDE KRK (Vua + Xe đấu Vua), lưu bảng Q ra checkpoints/l6_krk_qtable.pkl
và xuất dữ liệu đường cong học tập ra results/training/krk_learning_curve.json.
Phụ trách: TV1 (Core, Classical Search, UCT & RL Lead)

Thiết lập: Trắng (agent) có Vua+Xe, Đen là Vua đơn độc đi ngẫu nhiên. Trạng thái = FEN rút gọn
(vị trí quân + bên đi), hành động = UCI. Phần thưởng: +1 chiếu hết, -0.01 mỗi nước, -0.2 hòa/hết giới hạn.
"""
import json
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess

from chess_ai.config import load_yaml_config


def krk_state_key(board: chess.Board) -> str:
    return board.board_fen() + (" w" if board.turn == chess.WHITE else " b")


def random_krk_board(rng: random.Random) -> chess.Board:
    """Thế KRK hợp lệ ngẫu nhiên, Trắng đi, Đen không bị chiếu sẵn, không kết thúc."""
    while True:
        sq = rng.sample(range(64), 3)
        b = chess.Board(None)
        b.set_piece_at(sq[0], chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sq[1], chess.Piece(chess.ROOK, chess.WHITE))
        b.set_piece_at(sq[2], chess.Piece(chess.KING, chess.BLACK))
        b.turn = chess.WHITE
        if b.is_valid() and not b.is_game_over() and not b.is_check():
            return b


def _white_move_reward(board: chess.Board, move: chess.Move, step_penalty: float,
                       win_reward: float, draw_penalty: float) -> Tuple[float, bool]:
    board.push(move)
    if board.is_checkmate():
        return win_reward, True
    if board.is_game_over():  # hòa (pat, thiếu quân khi Xe bị ăn...)
        return draw_penalty, True
    return step_penalty, False


def _black_reply(board: chess.Board, rng: random.Random) -> None:
    board.push(rng.choice(list(board.legal_moves)))


def _greedy(q: Dict, key: str, moves: List[chess.Move], rng: random.Random) -> chess.Move:
    vals = [q.get((key, m.uci()), 0.0) for m in moves]
    best = max(vals)
    return rng.choice([m for m, v in zip(moves, vals) if v == best])


def train_krk_rl(
    config_path: str = "configs/search_params.yaml",
    output_qtable: str = "checkpoints/l6_krk_qtable.pkl",
    output_metrics: str = "results/training/krk_learning_curve.json",
    episodes: Optional[int] = None,
    max_moves: int = 50,
    seed: int = 42,
    eval_every: int = 1000,
    eval_episodes: int = 100,
) -> Dict[str, Any]:
    """Thực thi huấn luyện RL trên cờ tàn KRK (Q-learning + REINFORCE) và lưu bảng Q + Learning Curve."""
    cfg = load_yaml_config(config_path)["rl_level6"]
    episodes = episodes or int(cfg["episodes"])
    gamma, lr = float(cfg["gamma"]), float(cfg["learning_rate"])
    eps0, eps1 = float(cfg["epsilon_start"]), float(cfg["epsilon_end"])
    step_pen, win_r, draw_p = float(cfg["step_penalty"]), float(cfg["win_reward"]), float(cfg["draw_penalty"])
    rng = random.Random(seed)

    q: Dict[Tuple[str, str], float] = defaultdict(float)
    theta: Dict[Tuple[str, str], float] = defaultdict(float)  # tùy chọn ưu tiên cho REINFORCE
    q_ref: Dict[Tuple[str, str], float] = {}

    def evaluate(table: Dict, greedy: bool, n: int) -> Dict[str, float]:
        erng = random.Random(seed + 777)
        wins, lens = 0, []
        for _ in range(n):
            b = random_krk_board(erng)
            for t in range(max_moves):
                moves = list(b.legal_moves)
                mv = _greedy(table, krk_state_key(b), moves, erng)
                b.push(mv)
                if b.is_game_over():
                    break
                _black_reply(b, erng)
                if b.is_game_over():
                    break
            if b.is_checkmate() and b.turn == chess.BLACK:
                wins += 1; lens.append(len(b.move_stack))
        return dict(win_rate=wins / n, avg_mate_plies=(sum(lens) / len(lens)) if lens else None)

    curve: List[Dict[str, Any]] = []
    # ---- Q-learning (off-policy, bootstrap qua nước đáp ngẫu nhiên của Đen) ----
    for ep in range(episodes):
        eps = max(eps1, eps0 - (eps0 - eps1) * ep / max(1, episodes - 1))
        b = random_krk_board(rng)
        for _ in range(max_moves):
            key, moves = krk_state_key(b), list(b.legal_moves)
            mv = rng.choice(moves) if rng.random() < eps else _greedy(q, key, moves, rng)
            r, done = _white_move_reward(b, mv, step_pen, win_r, draw_p)
            if not done:
                _black_reply(b, rng)
                if b.is_game_over():
                    r, done = draw_p, True  # Đen đi hết nước/pat/hòa
            target = r
            if not done:
                nk = krk_state_key(b)
                nm = list(b.legal_moves)
                target = r + gamma * max(q.get((nk, m.uci()), 0.0) for m in nm)
            a = (key, mv.uci())
            q[a] += lr * (target - q[a])
            if done:
                break
        if (ep + 1) % eval_every == 0 or ep + 1 == episodes:
            curve.append(dict(episode=ep + 1, epsilon=eps, table_size=len(q), algo="q_learning",
                              **evaluate(q, True, eval_episodes)))

    # ---- REINFORCE (softmax trên bảng theta, trả về chiết khấu, baseline = 0) ----
    pg_episodes = max(1, episodes // 4)
    for ep in range(pg_episodes):
        b = random_krk_board(rng)
        traj: List[Tuple[str, str, List[str], float]] = []
        for _ in range(max_moves):
            key, moves = krk_state_key(b), list(b.legal_moves)
            ucis = [m.uci() for m in moves]
            prefs = [theta[(key, u)] for u in ucis]
            m_ = max(prefs)
            w = [math.exp(p - m_) for p in prefs]
            mv = rng.choices(moves, weights=w, k=1)[0]
            r, done = _white_move_reward(b, mv, step_pen, win_r, draw_p)
            if not done:
                _black_reply(b, rng)
                if b.is_game_over():
                    r, done = draw_p, True
            traj.append((key, mv.uci(), ucis, r))
            if done:
                break
        G = 0.0
        for key, u, ucis, r in reversed(traj):
            G = r + gamma * G
            prefs = [theta[(key, x)] for x in ucis]
            m_ = max(prefs)
            ex = [math.exp(p - m_) for p in prefs]
            Z = sum(ex)
            for x, e in zip(ucis, ex):
                theta[(key, x)] += 0.05 * G * ((1.0 if x == u else 0.0) - e / Z)
        if (ep + 1) % max(1, eval_every // 4) == 0 or ep + 1 == pg_episodes:
            curve.append(dict(episode=ep + 1, table_size=len(theta), algo="reinforce",
                              **evaluate(theta, True, eval_episodes)))

    Path(output_qtable).parent.mkdir(parents=True, exist_ok=True)
    with open(output_qtable, "wb") as f:
        pickle.dump(dict(q=dict(q), theta=dict(theta), config=cfg, seed=seed), f)
    out = dict(curve=curve, episodes=episodes, reinforce_episodes=pg_episodes, seed=seed,
               max_moves=max_moves, config=cfg)
    Path(output_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(output_metrics).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    train_krk_rl()


if __name__ == "__main__":
    main()
