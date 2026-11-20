"""
chess_ai/search/mcts_puct.py
Guided MCTS (PUCT) dùng Policy Prior P(s, a) và Value Head V(s) của mạng Dual-Head ResNet
cho Cấp 7 và Cấp 8. Tách rõ eval_mode (không nhiễu, chọn tất định) và selfplay_mode (Dirichlet noise).

Quy ước giá trị giống mcts_uct.py: value_sum của một nút lưu theo góc nhìn của nút cha
(bên đã đi nước dẫn tới nút đó), nên Q của nút con dùng thẳng khi chọn ở nút cha.

Giới hạn đã biết: phong cấp dưới (xe, tượng, mã) không được tìm kiếm vì không gian hành động 4096
chỉ có một chỉ số cho mỗi cặp (ô đi, ô đến); mặc định phong Hậu.
"""
import math
import random
import time
from typing import Any, Dict, Optional, Tuple

import chess
import numpy as np
import torch

from chess_ai.core.board_tensor import ACTION_SPACE, board_to_tensor, move_to_action_index
from chess_ai.core.zobrist import compute_zobrist_hash
from chess_ai.search.mcts_node import MCTSNode
from chess_ai.search.mcts_uct import _backpropagate, _is_fast_terminal

_UNDERPROMO = (chess.ROOK, chess.BISHOP, chess.KNIGHT)


def _terminal_value(board: chess.Board) -> Optional[float]:
    v = _is_fast_terminal(board)
    if v is not None:
        return v
    if board.is_repetition(3):
        return 0.0
    return None


def _searchable_moves(board: chess.Board):
    return [m for m in board.legal_moves if m.promotion not in _UNDERPROMO]


@torch.no_grad()
def _evaluate(model, board: chess.Board, device) -> Tuple[list, np.ndarray, float]:
    """Một lần suy luận mạng: trả về (nước hợp lệ, prior đã softmax trên các nước đó, value bên đi)."""
    moves = _searchable_moves(board)
    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    logits, value = model(x)
    idxs = [move_to_action_index(m, board.turn) for m in moves]
    probs = torch.softmax(logits[0, idxs], dim=0).cpu().numpy().astype(np.float64)
    return moves, probs, float(value[0, 0])


def _select_child(node: MCTSNode, c_puct: float):
    sqrt_n = math.sqrt(max(1, node.visit_count))
    best_move, best_child, best_score = None, None, -float("inf")
    for mv, ch in node.children.items():
        score = ch.q_value + c_puct * ch.prior * sqrt_n / (1 + ch.visit_count)
        if score > best_score:
            best_move, best_child, best_score = mv, ch, score
    return best_move, best_child


def _pick_by_temperature(visits: np.ndarray, temperature: float, rng: random.Random) -> int:
    if temperature <= 1e-6:
        return int(np.argmax(visits))
    w = np.power(visits.astype(np.float64), 1.0 / temperature)
    s = w.sum()
    if s <= 0:
        return int(np.argmax(visits))
    w /= s
    return int(rng.choices(range(len(w)), weights=w.tolist(), k=1)[0])


def run_puct_search(
    board: chess.Board,
    model: Any,
    simulations: int = 400,
    time_limit: Optional[float] = None,
    c_puct: float = 1.414,
    eval_mode: bool = True,
    temperature: float = 0.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    root: Optional[MCTSNode] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[chess.Move, Dict[str, Any], MCTSNode]:
    """Tìm kiếm PUCT; trả về (best_move, stats, root). stats['pi'] là phân phối lượt thăm (4096,)."""
    t0 = time.perf_counter()
    rng = rng or random.Random()
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        board = board.copy()  # giữ stack để phát hiện lặp thế
        if root is None:
            root = MCTSNode(zobrist_key=compute_zobrist_hash(board))
        root.parent = None
        net_evals = 0
        if not root.is_expanded():
            moves, probs, _ = _evaluate(model, board, device)
            net_evals += 1
            root.expand(moves, {m: float(p) for m, p in zip(moves, probs)})
        if not eval_mode and len(root.children) > 1:
            noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).dirichlet(
                [dirichlet_alpha] * len(root.children))
            for ch, n in zip(root.children.values(), noise):
                ch.prior = (1 - dirichlet_epsilon) * ch.prior + dirichlet_epsilon * float(n)

        sims_done, max_depth = 0, 0
        if len(root.children) > 1:
            for _ in range(simulations):
                if time_limit is not None and time.perf_counter() - t0 >= time_limit:
                    break
                node, depth = root, 0
                while node.is_expanded():
                    mv, node = _select_child(node, c_puct)
                    board.push(mv)
                    depth += 1
                    if node.zobrist_key is None:
                        node.zobrist_key = compute_zobrist_hash(board)
                term = _terminal_value(board)
                if term is not None:
                    _backpropagate(node, term)
                else:
                    moves, probs, v = _evaluate(model, board, device)
                    net_evals += 1
                    node.expand(moves, {m: float(p) for m, p in zip(moves, probs)})
                    _backpropagate(node, v)
                for _ in range(depth):
                    board.pop()
                sims_done += 1
                max_depth = max(max_depth, depth)

        children = list(root.children.items())
        visits = np.array([c.visit_count for _, c in children], dtype=np.float64)
        if visits.sum() == 0:  # một nước duy nhất hoặc hết ngân sách
            visits = np.array([c.prior for _, c in children], dtype=np.float64)
        pick = _pick_by_temperature(visits, 0.0 if eval_mode else temperature, rng)
        best_move = children[pick][0]
        pi = np.zeros(ACTION_SPACE, dtype=np.float32)
        total = visits.sum()
        for (mv, _), v in zip(children, visits):
            pi[move_to_action_index(mv, board.turn)] = v / total
        best_child = children[int(np.argmax([c.visit_count for _, c in children]))][1]
        stats = dict(
            simulations=sims_done, nodes=float(sims_done), net_evals=net_evals,
            time_sec=time.perf_counter() - t0, max_depth=max_depth,
            root_q=float(best_child.q_value), pi=pi,
        )
        return best_move, stats, root
    finally:
        model.train(was_training)


def search_puct(
    board: chess.Board,
    model: Any,
    simulations: int = 400,
    time_limit: Optional[float] = None,
    c_puct: float = 1.414,
    eval_mode: bool = True,
) -> Tuple[chess.Move, Dict[str, Any]]:
    """Thực thi tìm kiếm Guided MCTS (PUCT) và trả về (best_move, stats_and_pi)."""
    mv, stats, _ = run_puct_search(board, model, simulations, time_limit, c_puct, eval_mode)
    return mv, stats
