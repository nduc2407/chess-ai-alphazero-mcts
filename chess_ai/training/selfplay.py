"""
chess_ai/training/selfplay.py
Vòng lặp tự chơi (Self-play Loop), cập nhật mạng nơ-ron trên Replay Buffer
và cơ chế đấu loại trực tiếp Checkpoint Arena Gating (ngưỡng thắng >= 55%) cho Cấp 8.
Phụ trách: TV2 phối hợp TV1

Chỉ báo cáo số liệu từ lần chạy thật; test đơn vị dùng mạng nhỏ chỉ để kiểm tra mã.
"""
import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import numpy as np
import torch

from chess_ai.agents.level7_guided_mcts import Level7GuidedMCTSAgent
from chess_ai.arena.match import play_single_match
from chess_ai.config import load_yaml_config
from chess_ai.core.board_tensor import board_to_tensor
from chess_ai.models.dual_resnet import DualHeadResNet
from chess_ai.models.io import load_model, save_checkpoint
from chess_ai.models.loss import DualHeadLoss
from chess_ai.search.mcts_puct import run_puct_search
from chess_ai.training.replay_buffer import SelfPlayReplayBuffer
from chess_ai.utils.seeding import set_global_seed


def play_selfplay_game(
    model, simulations: int = 200, c_puct: float = 1.414, max_plies: int = 300,
    temp_initial: float = 1.0, temp_final: float = 0.1, temp_decay_plies: int = 20,
    dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
    start_fen: str = chess.STARTING_FEN, rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Một ván tự chơi; trả về (states, policies, values z theo bên đi, kết quả góc nhìn Trắng)."""
    rng = rng or random.Random()
    board = chess.Board(start_fen)
    states, pis, turns = [], [], []
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < max_plies:
        temp = temp_initial if len(board.move_stack) < temp_decay_plies else temp_final
        mv, st, _ = run_puct_search(
            board, model, simulations=simulations, c_puct=c_puct, eval_mode=False,
            temperature=temp, dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon, rng=rng)
        states.append(board_to_tensor(board)); pis.append(st["pi"]); turns.append(board.turn)
        board.push(mv)
    outcome = board.outcome(claim_draw=True)
    res = 0.0 if outcome is None or outcome.winner is None else (1.0 if outcome.winner == chess.WHITE else -1.0)
    z = np.array([res if t == chess.WHITE else -res for t in turns], dtype=np.float32)
    return np.stack(states), np.stack(pis), z, res


def train_on_buffer(model, opt, crit, buf: SelfPlayReplayBuffer, steps: int, batch_size: int, device: str) -> float:
    model.train()
    tot = 0.0
    for _ in range(steps):
        s, p, v = buf.sample_batch(batch_size)
        s, p, v = (torch.from_numpy(a).to(device) for a in (s, p, v))
        opt.zero_grad(set_to_none=True)
        logits, val = model(s)
        loss, _ = crit(logits, val, p, v)
        loss.backward()
        opt.step()
        tot += float(loss.detach())
    model.eval()
    return tot / max(steps, 1)


def arena_gating(challenger, champion, games: int, simulations: int, max_plies: int, seed: int,
                 device: str = "cpu") -> Dict[str, Any]:
    """Đấu challenger vs champion, đổi màu xen kẽ; điểm = thắng 1, hòa 0.5."""
    a = Level7GuidedMCTSAgent(model=challenger, simulations=simulations, eval_mode=True, device=device,
                              subtree_reuse=False, name="challenger", seed=seed)
    b = Level7GuidedMCTSAgent(model=champion, simulations=simulations, eval_mode=True, device=device,
                              subtree_reuse=False, name="champion", seed=seed)
    w = d = l = 0
    for g in range(games):
        cw = g % 2 == 0
        r = play_single_match(a if cw else b, b if cw else a, max_workload=simulations,
                              max_plies=max_plies, seed=seed + g)["result"]
        s = r if cw else -r
        w += s > 0; l += s < 0; d += s == 0
    return dict(wins=int(w), draws=int(d), losses=int(l), games=games,
                score=(w + 0.5 * d) / max(games, 1))


def run_selfplay_pipeline(
    config_path: str = "configs/search_params.yaml",
    warm_start_checkpoint: str = "checkpoints/best_model.pt",
    output_dir: str = "results/training/",
    iterations: int = 1,
    seed: int = 42,
    device: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    champion_path: str = "checkpoints/best_az_model.pt",
) -> Dict[str, Any]:
    """Thực thi chu trình Self-play -> Training -> Arena Gating cho Cấp 8."""
    cfg = load_yaml_config(config_path)
    az = dict(cfg["alphazero_level8"]); az.update(overrides or {})
    sp = cfg["guided_mcts"]["selfplay_mode"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(seed)
    rng = random.Random(seed)

    if az.get("warm_start_from_level5", True) and Path(warm_start_checkpoint).exists():
        champion = load_model(warm_start_checkpoint, device)
        model_config = champion.config
    else:  # Tabula rasa
        model_config = model_config or {}
        champion = DualHeadResNet(**model_config).to(device).eval()
    challenger = copy.deepcopy(champion)
    opt = torch.optim.AdamW(challenger.parameters(), lr=float(az.get("learning_rate", 1e-4)),
                            weight_decay=float(az.get("weight_decay", 1e-4)))
    crit = DualHeadLoss(1.0, 1.0)
    buf = SelfPlayReplayBuffer(int(az["replay_buffer_size"]), seed)
    sims = int(az["mcts_sims_per_move"])
    max_plies = int(az.get("max_plies", 300))
    log: List[Dict[str, Any]] = []
    t0 = time.time()
    for it in range(iterations):
        results = []
        for g in range(int(az["selfplay_games_per_iter"])):
            s, p, v, r = play_selfplay_game(
                champion, sims, max_plies=max_plies, temp_initial=sp["temp_initial"],
                temp_final=sp["temp_final"], temp_decay_plies=sp["temp_decay_plies"],
                dirichlet_alpha=sp["dirichlet_alpha"], dirichlet_epsilon=sp["dirichlet_epsilon"], rng=rng)
            buf.push_game_samples(s, p, v); results.append(r)
        steps = int(az.get("train_steps_per_iter", 100))
        loss = train_on_buffer(challenger, opt, crit, buf, steps, int(az.get("batch_size", 128)), device)
        gate = arena_gating(challenger, champion, int(az["arena_eval_games"]), sims, max_plies, seed + 1000 * it, device)
        promoted = gate["score"] >= float(az["arena_win_threshold"])
        if promoted:
            champion = copy.deepcopy(challenger).eval()
            save_checkpoint(champion_path, dict(model_state_dict=champion.state_dict(),
                            model_config=champion.config, iteration=it, random_seed=seed))
        else:  # challenger quay lại champion để tránh trôi
            challenger = copy.deepcopy(champion)
            opt = torch.optim.AdamW(challenger.parameters(), lr=float(az.get("learning_rate", 1e-4)),
                                    weight_decay=float(az.get("weight_decay", 1e-4)))
        log.append(dict(iteration=it, buffer=len(buf), train_loss=loss, selfplay_results=results,
                        gating=gate, promoted=bool(promoted), elapsed_sec=time.time() - t0))
        print(json.dumps({k: v for k, v in log[-1].items() if k != "selfplay_results"}, ensure_ascii=False), flush=True)
    out = dict(iterations=log, config=az, seed=seed, device=device)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "selfplay_log.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Self-play + gating Cấp 8")
    ap.add_argument("--config", default="configs/search_params.yaml")
    ap.add_argument("--warm_start", default="checkpoints/best_model.pt")
    ap.add_argument("--output_dir", default="results/training/")
    ap.add_argument("--iterations", type=int, default=1)
    a = ap.parse_args()
    run_selfplay_pipeline(a.config, a.warm_start, a.output_dir, a.iterations)


if __name__ == "__main__":
    main()
