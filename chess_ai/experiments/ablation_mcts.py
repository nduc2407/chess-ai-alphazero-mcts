"""
chess_ai/experiments/ablation_mcts.py
Ablation 2:
- Phần A: Khảo sát độ nhạy ngân sách mô phỏng N_sim in {50, 100, 200, 400, 800, 1000}
  cho cả Cấp 4 (Vanilla MCTS UCT) và Cấp 7 (Guided MCTS PUCT).
- Phần B: Bóc tách đóng góp giữa Policy-only (Cấp 5), Value-only MCTS và Full Guided MCTS (Cấp 7).
Phụ trách: TV3 phối hợp TV2
"""


import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chess

from chess_ai.agents.level2_heuristic import Level2HeuristicAgent
from chess_ai.agents.level4_mcts import Level4MCTSAgent
from chess_ai.arena.match import play_single_match
from chess_ai.arena.metrics import compute_normalized_score, wilson_confidence_interval
from chess_ai.arena.tournament import load_openings


def run_ablation_mcts(
    output_path: str = "results/ablation2/ablation2_results.json",
    sims_grid: Sequence[int] = (50, 100, 200, 400, 800, 1000),
    openings_path: str = "resources/openings_50.epd",
    games_per_point: int = 20,
    max_plies: int = 300,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Phần A: quét N_sim cho Cấp 4 (và Cấp 7 nếu có checkpoint) so với Cấp 2; lưu JSON."""
    openings = load_openings(openings_path) or [chess.STARTING_FEN]
    makers = {"level4": lambda n: Level4MCTSAgent(simulations=n)}
    if checkpoint_path:
        from chess_ai.agents.level7_guided_mcts import Level7GuidedMCTSAgent
        makers["level7"] = lambda n: Level7GuidedMCTSAgent(checkpoint_path=checkpoint_path, simulations=n)
    results: List[Dict[str, Any]] = []
    for name, mk in makers.items():
        for n in sims_grid:
            w = d = l = 0
            for g in range(games_per_point):
                fen = openings[(g // 2) % len(openings)]
                agent, opp = mk(n), Level2HeuristicAgent(seed=g)
                a_white = g % 2 == 0
                r = play_single_match(agent if a_white else opp, opp if a_white else agent, initial_fen=fen,
                                      max_workload=n, max_plies=max_plies, seed=g)["result"]
                s = r if a_white else -r
                w += s > 0; l += s < 0; d += s == 0
            sc = compute_normalized_score(w, d, l)
            results.append(dict(agent=name, sims=n, wins=w, draws=d, losses=l, score=sc,
                                wilson_ci=wilson_confidence_interval(sc, w + d + l)))
    out = dict(opponent="level2", games_per_point=games_per_point, results=results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    pass



def run_ablation_components(
    checkpoint_path: str,
    output_path: str = "results/ablation2/ablation2_components.json",
    simulations: int = 200,
    openings_path: str = "resources/openings_50.epd",
    games_per_pair: int = 20,
    max_plies: int = 300,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Phần B: bóc tách đóng góp Policy-only (Cấp 5), Value-only MCTS, Prior-only và Full Guided MCTS.

    Mỗi biến thể đấu với Cấp 4 (UCT) cùng ngân sách mô phỏng, đổi màu xen kẽ.
    """
    from chess_ai.agents.level5_supervised import Level5SupervisedAgent
    from chess_ai.agents.level7_guided_mcts import Level7GuidedMCTSAgent

    openings = load_openings(openings_path) or [chess.STARTING_FEN]
    variants = {
        "policy_only_L5": lambda: Level5SupervisedAgent(checkpoint_path=checkpoint_path, device=device),
        "value_only_mcts": lambda: Level7GuidedMCTSAgent(checkpoint_path=checkpoint_path, simulations=simulations,
                                                         use_prior=False, use_value=True, device=device),
        "prior_only_mcts": lambda: Level7GuidedMCTSAgent(checkpoint_path=checkpoint_path, simulations=simulations,
                                                         use_prior=True, use_value=False, device=device),
        "full_guided_L7": lambda: Level7GuidedMCTSAgent(checkpoint_path=checkpoint_path, simulations=simulations,
                                                        device=device),
    }
    results: List[Dict[str, Any]] = []
    for name, mk in variants.items():
        w = d = l = 0
        for g in range(games_per_pair):
            fen = openings[(g // 2) % len(openings)]
            agent, opp = mk(), Level4MCTSAgent(simulations=simulations, seed=g)
            a_white = g % 2 == 0
            r = play_single_match(agent if a_white else opp, opp if a_white else agent, initial_fen=fen,
                                  max_workload=simulations, max_plies=max_plies, seed=g)["result"]
            s_a = r if a_white else -r
            w += s_a > 0; l += s_a < 0; d += s_a == 0
        sc = compute_normalized_score(w, d, l)
        results.append(dict(variant=name, wins=w, draws=d, losses=l, score=sc,
                            wilson_ci=wilson_confidence_interval(sc, w + d + l)))
    out = dict(opponent="level4", simulations=simulations, games_per_variant=games_per_pair, results=results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out
