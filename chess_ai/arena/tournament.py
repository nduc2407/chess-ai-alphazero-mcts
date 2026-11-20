"""
chess_ai/arena/tournament.py
Động cơ điều phối giải đấu Round-Robin 100 ván/cặp (50 thế cờ khai cuộc cân bằng x 2 màu)
hỗ trợ hai chế độ ngân sách: Workload-Control và Time-Control.
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

import json
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional
import chess
from chess_ai.agents.base_agent import BaseAgent
from chess_ai.arena.match import play_single_match
from chess_ai.arena.metrics import (
    compute_normalized_score, paired_bootstrap_confidence_interval, wilson_confidence_interval,
)
from chess_ai.arena.pgn_exporter import append_match_jsonl, export_game_to_pgn
from chess_ai.config import load_yaml_config


def load_openings(path: str, limit: Optional[int] = None) -> List[str]:
    """Đọc thế cờ khai cuộc từ EPD (mỗi dòng: 4 trường FEN đầu, phần sau bỏ qua)."""
    fens = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split(";")[0].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            fens.append(" ".join(parts[:4]) + " 0 1")
    return fens[:limit] if limit else fens


def run_round_robin_tournament(
    agents: List[BaseAgent],
    config_path: str = "configs/default_config.yaml",
    budget_mode: str = "workload_control",
    workloads: Optional[Dict[str, int]] = None,
    games_per_pair: Optional[int] = None,
    max_plies: Optional[int] = None,
    output_dir: Optional[str] = None,
    openings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Vận hành giải đấu đối kháng giữa danh sách tác tử và lưu vết ra results/round_robin/.

    workloads: {tên tác tử: số nút/mô phỏng mỗi nước} (chế độ workload_control; thiếu = không giới hạn).
    Mỗi thế khai cuộc được đấu hai lần đổi màu -> một "cặp" cho bootstrap theo cặp.
    """
    cfg = load_yaml_config(config_path)
    arena = cfg["arena"]
    out = Path(output_dir or cfg["paths"]["round_robin_dir"])
    out.mkdir(parents=True, exist_ok=True)
    gpp = games_per_pair or int(arena["games_per_pair"])
    max_plies = max_plies or int(arena["max_plies"])
    if openings is None:
        openings = load_openings(cfg["paths"]["openings_epd"], gpp // 2)
    if not openings:
        openings = [chess.STARTING_FEN]
    n_open = max(1, gpp // 2)
    openings = (openings * ((n_open // len(openings)) + 1))[:n_open]
    workloads = workloads or {}
    jsonl, pgn = out / "matches.jsonl", out / "games.pgn"
    for f in (jsonl, pgn):
        f.unlink(missing_ok=True)
    summary: Dict[str, Any] = {"budget_mode": budget_mode, "pairs": {}}
    for a, b in itertools.combinations(agents, 2):
        w = d = l = 0
        pair_scores: List[float] = []
        for i, fen in enumerate(openings):
            ps = 0.0
            for swap in (False, True):
                white, black = (b, a) if swap else (a, b)
                kw: Dict[str, Any] = {}
                # ngân sách áp theo từng tác tử: bọc select_move qua thuộc tính tạm
                res = _play(white, black, fen, budget_mode, workloads, arena, max_plies, seed=1000 + i)
                s_a = -res["result"] if swap else res["result"]
                w += s_a > 0; l += s_a < 0; d += s_a == 0
                ps += (s_a + 1) / 2
                rec = {k: v for k, v in res.items() if k not in ("search_stats",)}
                rec.update(opening_id=i, agent_a=a.name, agent_b=b.name, a_is_white=not swap)
                append_match_jsonl(str(jsonl), rec)
                res_str = {1.0: "1-0", -1.0: "0-1", 0.0: "1/2-1/2"}[res["result"]]
                text = export_game_to_pgn(
                    [chess.Move.from_uci(m) for m in res["moves"]],
                    dict(Event="Round-Robin", White=white.name, Black=black.name, Result=res_str), fen)
                with open(pgn, "a", encoding="utf-8") as f:
                    f.write(text + "\n\n")
            pair_scores.append(ps / 2)
        n = w + d + l
        sc = compute_normalized_score(w, d, l)
        summary["pairs"][f"{a.name} vs {b.name}"] = dict(
            wins=w, draws=d, losses=l, games=n, score=sc,
            wilson_ci=wilson_confidence_interval(sc, n),
            bootstrap_ci=paired_bootstrap_confidence_interval(pair_scores))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def _play(white, black, fen, budget_mode, workloads, arena, max_plies, seed):
    """Một ván; mỗi tác tử nhận ngân sách riêng qua lớp bọc."""
    class _Budgeted(BaseAgent):
        def __init__(self, inner):
            super().__init__(inner.name); self.inner = inner
        def reset(self):
            self.inner.reset()
        def select_move(self, env, time_limit=None, max_workload=None):
            if budget_mode == "time_control":
                mv = self.inner.select_move(env, time_limit=float(arena["time_limit_sec"]))
            else:
                mv = self.inner.select_move(env, max_workload=workloads.get(self.inner.name))
            self.last_search_stats = dict(self.inner.last_search_stats)
            return mv
    return play_single_match(_Budgeted(white), _Budgeted(black), initial_fen=fen,
                             max_plies=max_plies, seed=seed)


def build_agents(levels: List[str], ckpt: Dict[str, str], workloads_cfg: Dict[str, int]):
    """Tạo danh sách tác tử từ mã cấp; trả về (agents, workloads theo tên tác tử).

    Cấp 6 không đấu 8x8 (chỉ KRK) nên bị loại khỏi giải vòng tròn."""
    from chess_ai.agents.registry import AGENT_SPEC_REGISTRY, create_agent
    agents, workloads = [], {}
    for lv in levels:
        if not AGENT_SPEC_REGISTRY[lv].get("full_board_supported", True):
            continue
        kw: Dict[str, Any] = {}
        if lv in ("level5", "level7", "level8"):
            if lv not in ckpt:
                raise SystemExit(f"{lv} cần --ckpt {lv}=đường_dẫn_checkpoint")
            kw["checkpoint_path"] = ckpt[lv]
        if lv in ("level1", "level2"):
            kw["seed"] = 42
        ag = create_agent(lv, **kw)
        agents.append(ag)
        if lv in workloads_cfg:
            workloads[ag.name] = workloads_cfg[lv]
    return agents, workloads


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Giải vòng tròn Cấp 1-8 (trừ Cấp 6, chỉ KRK)")
    ap.add_argument("--levels", nargs="+", default=["level1", "level2", "level3", "level4"])
    ap.add_argument("--ckpt", nargs="*", default=[], help="level5=path level7=path level8=path")
    ap.add_argument("--workload", nargs="*", default=["level3=2000", "level4=200", "level7=200", "level8=200"],
                    help="ngân sách nút/mô phỏng mỗi nước, dạng levelN=số")
    ap.add_argument("--mode", default="workload_control")
    ap.add_argument("--games_per_pair", type=int, default=None)
    ap.add_argument("--max_plies", type=int, default=None)
    ap.add_argument("--output_dir", default=None)
    a = ap.parse_args()
    ags, wl = build_agents(a.levels, dict(x.split("=", 1) for x in a.ckpt),
                           {k: int(v) for k, v in (x.split("=", 1) for x in a.workload)})
    run_round_robin_tournament(ags, budget_mode=a.mode, workloads=wl, games_per_pair=a.games_per_pair,
                               max_plies=a.max_plies, output_dir=a.output_dir)
