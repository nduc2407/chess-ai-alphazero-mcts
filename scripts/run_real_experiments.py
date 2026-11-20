"""
scripts/run_real_experiments.py
Chạy các thực nghiệm THẬT cho các tác tử đã cài đặt (Cấp 1-4) và ghi log đầy đủ.
Các Cấp 5-8 chưa được cài đặt trong repo nên KHÔNG được chạy / báo cáo.

Lệnh con:
  openings   - sinh bộ khai cuộc cân bằng có seed (file resources/openings_50.epd của repo rỗng)
  roundrobin - giải đấu vòng tròn cấp 1-4, cân bằng màu
  ablation_search - ablation thành phần Alpha-Beta trên WAC-50
  wac        - tỷ lệ giải đúng WAC-50 của L2/L3/L4
  ablation_mcts - số mô phỏng UCT vs Cấp 2
"""
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):  # Windows (cp1252) không in được tiếng Việt
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse, json, os, random, sys, time
from multiprocessing import Pool
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chess_ai.agents.level1_random import Level1RandomAgent
from chess_ai.agents.level2_heuristic import Level2HeuristicAgent
from chess_ai.agents.level3_alphabeta import Level3AlphaBetaAgent
from chess_ai.agents.level4_mcts import Level4MCTSAgent
from chess_ai.core.chess_env import ChessEnv
from chess_ai.core.evaluation import evaluate_board
from chess_ai.search.alphabeta import search_alphabeta

RES = ROOT / "results"
OPEN_FILE = RES / "round_robin" / "openings_generated.epd"
L3_NODES = 2000   # ngân sách nút / nước của Cấp 3
L4_SIMS = 200     # số mô phỏng / nước của Cấp 4
MAX_PLIES = 300


def make_agent(name, seed, sims=None):
    if name == "L1": return Level1RandomAgent(seed=seed)
    if name == "L2": return Level2HeuristicAgent(seed=seed)
    if name == "L3": return Level3AlphaBetaAgent()
    if name == "L4": return Level4MCTSAgent(seed=seed)
    raise ValueError(name)


def budget(name, sims=None):
    if name == "L3": return dict(max_workload=L3_NODES)
    if name == "L4": return dict(max_workload=sims or L4_SIMS)
    return {}


def gen_openings(n=50, plies=8, seed=2024, max_abs_cp=60):
    rng = random.Random(seed)
    out = []
    seen = set()
    while len(out) < n:
        b = chess.Board()
        for _ in range(plies):
            b.push(rng.choice(list(b.legal_moves)))
            if b.is_game_over():
                break
        if b.is_game_over():
            continue
        key = b.board_fen()
        if key in seen:
            continue
        ev = evaluate_board(b)
        ev_white = ev if b.turn == chess.WHITE else -ev
        if abs(ev_white) > max_abs_cp:
            continue
        seen.add(key)
        out.append(b.fen())
    return out


def load_openings():
    return [l.strip() for l in open(OPEN_FILE, encoding="utf-8") if l.strip() and not l.startswith("#")]


def play_game(a_name, b_name, fen, a_white, seed, a_sims=None, b_sims=None):
    """a_name là tác tử A; a_white xác định A cầm Trắng."""
    A = make_agent(a_name, seed, a_sims); B = make_agent(b_name, seed + 1, b_sims)
    A.reset(); B.reset()
    env = ChessEnv(fen)
    white, black = (A, B) if a_white else (B, A)
    wb = budget(a_name if a_white else b_name, a_sims if a_white else b_sims)
    bb = budget(b_name if a_white else a_name, b_sims if a_white else a_sims)
    lat = {"A": [], "B": []}
    moves = []
    nodes = {"A": 0.0, "B": 0.0}
    while not env.is_terminal() and len(env.board.move_stack) < MAX_PLIES:
        white_to_move = env.board.turn == chess.WHITE
        ag = white if white_to_move else black
        bud = wb if white_to_move else bb
        who = "A" if (white_to_move == a_white) else "B"
        t = time.perf_counter()
        mv = ag.select_move(env, **bud)
        lat[who].append(time.perf_counter() - t)
        st = ag.last_search_stats
        nodes[who] += float(st.get("nodes", 0) or 0)
        moves.append(env.board.san(mv))
        env.step(mv)
    if env.is_terminal():
        r = env.result(); reason = "terminal"
    else:
        r = 0.0; reason = "max_plies"
    a_score = (r if a_white else -r)  # +1 A thắng, -1 A thua
    a_res = "win" if a_score > 0 else ("loss" if a_score < 0 else "draw")
    return dict(A=a_name, B=b_name, a_white=a_white, fen=fen, seed=seed, result_white=r,
                a_result=a_res, plies=len(env.board.move_stack), reason=reason,
                a_sims=a_sims, b_sims=b_sims,
                lat_A_mean=sum(lat["A"]) / max(1, len(lat["A"])),
                lat_B_mean=sum(lat["B"]) / max(1, len(lat["B"])),
                lat_A_max=max(lat["A"] or [0]), lat_B_max=max(lat["B"] or [0]),
                nodes_A=nodes["A"], nodes_B=nodes["B"], n_moves_A=len(lat["A"]), n_moves_B=len(lat["B"]),
                san=moves)


def _task(args):
    return play_game(*args)


def run_games(tasks, out_path, procs=2):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f, Pool(procs) as pool:
        for rec in pool.imap_unordered(_task, tasks):
            f.write(json.dumps(rec) + "\n"); f.flush()
            done += 1
            print(f"[{done}/{len(tasks)}] {rec['A']} vs {rec['B']} white={'A' if rec['a_white'] else 'B'} "
                  f"-> {rec['a_result']} ({rec['plies']} plies, {rec['reason']}) t={time.time()-t0:.0f}s", flush=True)


def cmd_openings(a):
    ops = gen_openings(a.n)
    OPEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OPEN_FILE, "w", encoding="utf-8") as f:
        f.write("# Khai cuộc sinh tự động: 8 plies ngẫu nhiên (seed=2024), |eval tĩnh Trắng| <= 60cp, loại trùng\n")
        for i, fen in enumerate(ops, 1):
            f.write(fen + "\n")
    print(len(ops), "openings ->", OPEN_FILE)


def cmd_roundrobin(a):
    ops = load_openings()[: a.n]
    pairs = [("L1", "L2"), ("L1", "L3"), ("L1", "L4"), ("L2", "L3"), ("L2", "L4"), ("L3", "L4")]
    tasks = []
    for (x, y) in pairs:
        for i, fen in enumerate(ops):
            for a_white in (True, False):
                tasks.append((x, y, fen, a_white, 1000 + i, None, None))
    # xáo thứ tự cố định để cân bằng tải hai tiến trình
    random.Random(1).shuffle(tasks)
    out = RES / "round_robin" / "matches.jsonl"
    run_games(tasks, out, a.procs)


def cmd_ablation_mcts(a):
    ops = load_openings()[: a.n]
    tasks = []
    for sims in (25, 50, 100, 200):
        for i, fen in enumerate(ops):
            for a_white in (True, False):
                tasks.append(("L4", "L2", fen, a_white, 5000 + i, sims, None))
    random.Random(2).shuffle(tasks)
    run_games(tasks, RES / "ablation2" / "matches.jsonl", a.procs)


def _load_wac():
    rows = []
    for l in open(ROOT / "resources" / "benchmark_positions.epd", encoding="utf-8"):
        l = l.strip()
        if not l or l.startswith("#"): continue
        head, _, rest = l.partition(" bm ")
        bm, _, idp = rest.partition(";")
        board = chess.Board(head + " 0 1")
        sols = []
        for san in bm.split():
            try: sols.append(board.parse_san(san))
            except Exception: pass
        rows.append((idp.split('"')[1] if '"' in idp else idp, head + " 0 1", sols))
    return rows


def cmd_ablation_search(a):
    wac = _load_wac()
    cfgs = {
        "Minimax thuần (no AB, no order, no TT, no Q)": dict(use_alpha_beta=False, use_move_ordering=False, use_tt=False, use_quiescence=False),
        "+ Alpha-Beta": dict(use_alpha_beta=True, use_move_ordering=False, use_tt=False, use_quiescence=False),
        "+ Move Ordering": dict(use_alpha_beta=True, use_move_ordering=True, use_tt=False, use_quiescence=False),
        "+ TT": dict(use_alpha_beta=True, use_move_ordering=True, use_tt=True, use_quiescence=False),
        "+ Quiescence (đầy đủ)": dict(use_alpha_beta=True, use_move_ordering=True, use_tt=True, use_quiescence=True),
    }
    rows = []
    for depth in (2, 3):
        for name, kw in cfgs.items():
            nodes = 0; t = 0.0; solved = 0
            for pid, fen, sols in wac:
                b = chess.Board(fen)
                mv, st = search_alphabeta(b, max_depth=depth, **kw)
                nodes += st["nodes"]; t += st["time_sec"]; solved += int(mv in sols)
            r = dict(depth=depth, config=name, total_nodes=nodes, mean_nodes=nodes / len(wac),
                     total_time_sec=t, solved=solved, n=len(wac))
            rows.append(r); print(r, flush=True)
    (RES / "ablation1").mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(RES / "ablation1" / "ablation_search.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def cmd_wac(a):
    wac = _load_wac()
    rows = []
    for name, kw in [("L1", {}), ("L2", {}), ("L3", dict(max_workload=L3_NODES)), ("L3", dict(max_workload=10000)),
                     ("L4", dict(max_workload=200)), ("L4", dict(max_workload=800))]:
        solved = 0; t = 0.0; ids = []
        for pid, fen, sols in wac:
            ag = make_agent(name, 42); ag.reset()
            env = ChessEnv(fen)
            t0 = time.perf_counter(); mv = ag.select_move(env, **kw); t += time.perf_counter() - t0
            ok = mv in sols; solved += int(ok)
            if ok: ids.append(pid)
        r = dict(agent=name, budget=kw, solved=solved, n=len(wac), mean_time_sec=t / len(wac), solved_ids=ids)
        rows.append(r); print(r, flush=True)
    json.dump(rows, open(RES / "ablation1" / "wac.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--procs", type=int, default=2)
    a = p.parse_args()
    globals()["cmd_" + a.cmd](a)
