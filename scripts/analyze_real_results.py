"""
scripts/analyze_real_results.py
Tổng hợp log thật (results/*/matches.jsonl, ablation json) thành bảng + biểu đồ.
Mọi số liệu trong báo cáo lấy từ file summary.json do script này sinh ra.
"""
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):  # Windows (cp1252) không in được tiếng Việt
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json, math, random, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)
NAMES = {"L1": "Cấp 1 Random", "L2": "Cấp 2 Heuristic", "L3": "Cấp 3 Alpha-Beta (2000 nút)", "L4": "Cấp 4 UCT (200 mô phỏng)"}


def wilson(p, n, z=1.96):
    if n == 0: return (0, 0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if Path(path).exists() else []


def score_of(g):
    return {"win": 1.0, "draw": 0.5, "loss": 0.0}[g["a_result"]]


def elo(s):
    s = min(max(s, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / s - 1)


def boot_ci(games, B=10000, seed=7):
    """Bootstrap theo cặp khai cuộc (mỗi khai cuộc gồm 2 ván đổi màu)."""
    by = defaultdict(list)
    for g in games: by[g["fen"]].append(score_of(g))
    units = [np.mean(v) for v in by.values()]
    rng = np.random.default_rng(seed)
    m = [np.mean(rng.choice(units, size=len(units), replace=True)) for _ in range(B)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(games):
    w = sum(g["a_result"] == "win" for g in games); d = sum(g["a_result"] == "draw" for g in games)
    l = sum(g["a_result"] == "loss" for g in games); n = len(games)
    s = (w + 0.5 * d) / n
    lo, hi = wilson(s, n); blo, bhi = boot_ci(games)
    return dict(n=n, W=w, D=d, L=l, score=s, wilson=[lo, hi], boot=[blo, bhi], elo=elo(s),
                maxply_draws=sum(g["reason"] == "max_plies" for g in games),
                mean_plies=float(np.mean([g["plies"] for g in games])),
                white_score=float(np.mean([score_of(g) for g in games if g["a_white"]])),
                black_score=float(np.mean([score_of(g) for g in games if not g["a_white"]])))


def main():
    out = {}
    rr = load(RES / "round_robin" / "matches.jsonl")
    pairs = defaultdict(list)
    for g in rr: pairs[(g["A"], g["B"])].append(g)
    out["pairs"] = {f"{a}-{b}": summarize(v) for (a, b), v in sorted(pairs.items())}
    # độ trễ & nút/mô phỏng theo tác tử
    lat = defaultdict(list); latmax = defaultdict(float); work = defaultdict(lambda: [0.0, 0])
    for g in rr:
        for side in "AB":
            ag = g[side]; lat[ag].append(g[f"lat_{side}_mean"]); latmax[ag] = max(latmax[ag], g[f"lat_{side}_max"])
            work[ag][0] += g[f"nodes_{side}"]; work[ag][1] += g[f"n_moves_{side}"]
    out["latency"] = {k: dict(mean_ms=1000 * float(np.mean(v)), max_ms=1000 * latmax[k],
                              work_per_move=work[k][0] / max(1, work[k][1])) for k, v in lat.items()}
    # tổng điểm vòng tròn
    tot = defaultdict(lambda: [0.0, 0])
    for g in rr:
        tot[g["A"]][0] += score_of(g); tot[g["A"]][1] += 1
        tot[g["B"]][0] += 1 - score_of(g); tot[g["B"]][1] += 1
    out["overall"] = {k: dict(score=v[0] / v[1], n=v[1], elo_vs_field=elo(v[0] / v[1])) for k, v in tot.items()}

    # ma trận điểm
    ks = ["L1", "L2", "L3", "L4"]
    M = np.full((4, 4), np.nan)
    for (a, b), v in pairs.items():
        s = np.mean([score_of(g) for g in v]); M[ks.index(a), ks.index(b)] = s; M[ks.index(b), ks.index(a)] = 1 - s
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            if not np.isnan(M[i, j]): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=11)
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(ks); ax.set_yticklabels(ks)
    ax.set_xlabel("Đối thủ"); ax.set_ylabel("Tác tử (điểm trung bình)"); ax.set_title("Ma trận điểm đối đầu (thắng=1, hòa=0.5)")
    plt.colorbar(im); plt.tight_layout(); plt.savefig(FIG / "matrix_score.png", dpi=160); plt.close()

    # WDL stacked bars
    fig, ax = plt.subplots(figsize=(8, 4))
    labs = list(out["pairs"].keys()); W = [out["pairs"][k]["W"] for k in labs]; D = [out["pairs"][k]["D"] for k in labs]; L = [out["pairs"][k]["L"] for k in labs]
    ax.bar(labs, W, label="Thắng (A)", color="#3b8f5a"); ax.bar(labs, D, bottom=W, label="Hòa", color="#9aa0a6")
    ax.bar(labs, L, bottom=np.array(W) + np.array(D), label="Thua (A)", color="#c0504d")
    ax.set_ylabel("Số ván"); ax.set_title("Thắng/Hòa/Thua theo cặp (A-B, A là tác tử đầu tiên)"); ax.legend(); plt.tight_layout()
    plt.savefig(FIG / "wdl_pairs.png", dpi=160); plt.close()

    # ablation search
    ab = RES / "ablation1" / "ablation_search.json"
    if ab.exists():
        rows = json.load(open(ab, encoding="utf-8")); out["ablation_search"] = rows
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for ax, depth in zip(axs, (2, 3)):
            rr_ = [r for r in rows if r["depth"] == depth]
            ax.barh([r["config"] for r in rr_][::-1], [r["mean_nodes"] for r in rr_][::-1], color="#4472c4")
            ax.set_xscale("log"); ax.set_title(f"Nút trung bình / thế cờ (độ sâu {depth})"); ax.set_xlabel("nút (log)")
        plt.tight_layout(); plt.savefig(FIG / "ablation_search_nodes.png", dpi=160); plt.close()
    wac = RES / "ablation1" / "wac.json"
    if wac.exists():
        w = json.load(open(wac, encoding="utf-8")); out["wac"] = [{k: v for k, v in r.items() if k != "solved_ids"} for r in w]
        fig, ax = plt.subplots(figsize=(8, 4))
        lab = [f"{r['agent']} {list(r['budget'].values())[0] if r['budget'] else ''}" for r in w]
        ax.bar(lab, [r["solved"] for r in w], color="#ed7d31"); ax.set_ylabel("Số thế giải đúng / 50"); ax.set_title("WAC-50: tỷ lệ chọn đúng nước tốt nhất")
        plt.tight_layout(); plt.savefig(FIG / "wac_solved.png", dpi=160); plt.close()

    # ablation MCTS
    am = load(RES / "ablation2" / "matches.jsonl")
    if am:
        bys = defaultdict(list)
        for g in am: bys[g["a_sims"]].append(g)
        out["ablation_mcts"] = {str(k): summarize(v) for k, v in sorted(bys.items())}
        ks_ = sorted(bys); sc = [out["ablation_mcts"][str(k)]["score"] for k in ks_]
        lo = [out["ablation_mcts"][str(k)]["wilson"][0] for k in ks_]; hi = [out["ablation_mcts"][str(k)]["wilson"][1] for k in ks_]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(ks_, sc, yerr=[np.array(sc) - lo, np.array(hi) - sc], marker="o", capsize=4)
        ax.set_xlabel("Số mô phỏng UCT / nước"); ax.set_ylabel("Điểm vs Cấp 2"); ax.set_ylim(0, 1)
        ax.set_title("Ablation 2: UCT theo số mô phỏng (Wilson 95%)"); plt.tight_layout(); plt.savefig(FIG / "ablation_mcts.png", dpi=160); plt.close()

    json.dump(out, open(RES / "summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
