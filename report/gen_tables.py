"""Sinh các bảng LaTeX từ results/summary.json và log trận đấu (không gõ tay số liệu)."""
import json, chess
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "results"
G = Path(__file__).resolve().parent / "generated"
S = json.load(open(R / "summary.json"))
NM = {"L1": "Cấp 1", "L2": "Cấp 2", "L3": "Cấp 3", "L4": "Cấp 4"}
f3 = lambda x: f"{x:.3f}".replace(".", "{,}")  # dùng dấu phẩy thập phân
def w(name, s): (G / name).write_text(s + "\n", encoding="utf-8")
def tab(name, spec, head, body): w(name, "\\begin{tabular}{" + spec + "}\n\\toprule\n" + head + " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}")

# Bảng cặp đối đầu
rows = []
for k, v in S["pairs"].items():
    a, b = k.split("-")
    rows.append(f"{NM[a]} -- {NM[b]} & {v['n']} & {v['W']} / {v['D']} / {v['L']} & {f3(v['score'])} & {{[{f3(v['wilson'][0])}; {f3(v['wilson'][1])}]}} & {{[{f3(v['boot'][0])}; {f3(v['boot'][1])}]}} & {v['maxply_draws']} & {v["mean_plies"]:.1f} \\\\".replace(".", ",").replace("{,}", ","))
tab("pairs.tex","l r c c c c r r","Cặp & $N$ & W/D/L (A) & Điểm A & Wilson 95\\% & Bootstrap 95\\% & Hòa cap & Plies TB","\n".join(rows))

# Độ trễ
lat = S["latency"]; ov = S["overall"]
rows = []
for k in ["L1", "L2", "L3", "L4"]:
    l = lat[k]
    rows.append(f"{NM[k]} & {l['mean_ms']:.2f} & {l['max_ms']:.1f} & {l['work_per_move']:.1f} & {ov[k]['score']:.3f} \\\\".replace(".", ","))
tab("latency.tex","l r r r r","Tác tử & Trễ TB & Trễ tối đa & Nút/mô phỏng TB & Điểm toàn giải","\n".join(rows))

# Ablation search
ab = S["ablation_search"]; rows = []
for d in (2, 3):
    base = [r for r in ab if r["depth"] == d][0]["mean_nodes"]
    for r in [r for r in ab if r["depth"] == d]:
        rows.append(f"{d} & {r['config']} & {r['mean_nodes']:.0f} & {(base / r["mean_nodes"]):.1f}$\\times$ & {r['total_time_sec']:.2f} & {r['solved']}/{r['n']} \\\\".replace(".", ",").replace("{,}", ",").replace("Minimax thuần (no AB, no order, no TT, no Q)", "Minimax thuần"))
tab("ablation_search.tex","c l r r r r","$d$ & Cấu hình & Nút TB & Giảm & Tổng thời gian (s) & Giải đúng","\n".join(rows))

# WAC
rows = []
for r in S["wac"]:
    b = list(r["budget"].values())[0] if r["budget"] else "---"
    unit = "nút" if r["agent"] == "L3" else ("mô phỏng" if r["agent"] == "L4" else "")
    rows.append(f"{NM[r['agent']]} & {b} {unit} & {r['solved']}/{r['n']} & {r['mean_time_sec']*1000:.2f} \\\\".replace(".", ","))
tab("wac.tex","l l r r","Tác tử & Ngân sách & Giải đúng & Trễ TB (ms)","\n".join(rows))

# Ablation MCTS
rows = []
for k, v in S["ablation_mcts"].items():
    rows.append(f"{k} & {v['n']} & {v['W']} / {v['D']} / {v['L']} & {v['score']:.3f} & {{[{v['wilson'][0]:.3f}; {v['wilson'][1]:.3f}]}} & {v['mean_plies']:.1f} \\\\".replace(".", ","))
tab("ablation_mcts.tex","r r c c c r","Mô phỏng & $N$ & W/D/L (UCT) & Điểm & Wilson 95\\% & Plies TB","\n".join(rows))

# Ván minh họa: quy tắc cố định = khai cuộc đầu tiên trong openings_generated.epd, tác tử A cầm Trắng
ops = [l.strip() for l in open(R / "round_robin" / "openings_generated.epd") if l.strip() and not l.startswith("#")]
fen0 = ops[0]
games = [json.loads(l) for l in open(R / "round_robin" / "matches.jsonl")]
def fmt(g):
    b = chess.Board(fen0); out = []
    for i, san in enumerate(g["san"]):
        if b.turn == chess.WHITE: out.append(f"{b.fullmove_number}. {san}")
        elif i == 0: out.append(f"{b.fullmove_number}... {san}")
        else: out.append(san)
        b.push(b.parse_san(san))
    return " ".join(out)
for A, B in [("L3", "L4"), ("L2", "L3")]:
    g = [x for x in games if x["A"] == A and x["B"] == B and x["fen"] == fen0 and x["a_white"]][0]
    w(f"game_{A}_{B}.tex", f"Trắng: {NM[A]}, Đen: {NM[B]}; kết quả (góc nhìn {NM[A]}): {g['a_result']}, {g['plies']} plies.\\\\\n" + fmt(g).replace("#", "\\#"))
w("fen0.tex", fen0)
print("ok")
