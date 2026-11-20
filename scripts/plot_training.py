"""
scripts/plot_training.py
Vẽ đường cong học từ log huấn luyện thật: supervised_curve.json (Cấp 5), krk_learning_curve.json (Cấp 6),
selfplay_log.json (Cấp 8) và ablation Cấp 7. Bỏ qua file chưa tồn tại.
"""
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):  # Windows (cp1252) không in được tiếng Việt
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
TR, FIG = ROOT / "results" / "training", ROOT / "results" / "figures"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def plot_supervised():
    d = _load(TR / "supervised_curve.json")
    if not d:
        return None
    h = d["history"]; ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [r["train_loss"] for r in h], label="train"); ax[0].plot(ep, [r["val_loss"] for r in h], label="val")
    ax[0].set(title="Loss tổng", xlabel="epoch"); ax[0].legend()
    ax[1].plot(ep, [r["val_policy_acc"] for r in h], label="top-1"); ax[1].plot(ep, [r["val_top3"] for r in h], label="top-3")
    ax[1].set(title="Độ chính xác policy (val)", xlabel="epoch"); ax[1].legend()
    return _save(fig, "supervised_curve.png")


def plot_krk():
    d = _load(TR / "krk_learning_curve.json")
    if not d:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for algo in ("q_learning", "reinforce"):
        pts = [c for c in d["curve"] if c["algo"] == algo]
        ax.plot([c["episode"] for c in pts], [c["win_rate"] for c in pts], marker="o", ms=3, label=algo)
    ax.set(title="Cấp 6: tỷ lệ chiếu hết KRK", xlabel="tập huấn luyện", ylabel="tỷ lệ"); ax.legend()
    return _save(fig, "krk_learning_curve.png")


def plot_selfplay():
    d = _load(TR / "selfplay_log.json")
    if not d:
        return None
    it = d["iterations"]; x = [r["iteration"] for r in it]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(x, [r["train_loss"] for r in it]); ax[0].set(title="Loss huấn luyện", xlabel="vòng lặp")
    ax[1].plot(x, [r["gating"]["score"] for r in it], marker="o")
    ax[1].axhline(d["config"]["arena_win_threshold"], ls="--", c="gray")
    ax[1].set(title="Điểm gating vs champion", xlabel="vòng lặp", ylim=(0, 1))
    return _save(fig, "selfplay_gating.png")


def plot_ablation7():
    d = _load(ROOT / "results" / "ablation2" / "ablation2_results.json")
    if not d or "results" not in d:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for ag in sorted({r["agent"] for r in d["results"]}):
        pts = [r for r in d["results"] if r["agent"] == ag]
        ax.errorbar([r["sims"] for r in pts], [r["score"] for r in pts],
                    yerr=[[r["score"] - r["wilson_ci"][0] for r in pts], [r["wilson_ci"][1] - r["score"] for r in pts]],
                    marker="o", capsize=3, label=ag)
    ax.set(title="Độ nhạy số mô phỏng (vs Cấp 2)", xlabel="N_sim", ylabel="điểm", xscale="log"); ax.legend()
    return _save(fig, "ablation_sims_L4_L7.png")


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / name, dpi=160); plt.close(fig)
    return FIG / name


if __name__ == "__main__":
    out = [f() for f in (plot_supervised, plot_krk, plot_selfplay, plot_ablation7)]
    print("\n".join(str(p) for p in out if p) or "Chưa có log nào để vẽ.", file=sys.stdout)
