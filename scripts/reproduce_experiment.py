"""
scripts/reproduce_experiment.py
Kịch bản tái lập thực nghiệm tự động hỗ trợ 2 chế độ:
- --from-logs: Tái tạo toàn bộ bảng kết quả, Wilson/Bootstrap CI và biểu đồ từ dữ liệu logs/ (< 30s).
- --full: Chạy lại từ đầu toàn bộ Ablation 1, Ablation 2 và giải đấu Round-Robin 100 ván/cặp.
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):  # Windows (cp1252) không in được tiếng Việt
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _run(*args: str) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def main() -> None:
    """Điểm vào CLI của kịch bản tái lập thực nghiệm."""
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-logs", action="store_true", help="dựng lại bảng/biểu đồ từ log có sẵn")
    g.add_argument("--full", action="store_true", help="chạy lại thí nghiệm Cấp 1-4 từ đầu (lâu)")
    ap.add_argument("--procs", default="2")
    a = ap.parse_args()
    if a.full:
        for cmd in ("openings", "roundrobin", "ablation_search", "wac", "ablation_mcts"):
            _run("scripts/run_real_experiments.py", cmd, "--procs", a.procs)
    _run("scripts/analyze_real_results.py")
    _run("scripts/plot_training.py")


if __name__ == "__main__":
    main()
