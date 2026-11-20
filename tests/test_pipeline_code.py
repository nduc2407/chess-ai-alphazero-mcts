"""Kiểm tra mã điều phối/biểu đồ/đánh giá KRK (ít mẫu; không phải kết quả thực nghiệm)."""
import subprocess
import sys
from pathlib import Path

from chess_ai.agents.level6_reinforcement import Level6ReinforcementAgent
from chess_ai.arena.tournament import build_agents
from chess_ai.experiments.eval_krk import evaluate_krk_agent

ROOT = Path(__file__).resolve().parents[1]


def test_krk_eval_runs():
    r = evaluate_krk_agent(Level6ReinforcementAgent(table={}), positions=3, max_moves=6)
    assert r["positions"] == 3 and 0.0 <= r["mate_rate"] <= 1.0


def test_build_agents_skips_level6_and_needs_ckpt():
    ags, wl = build_agents(["level1", "level6", "level3"], {}, {"level3": 100})
    assert [a.name for a in ags] == ["Level1_Random", "Level3_AlphaBeta"] and wl == {"Level3_AlphaBeta": 100}
    try:
        build_agents(["level5"], {}, {})
        assert False
    except SystemExit:
        pass


def test_cli_help_and_plot_without_logs():
    import os
    env = dict(os.environ, PYTHONIOENCODING="cp1252")  # giả lập console Windows
    for script in ("scripts/run_full_pipeline.py", "scripts/reproduce_experiment.py"):
        r = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, capture_output=True, env=env)
        assert r.returncode == 0, r.stderr.decode(errors="replace")
    assert subprocess.run([sys.executable, "scripts/plot_training.py"], cwd=ROOT, capture_output=True).returncode == 0
