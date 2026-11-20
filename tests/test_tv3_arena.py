"""Test arena: metrics, PGN, match, tournament, registry (ván nhỏ, chỉ kiểm tra mã)."""
import json
import chess
import pytest

from chess_ai.agents.registry import create_agent
from chess_ai.arena.match import play_single_match
from chess_ai.arena.metrics import (compute_normalized_score, paired_bootstrap_confidence_interval,
                                    wilson_confidence_interval)
from chess_ai.arena.pgn_exporter import export_game_to_pgn
from chess_ai.arena.tournament import run_round_robin_tournament


def test_metrics():
    assert compute_normalized_score(3, 2, 5) == 0.4
    lo, hi = wilson_confidence_interval(0.5, 100)
    assert abs((lo + hi) / 2 - 0.5) < 1e-9 and 0.39 < lo < 0.41
    lo, hi = paired_bootstrap_confidence_interval([1, 0, 0.5, 1, 0.5] * 10, 500)
    assert 0 <= lo <= hi <= 1


def test_pgn_roundtrip():
    t = export_game_to_pgn([chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")], dict(White="A", Black="B"))
    assert "1. e4 e5" in t


def test_match_and_registry():
    r = play_single_match(create_agent("level1", seed=1), create_agent("level1", seed=2), max_plies=40)
    assert r["num_plies"] <= 40 and r["result"] in (-1.0, 0.0, 1.0)
    with pytest.raises(KeyError):
        create_agent("level99")


def test_tournament(tmp_path):
    s = run_round_robin_tournament([create_agent("level1", seed=1), create_agent("level2", seed=2)],
                                   games_per_pair=4, max_plies=40, output_dir=str(tmp_path))
    (pair,) = s["pairs"].values()
    assert pair["games"] == 4
    assert len((tmp_path / "matches.jsonl").read_text().splitlines()) == 4
