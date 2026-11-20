"""Test UI (không cần cửa sổ thật), ablation tìm kiếm, EPD."""
import os
import time
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import chess

from chess_ai.experiments.ablation_search import run_ablation_search
from chess_ai.ui.desktop_pygame import ChessApp, _draw, _square_at
from chess_ai.utils.epd import load_epd_file


def test_app_human_then_ai_move():
    app = ChessApp(ai_level="level1")
    assert app.click_square(chess.E2) is None and app.selected == chess.E2
    assert app.click_square(chess.E4) is not None
    for _ in range(100):
        app.update(); time.sleep(0.01)
        if len(app.env.board.move_stack) == 2:
            break
    assert len(app.env.board.move_stack) == 2 and app.san_list()[0] == "e4"
    app.undo(); assert not app.env.board.move_stack


def test_unavailable_level_reports_status():
    app = ChessApp(); app.set_level("level5")
    assert "checkpoint" in app.status


def test_draw_headless_and_square_map():
    import pygame
    pygame.init()
    s = pygame.display.set_mode((800, 576))
    app = ChessApp()
    _draw(s, app, pygame.font.SysFont(None, 20), pygame.font.SysFont(None, 30), pygame.pygame if False else pygame)
    assert _square_at((5, 5), False) == chess.A8 and _square_at((5, 5), True) == chess.H1


def test_ablation_search_small(tmp_path):
    p = tmp_path / "a.json"
    rows = run_ablation_search(output_path=str(p), depths=(1,))
    assert len(rows) == 5 and p.exists()
    assert rows[1]["total_nodes"] <= rows[0]["total_nodes"]  # alpha-beta không nhiều nút hơn minimax
    assert len(load_epd_file("resources/benchmark_positions.epd")) == 50
