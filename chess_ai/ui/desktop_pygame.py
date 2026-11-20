"""
chess_ai/ui/desktop_pygame.py
Giao diện bàn cờ Pygame Desktop tách luồng/tiến trình tính toán AI độc lập,
hỗ trợ kéo thả quân cờ, đồng hồ thi đấu, biên bản SAN và chọn đủ 8 cấp độ AI.
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)

Phím: 1-8 chọn cấp AI cho bên Đen, N ván mới, U hoàn tác một cặp nước, F đổi bên người chơi, Esc thoát.
Quân cờ vẽ bằng hình tròn + chữ cái (không phụ thuộc phông có ký tự cờ vua).
Cấp 5/7/8 cần checkpoint, cấp 6 cần bảng Q (chỉ chơi được ở thế KRK).
"""
import threading
import time
from typing import Any, Dict, Optional, Tuple

import chess

from chess_ai.agents.registry import create_agent
from chess_ai.core.chess_env import ChessEnv

SQ = 72
BOARD_PX = SQ * 8
SIDE = 260
LIGHT, DARK = (238, 238, 210), (118, 150, 86)
SEL, LAST = (246, 246, 105), (205, 210, 106)


class ChessApp:
    """Trạng thái + logic giao diện; tách khỏi vòng lặp vẽ để kiểm thử không cần cửa sổ."""

    def __init__(self, checkpoints: Optional[Dict[str, str]] = None, ai_level: str = "level2",
                 human_white: bool = True, ai_workload: int = 200) -> None:
        self.checkpoints = checkpoints or {}
        self.env = ChessEnv()
        self.human_white = human_white
        self.selected: Optional[int] = None
        self.ai_level = ai_level
        self.ai_workload = ai_workload
        self.agent = None
        self.status = ""
        self._thinking = False
        self._pending: Optional[chess.Move] = None
        self._lock = threading.Lock()
        self.clock = {chess.WHITE: 0.0, chess.BLACK: 0.0}
        self._tick = time.perf_counter()
        self.set_level(ai_level)

    # ---- cấu hình ----
    def set_level(self, level: str) -> None:
        kw: Dict[str, Any] = {}
        if level in ("level5", "level7", "level8"):
            if level not in self.checkpoints:
                self.status = f"{level}: chưa có checkpoint (--ckpt {level}=đường_dẫn)"
                return
            kw["checkpoint_path"] = self.checkpoints[level]
        if level == "level6":
            if "level6" not in self.checkpoints:
                self.status = "level6: chưa có bảng Q (chỉ chơi được thế KRK)"
                return
            kw["q_table_path"] = self.checkpoints["level6"]
        self.agent = create_agent(level, **kw)
        self.ai_level = level
        self.status = f"AI: {self.agent.name}"

    def new_game(self) -> None:
        with self._lock:
            self.env = ChessEnv(); self.selected = None; self._pending = None
            self.clock = {chess.WHITE: 0.0, chess.BLACK: 0.0}
            if self.agent:
                self.agent.reset()

    def undo(self) -> None:
        with self._lock:
            if self._thinking:
                return
            for _ in range(2):
                if self.env.board.move_stack:
                    self.env.undo()
            self.selected = None

    # ---- vòng lặp logic ----
    @property
    def human_color(self) -> bool:
        return chess.WHITE if self.human_white else chess.BLACK

    def click_square(self, sq: int) -> Optional[chess.Move]:
        """Xử lý chọn/đi quân của người; trả về nước đã đi (nếu có)."""
        b = self.env.board
        if self._thinking or b.turn != self.human_color or self.env.is_terminal():
            return None
        if self.selected is None:
            p = b.piece_at(sq)
            if p and p.color == b.turn:
                self.selected = sq
            return None
        mv = chess.Move(self.selected, sq)
        if b.piece_at(self.selected) and b.piece_at(self.selected).piece_type == chess.PAWN \
                and chess.square_rank(sq) in (0, 7):
            mv.promotion = chess.QUEEN
        if mv in b.legal_moves:
            self.env.step(mv); self.selected = None
            return mv
        p = b.piece_at(sq)
        self.selected = sq if p and p.color == b.turn else None
        return None

    def update(self) -> None:
        """Gọi mỗi khung hình: cập nhật đồng hồ, khởi chạy AI nền và áp nước AI đã tính xong."""
        now = time.perf_counter()
        if not self.env.is_terminal():
            self.clock[self.env.board.turn] += now - self._tick
        self._tick = now
        with self._lock:
            mv, self._pending = self._pending, None
        if mv is not None:
            if mv in self.env.board.legal_moves:
                self.env.step(mv)
            self._thinking = False
        b = self.env.board
        if (not self._thinking and self.agent and not self.env.is_terminal() and b.turn != self.human_color):
            self._thinking = True
            env_copy = self.env.clone()
            threading.Thread(target=self._think, args=(env_copy,), daemon=True).start()

    def _think(self, env: ChessEnv) -> None:
        try:
            mv = self.agent.select_move(env, max_workload=self.ai_workload)
        except Exception as e:  # ví dụ cấp 6 ngoài thế KRK
            mv = None
            self.status = f"Lỗi AI: {e}"
        with self._lock:
            if mv is None:
                self._thinking = False
            else:
                self._pending = mv

    def san_list(self) -> list:
        b = chess.Board()
        out = []
        for m in self.env.board.move_stack:
            out.append(b.san(m)); b.push(m)
        return out

    def result_text(self) -> str:
        if not self.env.is_terminal():
            return ""
        r = self.env.result()
        return "Trắng thắng" if r == 1.0 else "Đen thắng" if r == -1.0 else "Hòa"


def _square_at(pos: Tuple[int, int], flip: bool) -> Optional[int]:
    x, y = pos
    if not (0 <= x < BOARD_PX and 0 <= y < BOARD_PX):
        return None
    f, r = x // SQ, 7 - y // SQ
    if flip:
        f, r = 7 - f, 7 - r
    return chess.square(f, r)


def launch_desktop_app(checkpoints: Optional[Dict[str, str]] = None, ai_level: str = "level2") -> None:
    """Khởi chạy cửa sổ ứng dụng cờ vua Pygame."""
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((BOARD_PX + SIDE, BOARD_PX))
    pygame.display.set_caption("Game AI Cờ Vua - Nhóm 27")
    font = pygame.font.SysFont(None, 22)
    big = pygame.font.SysFont(None, 40, bold=True)
    app = ChessApp(checkpoints, ai_level)
    clk = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_n:
                    app.new_game()
                elif ev.key == pygame.K_u:
                    app.undo()
                elif ev.key == pygame.K_f:
                    app.human_white = not app.human_white; app.new_game()
                elif pygame.K_1 <= ev.key <= pygame.K_8:
                    app.set_level(f"level{ev.key - pygame.K_0}")
            elif ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP) and ev.button == 1:
                sq = _square_at(ev.pos, not app.human_white)
                # nhấp-nhấp hoặc kéo-thả: MOUSEUP trên ô khác ô đã chọn cũng đi quân
                if sq is not None and (ev.type == pygame.MOUSEBUTTONDOWN or sq != app.selected):
                    app.click_square(sq)
        app.update()
        _draw(screen, app, font, big, pygame)
        pygame.display.flip()
        clk.tick(30)
    pygame.quit()


def _draw(screen, app: ChessApp, font, big, pygame) -> None:
    flip = not app.human_white
    b = app.env.board
    last = b.peek() if b.move_stack else None
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x, y = (7 - f if flip else f) * SQ, (r if flip else 7 - r) * SQ
        col = LIGHT if (f + r) % 2 else DARK
        if last and sq in (last.from_square, last.to_square):
            col = LAST
        if sq == app.selected:
            col = SEL
        pygame.draw.rect(screen, col, (x, y, SQ, SQ))
        p = b.piece_at(sq)
        if p:
            c = (250, 250, 250) if p.color == chess.WHITE else (30, 30, 30)
            pygame.draw.circle(screen, c, (x + SQ // 2, y + SQ // 2), SQ // 2 - 8)
            pygame.draw.circle(screen, (90, 90, 90), (x + SQ // 2, y + SQ // 2), SQ // 2 - 8, 2)
            t = big.render(p.symbol().upper(), True, (30, 30, 30) if p.color == chess.WHITE else (250, 250, 250))
            screen.blit(t, t.get_rect(center=(x + SQ // 2, y + SQ // 2)))
    if app.selected is not None:
        for m in b.legal_moves:
            if m.from_square == app.selected:
                f, r = chess.square_file(m.to_square), chess.square_rank(m.to_square)
                x, y = (7 - f if flip else f) * SQ, (r if flip else 7 - r) * SQ
                pygame.draw.circle(screen, (70, 70, 70), (x + SQ // 2, y + SQ // 2), 8)
    pygame.draw.rect(screen, (40, 40, 44), (BOARD_PX, 0, SIDE, BOARD_PX))
    lines = [app.status, f"Trắng {app.clock[chess.WHITE]:6.1f}s", f"Đen   {app.clock[chess.BLACK]:6.1f}s",
             app.result_text() or ("AI đang nghĩ..." if app._thinking else "Lượt người" if b.turn == app.human_color else ""),
             "1-8 cấp AI | N ván mới | U hoàn tác | F đổi bên"]
    for i, ln in enumerate(lines):
        screen.blit(font.render(ln, True, (230, 230, 230)), (BOARD_PX + 10, 10 + i * 24))
    sans = app.san_list()
    for i in range(0, len(sans), 2):
        row = f"{i // 2 + 1}. {sans[i]} {sans[i + 1] if i + 1 < len(sans) else ''}"
        k = i // 2
        if k >= 16:
            row_y = 150 + (k - 16) * 20; x0 = BOARD_PX + 140
        else:
            row_y = 150 + k * 20; x0 = BOARD_PX + 10
        if row_y < BOARD_PX - 20:
            screen.blit(font.render(row, True, (200, 200, 200)), (x0, row_y))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="level2")
    ap.add_argument("--ckpt", nargs="*", default=[], help="level5=path level7=path level8=path level6=qtable.pkl")
    a = ap.parse_args()
    launch_desktop_app(dict(x.split("=", 1) for x in a.ckpt), a.level)
