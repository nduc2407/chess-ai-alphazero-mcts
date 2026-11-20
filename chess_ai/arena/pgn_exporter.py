"""
chess_ai/arena/pgn_exporter.py
Module xuất biên bản ván đấu chuẩn PGN quốc tế và ghi dòng log JSONL.
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

import json
from pathlib import Path
from typing import Any, Dict, Sequence
import chess
import chess.pgn


def export_game_to_pgn(
    moves: Sequence[chess.Move],
    headers: Dict[str, str],
    initial_fen: str = chess.STARTING_FEN,
) -> str:
    """Chuyển đổi danh sách nước đi và metadata thành chuỗi biên bản PGN chuẩn."""
    board = chess.Board(initial_fen)
    game = chess.pgn.Game()
    for k, v in headers.items():
        game.headers[k] = str(v)
    if initial_fen != chess.STARTING_FEN:
        game.headers["FEN"] = initial_fen
        game.headers["SetUp"] = "1"
    node = game
    for m in moves:
        node = node.add_variation(m)
    return str(game)


def append_match_jsonl(filepath: str, record: Dict[str, Any]) -> None:
    """Ghi nối tiếp kết quả một ván đấu vào tệp JSONL."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

