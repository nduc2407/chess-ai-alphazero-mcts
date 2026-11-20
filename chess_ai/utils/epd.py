"""
chess_ai/utils/epd.py
Bộ đọc tệp thế cờ chuẩn EPD dùng chung cho bộ 50 thế cờ chiến thuật WAC-50
(resources/benchmark_positions.epd) và bộ 50 thế cờ khai cuộc cân bằng (resources/openings_50.epd).
Phụ trách: TV3 (UI, Arena, QA & Analytics Lead)
"""

from dataclasses import dataclass
from typing import List
import chess


@dataclass
class EPDRecord:
    """Thông tin một dòng thế cờ EPD đã phân tích cú pháp."""
    record_id: str
    fen: str
    best_moves: List[chess.Move]


def load_epd_file(filepath: str) -> List[EPDRecord]:
    """Đọc tệp .epd và trả về danh sách các bản ghi EPDRecord hợp lệ."""
    records: List[EPDRecord] = []
    with open(filepath, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            fen4 = " ".join(parts[:4])
            ops = {}
            for op in " ".join(parts[4:]).split(";"):
                op = op.strip()
                if op:
                    k, _, v = op.partition(" ")
                    ops[k] = v.strip()
            board = chess.Board(fen4 + " 0 1")
            best: List[chess.Move] = []
            for san in ops.get("bm", "").split():
                try:
                    best.append(board.parse_san(san))
                except ValueError:
                    pass
            records.append(EPDRecord(ops.get("id", "").strip('"') or f"line{n}", board.fen(), best))
    return records

