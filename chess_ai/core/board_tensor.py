"""
chess_ai/core/board_tensor.py
Biểu diễn trạng thái bàn cờ thành Tensor chuẩn hóa góc nhìn bên đi (18, 8, 8)
và ánh xạ không gian hành động phẳng 4096 chiều (ADR-02 & ADR-03).

Kênh (theo configs/model_resnet.yaml):
  0-5   : P, N, B, R, Q, K của bên tới lượt đi
  6-11  : P, N, B, R, Q, K của đối thủ
  12-15 : quyền nhập thành Active_KS, Active_QS, Opp_KS, Opp_QS
  16    : ô đích bắt tốt qua đường hợp lệ
  17    : min(halfmove_clock / 100, 1)

Hành động: u_canon * 64 + v_canon, với ô đã lật theo góc nhìn bên đi (sq ^ 56 khi Đen đi).
Phong cấp: chỉ giữ một chỉ số cho mỗi cặp (từ, đến) và giải mã mặc định phong Hậu.
"""
import chess
import numpy as np

ACTION_SPACE = 4096
_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def canonical_square(sq: int, turn: chess.Color) -> int:
    """Lật tọa độ ô cờ theo góc nhìn bên đi (sq ^ 56 nếu Đen đi)."""
    return sq if turn == chess.WHITE else sq ^ 56


def _fill_planes(board: chess.Board, dtype) -> np.ndarray:
    planes = np.zeros((18, 8, 8), dtype=dtype)
    turn = board.turn
    for sq, piece in board.piece_map().items():
        c = canonical_square(sq, turn)
        base = 0 if piece.color == turn else 6
        planes[base + _PIECE_TYPES.index(piece.piece_type), c >> 3, c & 7] = 1
    opp = not turn
    planes[12][:] = board.has_kingside_castling_rights(turn)
    planes[13][:] = board.has_queenside_castling_rights(turn)
    planes[14][:] = board.has_kingside_castling_rights(opp)
    planes[15][:] = board.has_queenside_castling_rights(opp)
    if board.ep_square is not None and board.has_legal_en_passant():
        c = canonical_square(board.ep_square, turn)
        planes[16, c >> 3, c & 7] = 1
    return planes


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Chuyển đổi chess.Board thành mảng numpy shape (18, 8, 8) dtype float32."""
    planes = _fill_planes(board, np.float32)
    planes[17][:] = min(board.halfmove_clock / 100.0, 1.0)
    return planes


def board_to_uint8_planes(board: chess.Board) -> np.ndarray:
    """Mảng (18, 8, 8) uint8 để lưu np.memmap; kênh 17 lưu halfmove_clock (0-100) thay vì thực số."""
    planes = _fill_planes(board, np.uint8)
    planes[17][:] = min(board.halfmove_clock, 100)
    return planes


def uint8_planes_to_tensor(planes: np.ndarray) -> np.ndarray:
    """Khôi phục float32 từ mảng uint8 (chia kênh 17 cho 100)."""
    out = planes.astype(np.float32)
    out[17] = np.minimum(out[17] / 100.0, 1.0)
    return out


def move_to_action_index(move: chess.Move, turn: chess.Color) -> int:
    """Ánh xạ nước đi sang chỉ số nguyên phẳng [0, 4095] theo góc nhìn bên đi."""
    return canonical_square(move.from_square, turn) * 64 + canonical_square(move.to_square, turn)


def action_index_to_move(action_idx: int, board: chess.Board) -> chess.Move:
    """Giải mã chỉ số [0, 4095] về chess.Move hợp lệ (mặc định phong Hậu).

    Nếu chỉ số không ứng với nước hợp lệ nào, trả về nước hợp lệ đầu tiên (fallback).
    """
    turn = board.turn
    u = canonical_square(action_idx // 64, turn)
    v = canonical_square(action_idx % 64, turn)
    move = chess.Move(u, v)
    if board.piece_type_at(u) == chess.PAWN and chess.square_rank(v) in (0, 7):
        move = chess.Move(u, v, promotion=chess.QUEEN)
    if move in board.legal_moves:
        return move
    return next(iter(board.legal_moves))


def get_legal_action_mask(board: chess.Board) -> np.ndarray:
    """Mặt nạ nhị phân shape (4096,) đánh dấu các chỉ số hành động hợp lệ."""
    mask = np.zeros(ACTION_SPACE, dtype=np.float32)
    turn = board.turn
    for m in board.legal_moves:
        mask[move_to_action_index(m, turn)] = 1.0
    return mask
