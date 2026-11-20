"""
chess_ai/etl/pgn_etl.py
Đường ống ETL đọc Lichess Elite Database (.pgn, .pgn.zst hoặc .zip), lọc ở mức header
(ELO, kiểm soát thời gian), lấy mẫu cách quãng k in {3, 4}, khử trùng lặp thế cờ bằng Zobrist 64-bit,
chia tách theo ván tất định qua MD5(Game ID) và ghi np.memmap-ready binary.

Đầu ra trong output_dir:
  {train,val,test}_states.dat   uint8   (N, 18, 8, 8)
  {train,val,test}_actions.dat  int16   (N,)
  {train,val,test}_values.dat   int8    (N,)   giá trị z in {-1, 0, +1} theo góc nhìn bên đi
  dataset_metadata.json
"""
import argparse
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import chess
import chess.pgn
import chess.polyglot

from chess_ai.config import load_yaml_config
from chess_ai.core.board_tensor import board_to_uint8_planes, move_to_action_index

SPLITS = ("train", "val", "test")


def _open_text(path: str):
    if path.endswith(".zip"):  # Lichess Elite phát hành dạng .zip chứa một file .pgn
        import zipfile

        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if n.lower().endswith(".pgn")]
        if not names:
            raise ValueError(f"Không thấy file .pgn trong {path}")
        return io.TextIOWrapper(zf.open(names[0]), encoding="utf-8", errors="replace")
    if path.endswith(".zst"):
        import zstandard

        fh = open(path, "rb")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def estimated_time_sec(time_control: str) -> Optional[float]:
    """base + 40 * increment; None nếu không có thông tin (ví dụ '-')."""
    try:
        base, inc = time_control.split("+")
        return float(base) + 40.0 * float(inc)
    except Exception:
        return None


def game_id_of(game: chess.pgn.Game, fallback_index: int) -> str:
    site = game.headers.get("Site", "")
    if site and site != "?":
        return site.rstrip("/").split("/")[-1]
    return game.headers.get("GameId", f"game{fallback_index}")


def split_of(game_id: str, ratios: Dict[str, float]) -> str:
    """Chia tập tất định theo MD5(game_id)."""
    h = int(hashlib.md5(game_id.encode("utf-8")).hexdigest(), 16) % 10_000 / 10_000.0
    if h < ratios["train"]:
        return "train"
    if h < ratios["train"] + ratios["val"]:
        return "val"
    return "test"


def passes_header_filter(game: chess.pgn.Game, cfg: Dict[str, Any]) -> bool:
    h = game.headers
    try:
        if min(int(h.get("WhiteElo", 0)), int(h.get("BlackElo", 0))) < cfg["min_elo"]:
            return False
    except ValueError:
        return False
    t = estimated_time_sec(h.get("TimeControl", "-"))
    if t is None or t < cfg["min_base_time_sec"]:
        return False
    return h.get("Result") in ("1-0", "0-1", "1/2-1/2")


def iter_samples(game: chess.pgn.Game, cfg: Dict[str, Any], rng: random.Random) -> Iterator[Dict[str, Any]]:
    """Sinh các mẫu (board, move, z) từ một ván theo quy tắc lấy mẫu cách quãng."""
    moves = list(game.mainline_moves())
    n = len(moves)
    if n < cfg["min_plies"]:
        return
    result = game.headers["Result"]
    board = game.board()
    # loại ván hòa sớm không bắt quân
    if result == "1/2-1/2" and n < cfg["early_draw_ply_cutoff"]:
        b = game.board()
        captured = False
        for m in moves:
            captured |= b.is_capture(m)
            b.push(m)
        if not captured:
            return
    z_white = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[result]
    next_ply = cfg["skip_opening_plies"]
    for ply, mv in enumerate(moves):
        if ply >= next_ply:
            keep = True
            if cfg.get("drop_underpromotions", True) and mv.promotion not in (None, chess.QUEEN):
                keep = False
            if keep:
                z = z_white if board.turn == chess.WHITE else -z_white
                yield dict(board=board.copy(stack=False), move=mv, z=z)
            next_ply = ply + rng.randint(cfg["subsample_k_min"], cfg["subsample_k_max"])
        board.push(mv)


def run_etl_from_stream(stream, output_dir: str, cfg: Dict[str, Any], seed: int = 42,
                        max_samples: Optional[int] = None, max_games: Optional[int] = None) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    files = {s: {k: open(out / f"{s}_{k}.dat", "wb") for k in ("states", "actions", "values")} for s in SPLITS}
    counts = {s: 0 for s in SPLITS}
    games_per_split = {s: 0 for s in SPLITS}
    seen = set()
    stats = dict(games_read=0, games_kept=0, duplicates_dropped=0)
    import numpy as np

    idx = 0
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        stats["games_read"] += 1
        idx += 1
        if not passes_header_filter(game, cfg):
            continue
        split = split_of(game_id_of(game, idx), cfg["split_ratios"])
        wrote = False
        for s in iter_samples(game, cfg, rng):
            key = chess.polyglot.zobrist_hash(s["board"])
            if cfg.get("deduplicate_zobrist", True):
                if key in seen:
                    stats["duplicates_dropped"] += 1
                    continue
                seen.add(key)
            files[split]["states"].write(board_to_uint8_planes(s["board"]).tobytes())
            files[split]["actions"].write(np.int16(move_to_action_index(s["move"], s["board"].turn)).tobytes())
            files[split]["values"].write(np.int8(s["z"]).tobytes())
            counts[split] += 1
            wrote = True
        if wrote:
            games_per_split[split] += 1
            stats["games_kept"] += 1
        total = sum(counts.values())
        if (max_samples and total >= max_samples) or (max_games and stats["games_kept"] >= max_games):
            break
    for s in SPLITS:
        for f in files[s].values():
            f.close()
    meta = dict(
        source=cfg.get("source_name"), seed=seed, config=cfg, counts=counts,
        games_per_split=games_per_split, stats=stats,
        shapes=dict(states=[18, 8, 8], actions=[], values=[]),
        dtypes=dict(states="uint8", actions="int16", values="int8"),
    )
    (out / "dataset_metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return meta


def run_etl_pipeline(
    config_path: str = "configs/default_config.yaml",
    input_path: str = "data/raw/lichess_elite.pgn.zst",
    output_dir: str = "data/processed",
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Chạy toàn bộ quy trình ETL ngoại tuyến và xuất dataset_metadata.json."""
    cfg = load_yaml_config(config_path)["data_etl"]
    with _open_text(input_path) as stream:
        return run_etl_from_stream(stream, output_dir, cfg, seed=seed,
                                   max_samples=max_samples or cfg.get("target_samples"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ETL Lichess Elite -> dữ liệu huấn luyện Cấp 5")
    ap.add_argument("--config", default="configs/default_config.yaml")
    ap.add_argument("--input", default="data/raw/lichess_elite.pgn.zst")
    ap.add_argument("--output", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=None)
    a = ap.parse_args()
    m = run_etl_pipeline(a.config, a.input, a.output, a.seed, a.max_samples)
    print(json.dumps({"counts": m["counts"], "stats": m["stats"]}, indent=1))
