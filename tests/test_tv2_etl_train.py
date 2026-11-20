"""Smoke test ETL -> dataset -> huấn luyện -> Cấp 5. Dữ liệu là ván tự sinh chỉ để kiểm tra mã,
KHÔNG phải số liệu thực nghiệm của báo cáo."""
import io
import json
import random

import chess
import chess.pgn
import numpy as np
import pytest
import torch

from chess_ai.agents.level5_supervised import Level5SupervisedAgent
from chess_ai.core.chess_env import ChessEnv
from chess_ai.etl.memmap_dataset import ChessMemmapDataset
from chess_ai.etl.pgn_etl import estimated_time_sec, run_etl_from_stream, split_of
from chess_ai.training.train_supervised import train_supervised

CFG = dict(min_elo=2100, min_base_time_sec=180, min_plies=30, early_draw_ply_cutoff=60,
           skip_opening_plies=12, subsample_k_min=3, subsample_k_max=4, deduplicate_zobrist=True,
           drop_underpromotions=True, source_name="synthetic-test",
           split_ratios=dict(train=0.8, val=0.1, test=0.1))


def _synthetic_pgn(n_games=60, seed=0):
    rng = random.Random(seed)
    out = io.StringIO()
    made = 0
    while made < n_games:
        b = chess.Board()
        game = chess.pgn.Game()
        node = game
        for _ in range(160):
            if b.is_game_over():
                break
            m = rng.choice(list(b.legal_moves))
            b.push(m)
            node = node.add_variation(m)
        if len(b.move_stack) < 40:
            continue
        game.headers["Site"] = f"https://lichess.org/g{made:06d}"
        game.headers["WhiteElo"] = "2300"; game.headers["BlackElo"] = "2250"
        game.headers["TimeControl"] = "300+0"
        game.headers["Result"] = b.result(claim_draw=True) if b.is_game_over() else "1/2-1/2"
        print(game, file=out, end="\n\n")
        made += 1
    return out.getvalue()


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("proc")
    meta = run_etl_from_stream(io.StringIO(_synthetic_pgn()), str(d), CFG)
    assert meta["counts"]["train"] > 100
    return d


def test_estimated_time_and_split_deterministic():
    assert estimated_time_sec("300+0") == 300 and estimated_time_sec("60+1") == 100
    assert estimated_time_sec("-") is None
    ratios = CFG["split_ratios"]
    assert split_of("abc", ratios) == split_of("abc", ratios)


def test_etl_header_filter_and_no_game_leak(tmp_path):
    pgn = _synthetic_pgn(20, seed=5).replace('"2300"', '"1500"')  # Elo thấp -> bị loại hết
    meta = run_etl_from_stream(io.StringIO(pgn), str(tmp_path), CFG)
    assert sum(meta["counts"].values()) == 0


def test_dataset_shapes_and_values(dataset_dir):
    ds = ChessMemmapDataset(str(dataset_dir), "train")
    x, a, z = ds[0]
    assert x.shape == (18, 8, 8) and x.dtype == torch.float32
    assert 0 <= int(a) < 4096 and float(z) in (-1.0, 0.0, 1.0)
    total = sum(json.loads((dataset_dir / "dataset_metadata.json").read_text())["counts"].values())
    assert total == sum(len(ChessMemmapDataset(str(dataset_dir), s)) for s in ("train", "val", "test"))


def test_train_resume_and_level5_agent(dataset_dir, tmp_path):
    cfgfile = tmp_path / "m.yaml"
    import yaml
    from chess_ai.config import load_yaml_config
    cfg = load_yaml_config("model_resnet.yaml")
    cfg["network"]["num_res_blocks"] = 1; cfg["network"]["filters"] = 16
    cfg["training"]["batch_size"] = 64
    cfgfile.write_text(yaml.safe_dump(cfg))
    res = train_supervised(str(cfgfile), data_dir=str(dataset_dir), checkpoint_dir=str(tmp_path / "ck"),
                           metrics_path=str(tmp_path / "curve.json"), epochs=2, max_steps_per_epoch=5)
    assert len(res["history"]) == 2 and (tmp_path / "ck" / "best_model.pt").exists()
    required = {"epoch", "global_step", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                "scaler_state_dict", "train_loss", "val_policy_acc", "val_value_mse", "best_val_loss",
                "model_config", "random_seed"}
    ck = torch.load(tmp_path / "ck" / "last_supervised.pt", weights_only=False)
    assert required <= set(ck)
    res2 = train_supervised(str(cfgfile), resume_path=str(tmp_path / "ck" / "last_supervised.pt"),
                            data_dir=str(dataset_dir), checkpoint_dir=str(tmp_path / "ck"),
                            metrics_path=str(tmp_path / "curve2.json"), epochs=3, max_steps_per_epoch=5)
    assert [h["epoch"] for h in res2["history"]] == [0, 1, 2]
    agent = Level5SupervisedAgent(checkpoint_path=str(tmp_path / "ck" / "best_model.pt"))
    env = ChessEnv()
    for _ in range(6):
        mv = agent.select_move(env)
        assert mv in env.board.legal_moves
        env.step(mv)


def test_etl_reads_zip(tmp_path):
    import zipfile
    z = tmp_path / "elite.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("lichess_elite_2024-01.pgn", _synthetic_pgn(12, seed=3))
    from chess_ai.etl.pgn_etl import run_etl_pipeline
    import yaml
    from chess_ai.config import load_yaml_config
    cfgp = tmp_path / "c.yaml"
    full = load_yaml_config("default_config.yaml"); full["etl"] = {**full.get("etl", {}), **CFG}
    cfgp.write_text(yaml.safe_dump(full))
    meta = run_etl_pipeline(str(cfgp), str(z), str(tmp_path / "out"), 42, None)
    assert sum(meta["counts"].values()) > 0
