"""
scripts/run_full_pipeline.py
Điều phối toàn bộ quy trình Cấp 5-8 bằng một lệnh con cho mỗi bước (mỗi bước độc lập, chạy theo thứ tự):

  krk        Cấp 6: huấn luyện Q-learning + REINFORCE (20000 tập) rồi đánh giá KRK
  etl        Cấp 5: PGN (.pgn/.pgn.zst) -> data/processed
  train5     Cấp 5: huấn luyện có giám sát -> checkpoints/best_model.pt
  ablation7  Cấp 4 vs Cấp 7 theo N_sim và bóc tách thành phần (cần checkpoint)
  selfplay   Cấp 8: self-play + gating -> checkpoints/best_az_model.pt
  tournament Giải vòng tròn Cấp 1-8
  figures    Vẽ biểu đồ từ log

Ví dụ:
  python scripts/run_full_pipeline.py etl --pgn data/raw/lichess_elite_2024-01.pgn.zst
  python scripts/run_full_pipeline.py train5
Chỉ báo cáo số liệu sinh ra từ lần chạy thật của các bước này.
"""
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):  # Windows (cp1252) không in được tiếng Việt
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def cmd_krk(a):
    from chess_ai.experiments.eval_krk import run_krk_evaluation
    from chess_ai.training.train_rl_krk import train_krk_rl
    train_krk_rl(episodes=a.episodes)
    print(run_krk_evaluation(positions=a.positions))


def cmd_etl(a):
    from chess_ai.etl.pgn_etl import run_etl_pipeline
    run_etl_pipeline("configs/default_config.yaml", a.pgn, a.out, 42, a.max_samples or None)


def cmd_train5(a):
    from chess_ai.training.train_supervised import train_supervised
    train_supervised(resume_path=a.resume, data_dir=a.data, epochs=a.epochs,
                     max_steps_per_epoch=a.max_steps, num_workers=a.workers)


def cmd_ablation7(a):
    from chess_ai.experiments.ablation_mcts import run_ablation_components, run_ablation_mcts
    run_ablation_mcts(checkpoint_path=a.ckpt, games_per_point=a.games)
    run_ablation_components(a.ckpt, games_per_pair=a.games)


def cmd_selfplay(a):
    from chess_ai.training.selfplay import run_selfplay_pipeline
    ov = {k: v for k, v in dict(selfplay_games_per_iter=a.games, mcts_sims_per_move=a.sims,
                                arena_eval_games=a.arena_games, max_plies=a.max_plies,
                                train_steps_per_iter=a.train_steps).items() if v is not None}
    run_selfplay_pipeline(warm_start_checkpoint=a.ckpt, iterations=a.iterations, overrides=ov)


def cmd_tournament(a):
    from chess_ai.arena.tournament import build_agents, run_round_robin_tournament
    ck = dict(x.split("=", 1) for x in a.ckpt)
    levels = []
    for lv in a.levels:  # bỏ qua cấp chưa có checkpoint (ví dụ Cấp 8 chưa self-play xong)
        if lv in ck and not Path(ck[lv]).exists():
            print(f"[Giải đấu] bỏ qua {lv}: không thấy {ck[lv]}", file=sys.stderr)
            continue
        levels.append(lv)
    ags, wl = build_agents(levels, ck, dict(level3=2000, level4=a.sims, level7=a.sims, level8=a.sims))
    run_round_robin_tournament(ags, budget_mode=a.mode, workloads=wl, games_per_pair=a.games,
                               output_dir=a.out)


def cmd_figures(a):
    import runpy
    runpy.run_path(str(ROOT / "scripts" / "plot_training.py"), run_name="__main__")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("krk"); s.add_argument("--episodes", type=int, default=None); s.add_argument("--positions", type=int, default=200); s.set_defaults(f=cmd_krk)
    s = sp.add_parser("etl"); s.add_argument("--pgn", required=True); s.add_argument("--out", default="data/processed"); s.add_argument("--max_samples", type=int, default=2_000_000, help="giới hạn số mẫu (0 = không giới hạn; một tháng đầy đủ có thể lên hàng chục triệu mẫu, hàng chục GB)"); s.set_defaults(f=cmd_etl)
    s = sp.add_parser("train5"); s.add_argument("--data", default="data/processed"); s.add_argument("--epochs", type=int, default=None); s.add_argument("--max_steps", type=int, default=None); s.add_argument("--resume", default=""); s.add_argument("--workers", type=int, default=0); s.set_defaults(f=cmd_train5)
    s = sp.add_parser("ablation7"); s.add_argument("--ckpt", default="checkpoints/best_model.pt"); s.add_argument("--games", type=int, default=20); s.set_defaults(f=cmd_ablation7)
    s = sp.add_parser("selfplay"); s.add_argument("--ckpt", default="checkpoints/best_model.pt"); s.add_argument("--iterations", type=int, default=5); s.add_argument("--games", type=int, default=None, help="ván tự chơi mỗi vòng (chuẩn 50)"); s.add_argument("--sims", type=int, default=None, help="mô phỏng mỗi nước (chuẩn 200)"); s.add_argument("--arena_games", type=int, default=None, help="ván đấu gating (chuẩn 40)"); s.add_argument("--max_plies", type=int, default=None, help="số nửa nước tối đa mỗi ván (chuẩn 300)"); s.add_argument("--train_steps", type=int, default=None); s.set_defaults(f=cmd_selfplay)
    s = sp.add_parser("tournament"); s.add_argument("--levels", nargs="+", default=[f"level{i}" for i in (1, 2, 3, 4, 5, 7, 8)]); s.add_argument("--ckpt", nargs="*", default=["level5=checkpoints/best_model.pt", "level7=checkpoints/best_model.pt", "level8=checkpoints/best_az_model.pt"]); s.add_argument("--mode", default="workload_control"); s.add_argument("--games", type=int, default=None); s.add_argument("--sims", type=int, default=200, help="mô phỏng mỗi nước cho Cấp 4, 7, 8 (cùng ngân sách)"); s.add_argument("--out", default="results/round_robin_all", help="thư mục kết quả (không ghi đè results/round_robin của Cấp 1-4)"); s.set_defaults(f=cmd_tournament)
    s = sp.add_parser("figures"); s.set_defaults(f=cmd_figures)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()