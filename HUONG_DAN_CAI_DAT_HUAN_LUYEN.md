# Hướng dẫn cài đặt và huấn luyện (Nhóm 27)

Dự án: Game AI Cờ vua 8 cấp, môn Trí tuệ Nhân tạo Nâng cao. Yêu cầu Python 3.11 trở lên.

## 1. Cài đặt

```bash
cd 101_ttntnc
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q          # kỳ vọng: toàn bộ test pass
```

Có GPU NVIDIA thì cài `torch` bản CUDA (theo pytorch.org) **trước** khi chạy `pip install -r requirements.txt`.
Bản `torch` mặc định chỉ chạy CPU; huấn luyện Cấp 5 trên CPU rất chậm.

## 2. Huấn luyện từng cấp

Mọi lệnh chạy từ thư mục gốc `101_ttntnc/`. Chỉ đưa vào báo cáo số liệu sinh ra từ lần chạy thật.

### Cấp 1–4 (không cần huấn luyện)
Đã có kết quả thật trong `results/`. Chạy lại: `python scripts/reproduce_experiment.py --full`
(dựng lại bảng và biểu đồ từ log: `--from-logs`).

### Cấp 6: học tăng cường trên cờ tàn KRK (không cần dữ liệu ngoài)
```bash
python scripts/run_full_pipeline.py krk
```
Chạy Q-learning (20000 tập) và REINFORCE, lưu `checkpoints/l6_krk_qtable.pkl`,
`results/training/krk_learning_curve.json` và `krk_eval.json`.

### Cấp 5: học có giám sát
1. Tải dữ liệu **Lichess Elite Database** tại https://database.nikonoel.fr/ (mỗi tháng một file `.zip`,
   khoảng vài trăm MB, ván của kỳ thủ Elo cao). Chỉ cần 1–3 tháng là đủ. Đặt file vào `data/raw/`;
   ETL đọc trực tiếp `.zip`, `.pgn` và `.pgn.zst`, không cần giải nén.
   (Có thể dùng dump đầy đủ tại https://database.lichess.org/ dạng `.pgn.zst`, nhưng rất lớn và ETL phải lọc rất nhiều.)
2. Tạo dữ liệu huấn luyện:
```bash
python scripts/run_full_pipeline.py etl --pgn data/raw/<tên_file>.zip
```
(thêm `--max_samples 2000000` nếu máy yếu). Kết quả nằm ở `data/processed/`.
3. Huấn luyện (cấu hình trong `configs/model_resnet.yaml`: 20 epoch, batch 256, lr 0.001):
```bash
python scripts/run_full_pipeline.py train5
python scripts/run_full_pipeline.py train5 --resume checkpoints/last_supervised.pt   # tiếp tục
```
Sinh `checkpoints/best_model.pt` và `results/training/supervised_curve.json`.

### Cấp 7: Guided MCTS (cần checkpoint Cấp 5)
Không có bước huấn luyện riêng; dùng `checkpoints/best_model.pt`. Chạy ablation:
```bash
python scripts/run_full_pipeline.py ablation7 --ckpt checkpoints/best_model.pt
```

### Cấp 8: self-play và đấu loại trực tiếp (cần checkpoint Cấp 5)
```bash
python scripts/run_full_pipeline.py selfplay --ckpt checkpoints/best_model.pt --iterations 5
```
Mỗi vòng: 50 ván tự chơi (200 mô phỏng/nước), huấn luyện, rồi đấu 40 ván với champion;
chỉ cập nhật khi điểm >= 0.55. Sinh `checkpoints/best_az_model.pt` và `results/training/selfplay_log.json`.
Bước này chạy lâu; có thể giảm `selfplay_games_per_iter` trong `configs/search_params.yaml` và ghi rõ trong báo cáo.

## 3. Đánh giá và biểu đồ
```bash
python scripts/run_full_pipeline.py tournament   # giải vòng tròn Cấp 1-5,7,8 (Cấp 6 chỉ chơi được KRK)
python scripts/run_full_pipeline.py figures      # vẽ biểu đồ vào results/figures/
python -m chess_ai.ui.desktop_pygame --level level3 --ckpt level5=checkpoints/best_model.pt
```

## 4. Thứ tự chạy gợi ý
`krk` → `etl` → `train5` → `ablation7` → `selfplay` → `tournament` → `figures` → cập nhật báo cáo.

## 5. Lỗi thường gặp
- `Cần checkpoint_path ...`: chưa huấn luyện Cấp 5, hoặc sai đường dẫn `--ckpt`.
- Hết bộ nhớ khi huấn luyện: giảm `batch_size` trong `configs/model_resnet.yaml` hoặc dùng `--max_samples` ở bước `etl`.
- Không có GPU: vẫn chạy được nhưng rất chậm; nên dùng Google Colab hoặc máy có GPU.
