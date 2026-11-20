"""
chess_ai/training/train_supervised.py
Huấn luyện có giám sát mạng Dual-Head ResNet (Cấp 5): AdamW + cosine annealing,
AMP FP16 khi có GPU, CheckpointManager hỗ trợ --resume, ghi đường cong học ra JSON.

Chỉ báo cáo số liệu sinh ra từ lần chạy thật của script này.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from chess_ai.config import load_yaml_config
from chess_ai.etl.memmap_dataset import ChessMemmapDataset
from chess_ai.models.dual_resnet import DualHeadResNet
from chess_ai.models.io import load_checkpoint, save_checkpoint
from chess_ai.models.loss import DualHeadLoss
from chess_ai.utils.seeding import set_global_seed


@torch.no_grad()
def evaluate(model: DualHeadResNet, loader: DataLoader, crit: DualHeadLoss, device: str) -> Dict[str, float]:
    model.eval()
    n = 0
    tot = pol = val = top1 = top3 = 0.0
    for x, a, z in loader:
        x, a, z = x.to(device), a.to(device), z.to(device)
        logits, v = model(x)
        loss, d = crit(logits, v, a, z)
        b = x.shape[0]
        n += b
        tot += d["total"] * b
        pol += d["policy"] * b
        val += d["value"] * b
        t3 = logits.topk(3, dim=1).indices
        top1 += (t3[:, 0] == a).sum().item()
        top3 += (t3 == a.unsqueeze(1)).any(dim=1).sum().item()
    n = max(n, 1)
    return dict(loss=tot / n, policy_loss=pol / n, value_mse=val / n, top1=top1 / n, top3=top3 / n)


def train_supervised(
    config_path: str = "configs/model_resnet.yaml",
    resume_path: str = "",
    data_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    metrics_path: str = "results/training/supervised_curve.json",
    epochs: Optional[int] = None,
    max_steps_per_epoch: Optional[int] = None,
    seed: int = 42,
    device: Optional[str] = None,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """Thực thi vòng lặp huấn luyện, đánh giá val/test và lưu checkpoint chuẩn."""
    cfg = load_yaml_config(config_path)
    tcfg, ncfg = cfg["training"], cfg["network"]
    set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or int(tcfg["epochs"])

    model_config = dict(
        in_channels=cfg["representation"]["input_shape"][0],
        num_res_blocks=ncfg["num_res_blocks"], filters=ncfg["filters"],
        policy_conv_filters=ncfg["policy_head"]["conv_filters"],
        value_conv_filters=ncfg["value_head"]["conv_filters"],
        value_hidden_units=ncfg["value_head"]["hidden_units"],
        action_space=cfg["representation"]["action_space"],
    )
    model = DualHeadResNet(**model_config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["learning_rate"]), weight_decay=float(tcfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=float(tcfg["lr_scheduler"]["eta_min"]))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = DualHeadLoss(tcfg["loss"]["policy_weight"], tcfg["loss"]["value_weight"])

    train_ds = ChessMemmapDataset(data_dir, "train")
    val_ds = ChessMemmapDataset(data_dir, "val")
    test_ds = ChessMemmapDataset(data_dir, "test")
    bs = int(tcfg["batch_size"])
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=bs, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=bs, num_workers=num_workers)

    start_epoch, global_step, best_val = 0, 0, float("inf")
    history = []
    if resume_path:
        ck = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["scheduler_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch, global_step = int(ck["epoch"]) + 1, int(ck["global_step"])
        best_val = float(ck.get("best_val_loss", best_val))
        history = list(ck.get("history", []))

    ckdir = Path(checkpoint_dir)
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        run, nb = 0.0, 0
        for step, (x, a, z) in enumerate(train_loader):
            if max_steps_per_epoch and step >= max_steps_per_epoch:
                break
            x, a, z = x.to(device), a.to(device), z.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, v = model(x)
                loss, _ = crit(logits, v, a, z)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += float(loss.detach()); nb += 1; global_step += 1
        sched.step()
        val = evaluate(model, val_loader, crit, device) if len(val_ds) else dict(loss=float("nan"), top1=float("nan"), top3=float("nan"), value_mse=float("nan"), policy_loss=float("nan"))
        rec = dict(epoch=epoch, global_step=global_step, train_loss=run / max(nb, 1),
                   val_loss=val["loss"], val_policy_acc=val["top1"], val_top3=val["top3"],
                   val_value_mse=val["value_mse"], lr=opt.param_groups[0]["lr"], elapsed_sec=time.time() - t0)
        history.append(rec)
        is_best = len(val_ds) > 0 and val["loss"] < best_val
        if is_best:
            best_val = val["loss"]
        payload = dict(
            epoch=epoch, global_step=global_step, model_state_dict=model.state_dict(),
            optimizer_state_dict=opt.state_dict(), scheduler_state_dict=sched.state_dict(),
            scaler_state_dict=scaler.state_dict(), train_loss=rec["train_loss"],
            val_policy_acc=rec["val_policy_acc"], val_value_mse=rec["val_value_mse"],
            best_val_loss=best_val, model_config=model_config, random_seed=seed, history=history,
        )
        save_checkpoint(str(ckdir / "last_supervised.pt"), payload)
        if is_best or not len(val_ds):
            save_checkpoint(str(ckdir / "best_model.pt"), payload)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    result = dict(history=history, best_val_loss=best_val, device=device, epochs=epochs, seed=seed,
                  n_train=len(train_ds), n_val=len(val_ds), n_test=len(test_ds))
    if len(test_ds):
        best = load_checkpoint(str(ckdir / "best_model.pt"), map_location=device)
        model.load_state_dict(best["model_state_dict"])
        result["test"] = evaluate(model, test_loader, crit, device)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Huấn luyện có giám sát Cấp 5")
    ap.add_argument("--config", default="configs/model_resnet.yaml")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--metrics", default="results/training/supervised_curve.json")
    ap.add_argument("--resume", default="")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_steps_per_epoch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    train_supervised(a.config, a.resume, a.data_dir, a.checkpoint_dir, a.metrics, a.epochs, a.max_steps_per_epoch, a.seed)
