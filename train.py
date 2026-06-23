import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import NestedUResnet
from mydatasets import PairedImageDataset, make_default_dirs
from utils import (
    append_history,
    charbonnier_loss,
    checkpoint_score,
    ensure_dir,
    evaluate_images,
    ms_ssim_loss,
    set_seed,
    tensor_to_uint8_image,
    write_json,
)


def build_model(opt):
    model_kwargs = {
        "encoder_type": opt.encoder_type,
        "decoder_type": opt.decoder_type,
        "decoder_attention": opt.decoder_attention,
    }
    return NestedUResnet(**model_kwargs), model_kwargs


def train_one_epoch(model, loader, optimizer, device, opt):
    model.train()
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    totals = {"loss_G": 0.0, "loss_L1": 0.0, "loss_MS_SSIM": 0.0, "loss_L2": 0.0}

    progress = tqdm(loader, desc="train", leave=False)
    for source, target in progress:
        source = source.to(device)
        target = target.to(device)
        pred = model(source)

        loss_l1 = l1_loss(pred, target)
        loss_ms = ms_ssim_loss((pred + 1.0) * 0.5, (target + 1.0) * 0.5)
        loss_l2 = mse_loss(pred, target)
        loss_charb = charbonnier_loss(pred, target)
        loss_g = (
            opt.l1_weight * loss_l1
            + opt.ms_ssim_weight * loss_ms
            + opt.l2_weight * loss_l2
            + opt.charbonnier_weight * loss_charb
        )

        optimizer.zero_grad()
        loss_g.backward()
        optimizer.step()

        totals["loss_G"] += float(loss_g.item())
        totals["loss_L1"] += float(loss_l1.item())
        totals["loss_MS_SSIM"] += float(loss_ms.item())
        totals["loss_L2"] += float(loss_l2.item())
        progress.set_description(f"train loss_G={loss_g.item():.4f}")

    count = max(1, len(loader))
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    sums = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0, "MS-SSIM": 0.0, "MSE": 0.0, "NMSE": 0.0}
    count = 0
    for source, target in loader:
        source = source.to(device)
        pred = model(source)
        for index in range(pred.shape[0]):
            pred_img = tensor_to_uint8_image(pred[index])
            target_img = tensor_to_uint8_image(target[index])
            metrics = evaluate_images(target_img, pred_img, lpips_model=None, device=device)
            for key in sums:
                sums[key] += metrics[key]
            count += 1
    return {key: value / max(1, count) for key, value in sums.items()}


def save_checkpoint(path, model, optimizer, epoch, best_score, best_epoch, model_kwargs, metrics):
    torch.save(
        {
            "epoch": epoch,
            "G_model": model.state_dict(),
            "optimizer_G_model": optimizer.state_dict(),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "model_kwargs": model_kwargs,
            "metrics": metrics,
        },
        path,
    )


def main(opt):
    set_seed(opt.seed)
    device = "cuda" if torch.cuda.is_available() and not opt.cpu else "cpu"
    run_dir = ensure_dir(opt.run_dir)
    weight_dir = ensure_dir(run_dir / "weights")
    result_dir = ensure_dir(run_dir / "results")
    write_json(run_dir / "config.json", vars(opt))

    train_set = PairedImageDataset(opt.source_train, opt.target_train, opt.imgsize)
    val_set = PairedImageDataset(opt.source_val, opt.target_val, opt.imgsize)
    train_loader = DataLoader(train_set, batch_size=opt.batch, shuffle=True, num_workers=opt.numworker)
    val_loader = DataLoader(val_set, batch_size=opt.batch, shuffle=False, num_workers=opt.numworker)

    model, model_kwargs = build_model(opt)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    start_epoch = 0
    best_score = float("-inf")
    best_epoch = None

    if opt.weight:
        checkpoint = torch.load(opt.weight, map_location=device)
        model.load_state_dict(checkpoint["G_model"], strict=False)
        if opt.resume_optimizer and "optimizer_G_model" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_G_model"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", best_score))
        best_epoch = checkpoint.get("best_epoch", best_epoch)

    writer = SummaryWriter(str(run_dir / "train_logs"))
    history_fields = ["epoch", "loss_G", "loss_L1", "loss_MS_SSIM", "loss_L2", "PSNR", "SSIM", "LPIPS", "MS-SSIM", "MSE", "NMSE"]

    for epoch in range(start_epoch, opt.epoch):
        train_losses = train_one_epoch(model, train_loader, optimizer, device, opt)
        val_metrics = validate(model, val_loader, device)
        score = checkpoint_score(val_metrics, opt.best_metric)

        history_row = {"epoch": epoch, **train_losses, **val_metrics}
        append_history(run_dir / "history.csv", history_row, history_fields)
        for key, value in {**train_losses, **val_metrics}.items():
            writer.add_scalar(key, value, epoch)

        save_checkpoint(
            weight_dir / "generator_only_last.pth",
            model,
            optimizer,
            epoch,
            max(best_score, score),
            best_epoch if score <= best_score else epoch,
            model_kwargs,
            val_metrics,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                weight_dir / "generator_only_best.pth",
                model,
                optimizer,
                epoch,
                best_score,
                best_epoch,
                model_kwargs,
                val_metrics,
            )
        print(f"epoch={epoch} PSNR={val_metrics['PSNR']:.4f} SSIM={val_metrics['SSIM']:.4f} best_epoch={best_epoch}")

    writer.close()
    print(f"Saved weights to {weight_dir}")
    print(f"Saved validation samples/logs to {result_dir}")


def cfg():
    parser = argparse.ArgumentParser(description="Train SRUG-style generator-only SRU-Pix2Pix.")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--source_train", type=str, default="")
    parser.add_argument("--target_train", type=str, default="")
    parser.add_argument("--source_val", type=str, default="")
    parser.add_argument("--target_val", type=str, default="")
    parser.add_argument("--run_dir", type=Path, default=Path("runs") / "generator_only")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--imgsize", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--numworker", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--weight", type=str, default="")
    parser.add_argument("--resume_optimizer", action="store_true")
    parser.add_argument("--encoder_type", choices=("resnet", "seresnet"), default="seresnet")
    parser.add_argument("--decoder_type", choices=("unet", "unetpp"), default="unetpp")
    parser.add_argument("--decoder_attention", action="store_true")
    parser.add_argument("--l1_weight", type=float, default=100.0)
    parser.add_argument("--ms_ssim_weight", type=float, default=100.0)
    parser.add_argument("--l2_weight", type=float, default=0.0)
    parser.add_argument("--charbonnier_weight", type=float, default=0.0)
    parser.add_argument("--best_metric", choices=("psnr", "composite"), default="psnr")
    parser.add_argument("--cpu", action="store_true")
    opt = parser.parse_args()

    if opt.data_root:
        defaults = make_default_dirs(opt.data_root)
        opt.source_train = opt.source_train or str(defaults["source_train"])
        opt.target_train = opt.target_train or str(defaults["target_train"])
        opt.source_val = opt.source_val or str(defaults["source_val"])
        opt.target_val = opt.target_val or str(defaults["target_val"])

    missing = [name for name in ("source_train", "target_train", "source_val", "target_val") if not getattr(opt, name)]
    if missing:
        raise SystemExit("Missing data paths: " + ", ".join(missing))
    return opt


if __name__ == "__main__":
    main(cfg())
