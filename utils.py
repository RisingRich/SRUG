import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ms_ssim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error


METRIC_KEYS = ["PSNR", "SSIM", "LPIPS", "MS-SSIM", "MSE", "NMSE"]


def set_seed(seed):
    if seed is None or int(seed) < 0:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_history(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def ms_ssim_loss(pred, target):
    return 1.0 - ms_ssim(pred, target, data_range=1.0, size_average=True)


def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))


def tensor_to_uint8_image(tensor):
    tensor = tensor.detach().cpu()
    tensor = ((tensor + 1.0) * 0.5).clamp(0, 1)
    array = tensor.permute(1, 2, 0).numpy()
    return (array * 255.0).round().astype(np.uint8)


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def calculate_nmse(ref_img, gen_img):
    ref_img = ref_img.astype(np.float32)
    gen_img = gen_img.astype(np.float32)
    mse = np.mean((ref_img - gen_img) ** 2)
    denom = np.mean(ref_img ** 2)
    return float(mse / denom) if denom > 0 else 0.0


def evaluate_images(ref_img, gen_img, lpips_model=None, device="cpu"):
    if ref_img.shape != gen_img.shape:
        gen_img = np.array(Image.fromarray(gen_img).resize((ref_img.shape[1], ref_img.shape[0])))

    h, w = ref_img.shape[:2]
    win_size = min(7, h, w)
    psnr = peak_signal_noise_ratio(ref_img, gen_img, data_range=255)
    ssim = structural_similarity(ref_img, gen_img, data_range=255, win_size=win_size, channel_axis=2)
    mse = mean_squared_error(ref_img, gen_img)
    nmse = calculate_nmse(ref_img, gen_img)

    ref_tensor = torch.tensor(ref_img.transpose(2, 0, 1) / 255.0).unsqueeze(0).float().to(device)
    gen_tensor = torch.tensor(gen_img.transpose(2, 0, 1) / 255.0).unsqueeze(0).float().to(device)
    ms_ssim_score = ms_ssim(ref_tensor, gen_tensor, data_range=1.0, size_average=True, win_size=7).item()
    lpips_score = 0.0
    if lpips_model is not None:
        lpips_score = float(lpips_model(ref_tensor, gen_tensor).item())

    return {
        "PSNR": float(psnr),
        "SSIM": float(ssim),
        "LPIPS": float(lpips_score),
        "MS-SSIM": float(ms_ssim_score),
        "MSE": float(mse),
        "NMSE": float(nmse),
    }


def checkpoint_score(metrics, mode="psnr"):
    if mode == "psnr":
        return float(metrics["PSNR"])
    if mode == "composite":
        return float(metrics["PSNR"] + 2.0 * metrics["SSIM"] + 10.0 * metrics["MS-SSIM"])
    raise ValueError(f"Unknown best metric: {mode}")
