import argparse
import json
from pathlib import Path

import lpips
import torch

from mydatasets import IMAGE_EXTENSIONS
from utils import METRIC_KEYS, evaluate_images, load_rgb


def case_id(name):
    stem = Path(name).stem
    if stem.startswith(("A_", "B_")):
        return stem[2:]
    for suffix in ("_T1", "_T2", "_FLAIR", "_PD"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def match_prediction(ref_file, pred_dir):
    pred_dir = Path(pred_dir)
    exact = pred_dir / ref_file
    if exact.exists():
        return exact

    ref_key = case_id(ref_file)
    for pred_path in sorted(pred_dir.iterdir()):
        if pred_path.suffix.lower() in IMAGE_EXTENSIONS and case_id(pred_path.name) == ref_key:
            return pred_path
    return None


def evaluate_folder(ref_dir, pred_dir, device):
    ref_dir = Path(ref_dir)
    pred_dir = Path(pred_dir)
    lpips_model = lpips.LPIPS(net="alex").to(device)
    sums = {key: 0.0 for key in METRIC_KEYS}
    missing = []
    count = 0

    ref_files = sorted(path.name for path in ref_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    for ref_file in ref_files:
        ref_path = ref_dir / ref_file
        pred_path = match_prediction(ref_file, pred_dir)
        if pred_path is None:
            missing.append(ref_file)
            continue
        scores = evaluate_images(load_rgb(ref_path), load_rgb(pred_path), lpips_model=lpips_model, device=device)
        for key in METRIC_KEYS:
            sums[key] += scores[key]
        count += 1

    metrics = {key: sums[key] / max(1, count) for key in METRIC_KEYS}
    return {"metrics": metrics, "evaluated_count": count, "missing_count": len(missing), "missing": missing[:20]}


def main(opt):
    device = "cuda" if torch.cuda.is_available() and not opt.cpu else "cpu"
    result = evaluate_folder(opt.ref_dir, opt.pred_dir, device=device)
    if opt.output_json:
        output_path = Path(opt.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cfg():
    parser = argparse.ArgumentParser(description="Evaluate generated images against references.")
    parser.add_argument("--ref_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(cfg())
