import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from models import NestedUResnet
from mydatasets import IMAGE_EXTENSIONS


def preprocess_image(path, device, img_size):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


@torch.no_grad()
def inference(model, tensor, out_size):
    output = model(tensor)[0]
    output = F.interpolate(output.unsqueeze(0), size=out_size, mode="bilinear", align_corners=False)[0]
    output = ((output + 1.0) * 0.5).clamp(0, 1)
    output = (output * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(output)


def load_generator(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = NestedUResnet(**checkpoint.get("model_kwargs", {})).to(device)
    model.load_state_dict(checkpoint["G_model"], strict=False)
    model.eval()
    return model


def run(opt):
    device = "cuda" if torch.cuda.is_available() and not opt.cpu else "cpu"
    model = load_generator(opt.checkpoint, device)
    input_dir = Path(opt.input_dir)
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            input_path = Path(root) / name
            tensor = preprocess_image(input_path, device=device, img_size=opt.imgsize)
            image = inference(model, tensor, out_size=(opt.out_h, opt.out_w))
            image.save(output_dir / name)
            count += 1
    print(f"Saved {count} generated images to {output_dir}")


def cfg():
    parser = argparse.ArgumentParser(description="Inference with a generator-only checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--imgsize", type=int, default=256)
    parser.add_argument("--out_h", type=int, default=256)
    parser.add_argument("--out_w", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(cfg())
