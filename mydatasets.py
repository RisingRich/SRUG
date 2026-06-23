import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_image_files(image_dir):
    image_dir = Path(image_dir)
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class PairedImageDataset(Dataset):
    """Paired A/B image dataset with SRUG-style trainA/trainB folders."""

    def __init__(self, source_dir, target_dir, img_size=256):
        self.source_paths = list_image_files(source_dir)
        self.target_paths = list_image_files(target_dir)
        if len(self.source_paths) != len(self.target_paths):
            raise ValueError(
                f"source/target count mismatch: "
                f"{len(self.source_paths)} vs {len(self.target_paths)}"
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        source = Image.open(self.source_paths[index]).convert("RGB")
        target = Image.open(self.target_paths[index]).convert("RGB")
        return self.transform(source), self.transform(target)


def make_default_dirs(data_root):
    data_root = Path(data_root)
    return {
        "source_train": data_root / "trainA",
        "target_train": data_root / "trainB",
        "source_val": data_root / "testA",
        "target_val": data_root / "testB",
    }
