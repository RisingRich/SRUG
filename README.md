# SRUG

Implementation of SRUG for paired MRI sequence synthesis. The generator uses a
channel-recalibrated feature pyramid and a Nested Multi-Scale Decoder (NMD),
trained with L1 loss and Multi-Scale Structural Loss (MSS loss).

## Architecture and terminology

| Paper component | Code | Description |
| --- | --- | --- |
| SRUG | `SRUG` | Direct source-to-target generator |
| Channel-Recalibrated Residual Block (CRRB) | `CRRB` | Residual bottleneck with SE recalibration before shortcut addition |
| Adapted residual encoder | `PyramidEncoder`, `crrb_encoder()` | Full-resolution convolutional stem and four pooling-separated residual stages |
| Nested Multi-Scale Decoder (NMD) | `SRUG.forward_nmd()`, `decoder_type="nmd"` | Repeated fusion of encoder and decoder features through nested skip connections |
| Multi-Scale Structural Loss (MSS loss) | `mss_loss()`, `--mss_weight` | `1 - MS-SSIM` on the rescaled prediction and target |

The encoder is adapted for reconstruction. The NMD takes inspiration from
U-Net++ connectivity; this implementation does not use the complete original
segmentation network. The default generator has one output head, with no
discriminator or adversarial objective.

```python
from models import SRUG

model = SRUG(encoder_type="crrb", decoder_type="nmd")
prediction = model(source)  # source: N x 3 x H x W; H and W divisible by 16
```

The paired-image loader converts images to RGB. The default model expects three
input channels and predicts three output channels.

## Files

- `models.py`: CRRB, feature pyramid, SRUG and decoder paths.
- `train.py`: supervised training and checkpoint saving.
- `test.py`: checkpoint loading and image synthesis.
- `mydatasets.py`: paired-image loading.
- `utils.py`: MSS loss, evaluation metrics and logging helpers.
- `ping.py`: image-folder evaluation.
- `smoke_test.py`: default-model forward pass and loss check.

## Setup and data

```bash
pip install -r requirements.txt
```

```text
dataset_root/
  trainA/
  trainB/
  testA/
  testB/
```

Source and target files must be paired in matching sorted order. The compact
training entry uses `testA/testB` for validation by default. Use
`--source_val` and `--target_val` for a separate validation split when reserving
a held-out test set.

## Train

```bash
python train.py --data_root /path/to/dataset_root --run_dir runs/example --encoder_type crrb --decoder_type nmd --batch 2 --epoch 200 --imgsize 256
```

L1 and MSS loss have a default weight ratio of **1:1**. Training applies MSS loss
to `(prediction + 1) / 2` and `(target + 1) / 2`; the underlying MS-SSIM
calculation is unchanged. The logged structural-loss field is `loss_MSS`.
The evaluation metric retains its standard name, `MS-SSIM`.

Checkpoints are saved to:

```text
runs/example/weights/generator_only_best.pth
runs/example/weights/generator_only_last.pth
```

The encoder ablation uses `--encoder_type without_se`; the sequential decoder
ablation uses `--decoder_type plain`. The default is `crrb` with `nmd`.

## Inference and evaluation

The default output and evaluation are both 256 x 256.

```bash
python test.py --checkpoint runs/example/weights/generator_only_best.pth --input_dir /path/to/testA --output_dir runs/example/pred_best
python ping.py --ref_dir /path/to/testB --pred_dir runs/example/pred_best --output_json runs/example/metrics.json
python smoke_test.py
```

`ping.py` reports PSNR, SSIM, LPIPS, MS-SSIM, MSE and NMSE. Pretrained
checkpoints are not bundled in this repository.

## Compatibility with earlier names

| Previous name | Current name |
| --- | --- |
| `NestedUResnet` | `SRUG` |
| `SEBottleNeck` | `CRRB` |
| `BottleNeck` | `ResidualBlock` |
| `VGGBlock` | `FeatureFusionBlock` |
| `CNN_Encoder` | `PyramidEncoder` |
| `seresnet` / `resnet` | `crrb` / `without_se` |
| `unetpp` / `unet` | `nmd` / `plain` |
| `ms_ssim_loss`, `--ms_ssim_weight` | `mss_loss`, `--mss_weight` |

Previous imports, encoder/decoder option values, and the old loss-weight flag
remain accepted. New configurations use the current names. Tensor operations,
parameter registration keys, checkpoint filenames and loss weights are
unchanged, so existing state-dict checkpoints can be loaded directly. Historical
CSV files retain their original headers when training resumes.
