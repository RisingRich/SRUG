"""Run a small SRUG forward pass and an MSS-loss check on CPU."""

import torch

from models import CRRB, SRUG
from utils import mss_loss


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = SRUG().eval()
    assert isinstance(model.encoder.conv1_0[0], CRRB)
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))
        target = torch.rand(1, 3, 176, 176)
        loss = mss_loss(target, target)
    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=1e-6)
    print("SRUG smoke test passed: CRRB encoder, NMD, and MSS loss.")


if __name__ == "__main__":
    main()
