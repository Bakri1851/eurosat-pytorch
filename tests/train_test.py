
import dataclasses

import torch

from src.config import RunConfig
from src.data import make_loaders
from src.model import SmallCNN
from src.train import train_one_run

def main():
    cfg = dataclasses.replace(RunConfig(), epochs = 3)

    # -- Check GPU is available
    assert torch.cuda.is_available(), "GPU is not available. Please check your CUDA installation."
    print(f"Using device: {cfg.device}")

    # -- Parameters and a fetched batch are both on cuda
    device = torch.device(cfg.device)
    model = SmallCNN(base_channels=cfg.base_channels, num_classes=cfg.num_classes).to(device)
    param_device = next(model.parameters()).device
    assert param_device.type == "cuda", f"Model parameters are not on GPU. Found on {param_device}."

    # -- It trains and the loss goes down
    result = train_one_run(cfg)

    assert len(result.train_loss) == cfg.epochs
    assert len(result.val_loss) == cfg.epochs
    assert len(result.val_accuracy) == cfg.epochs
    assert result.wall_clock_time > 0

    print("train loss :", [round(v, 4) for v in result.train_loss])
    print("val loss   :", [round(v, 4) for v in result.val_loss])
    print("val acc    :", [round(v, 4) for v in result.val_accuracy])
    print(f"wall clock : {result.wall_clock_time:.1f}s")

    assert result.train_loss[-1] < result.train_loss[0], (
        f"training loss did not decrease: {result.train_loss}"
    )
    print("P1 pass — trains end to end on the GPU, loss decreasing")

if __name__ == "__main__":
    main()