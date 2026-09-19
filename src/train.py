import time
from dataclasses import dataclass

import torch
from torch import nn 

from src.config import RunConfig
from src.data import make_loaders
from src.model import SmallCNN


@dataclass # this is a decorator that automatically generates special methods like __init__() and __repr__() for the class based on its attributes.
class RunResult:
    config: RunConfig
    train_loss: list[float] # per epoch
    val_loss: list[float]
    val_accuracy: list[float]
    test_accuracy: float | None # None unless this is final run
    wall_clock_time: float # seconds
    gpu_temp_c: list[float] | None # Empty list for now

def set_determinism(cfg):
    torch.manual_seed(cfg.init_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(cfg.deterministic)


def build_optimiser(cfg, model):
    if cfg.optimiser == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimiser: {cfg.optimiser}")


def train_one_run(cfg) -> RunResult:
    set_determinism(cfg)
    device = torch.device(cfg.device)

    train_loader, val_loader, test_loader = make_loaders(cfg)
    model = SmallCNN(base_channels=cfg.base_channels, num_classes=cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = build_optimiser(cfg, model)

    train_loss, val_loss, val_accuracy = [], [], []
    start = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimiser.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()
            running += loss.item() * images.size(0) # multiply by batch size to get total loss for this batch

        train_loss.append(running / len(train_loader.dataset))

        model.eval()
        running, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                running += loss.item() * images.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
        val_loss.append(running / len(val_loader.dataset))
        val_accuracy.append(correct / len(val_loader.dataset))

    return RunResult(
        config = cfg,
        train_loss = train_loss,
        val_loss = val_loss,
        val_accuracy = val_accuracy,
        test_accuracy = None, # not computed yet 
        wall_clock_time = time.perf_counter() - start,
        gpu_temp_c = []
    )

