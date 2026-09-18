"""Run configuration.

One dataclass, every field that changes a number. Phase 2's sweep varies
fields of this object and nothing else, so anything that affects a result
and lives outside it is a reproducibility hole.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    # data
    data_root: str = "data"
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 0          # fixes which images land where; separate from init_seed

    # model
    base_channels: int = 32      # fixed, not swept (protocol section 9) - but it
                                 # changes a number, so it lives here like everything else
    num_classes: int = 10        # assert against len(dataset.classes); never trust this

    # optimisation
    optimiser: str = "sgd"
    lr: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 0.0
    batch_size: int = 128
    epochs: int = 10

    # execution
    init_seed: int = 0           # fixes weights and shuffle order; separate from split_seed
    num_workers: int = 0
    pin_memory: bool = True
    device: str = "cuda"
    deterministic: bool = True
