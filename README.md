# eurosat-pytorch

A `Dataset` written from scratch and verified against torchvision's, then trained end to end on GPU.

## What this is

EuroSAT is 27,000 Sentinel-2 images across 10 land-use classes. Classifying them is not the point — `torchvision.datasets.EuroSAT` already exists, and a small CNN handles the task comfortably.

The point is that the data layer is mine and it is checked. `EuroSATRaw` implements `Dataset` directly — class discovery, `__len__`, `__getitem__` — and is asserted against torchvision's `EuroSAT` on three things: identical length, identical `classes` in identical order, and identical `(tensor, label)` pairs over a seeded sample of indices.

That check is the deliverable. It is the difference between using a dataset and having written one.

## Status

**In progress.** Protocol and pass criteria are in [`docs/eurosat-pipeline.md`](docs/eurosat-pipeline.md), committed before the first run.

- [ ] **P1** — trains end to end on the GPU
- [-] **P2** — `EuroSATRaw` agrees with `torchvision.datasets.EuroSAT`
- [ ] **P3** — two runs at one config produce identical loss at every logged step
- [ ] **P4** — the split is reproducible across processes
- [ ] **P5** — *(optional)* seed-to-seed spread in final validation accuracy

**Accuracy is deliberately not a pass criterion, and no accuracy claim is made here.** A mediocre classifier that runs the full pipeline on device satisfies every criterion above; a good one obtained by fighting the dataloader for three days satisfies none of them better.

## Why the criteria are shaped this way

Three of the five are about reproducibility rather than performance, which is the unusual part and the deliberate part.

A run that cannot be reproduced cannot support a comparison. P3 and P4 exist because the next thing this code does is compare optimisers, and a difference between two runs is only attributable to the thing you changed if nothing else moved. Most of the cost of getting that right is paid once, in the data layer, which is where it is paid here.

P5 is the same argument one step further: without knowing the spread across seeds at a fixed configuration, there is no scale against which to judge whether a gap between two methods means anything.

## Setup

```bash
git clone https://github.com/Bakri1851/eurosat-pytorch
cd eurosat-pytorch

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

A plain `pip install torch` returns a CPU build on Windows — the index URL is not optional. Verify before running anything:

```python
import torch
print(torch.__version__)                     # 2.11.0+cu128
print(torch.cuda.get_device_capability())    # (12, 0) on Blackwell
```

The dataset downloads itself on first run to `data/`, which is gitignored. See the protocol, §3.1 for the extraction trap — an interrupted first download leaves a partial tree that `download=True` will never repair.

### Environment

| | |
|---|---|
| Python | 3.13 |
| PyTorch | 2.11.0+cu128 |
| GPU | RTX 5070 Laptop, 8 GB, sm_120 (Blackwell) |
| Host | Ryzen 9 270, 32 GB |

## Layout

```
eurosat-pytorch/
├── docs/                 # protocol and pass criteria, fixed before the run
├── src/                  # config, data, model, train
├── notebooks/            # figures only; the logic lives in src/
├── figures/
├── requirements.txt
└── README.md
```

Logic lives in `src/` rather than in a notebook: once a run contributes to a reported number, a notebook cannot establish what ran in what order.

## Context

This is a Phase 0 exercise from a larger project on
[training dynamics and optimisation](https://github.com/Bakri1851/training-dynamics) —
whether the neural scaling exponent depends on the optimiser. It is kept separate
because it is a self-contained piece of work with its own criteria, and because the
model here is not the model that project scales.

## Licence

MIT — see [`LICENSE`](LICENSE).
