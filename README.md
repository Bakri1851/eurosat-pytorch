# eurosat-pytorch

A `Dataset` written from scratch and verified against torchvision's, then trained end to end on GPU.

## What this is

EuroSAT is 27,000 Sentinel-2 images across 10 land-use classes. Classifying them is not the point — `torchvision.datasets.EuroSAT` already exists, and a small CNN handles the task comfortably.

The point is that the data layer is mine and it is checked. `EuroSATRaw` implements `Dataset` directly — class discovery, `__len__`, `__getitem__` — and is asserted against torchvision's `EuroSAT` on three things: identical length, identical `classes` in identical order, and identical `(tensor, label)` pairs over a seeded sample of indices.

That check is the deliverable. It is the difference between using a dataset and having written one.

## Status

**In progress.** Protocol and pass criteria are in [`docs/eurosat-pipeline.md`](docs/eurosat-pipeline.md), committed before the first run.

- [x] **P1** — trains end to end on the GPU
- [x] **P2** — `EuroSATRaw` agrees with `torchvision.datasets.EuroSAT`
- [x] **P3** — two runs at one config produce identical loss at every logged step
- [x] **P4** — the split is reproducible across processes
- [x] **P5** — *(optional)* seed-to-seed spread in final validation accuracy

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


---

# Results — 20 September 2026

Appended on closing. Everything above this line was committed before the first
run; everything below is measured.

## Criteria

| | Criterion | Outcome |
|---|---|---|
| **P1** | trains end to end on GPU | pass — 3 epochs, 63.6 s, train loss 1.069 → 0.612 |
| **P2** | `EuroSATRaw` agrees with torchvision | pass — 300 seeded indices, `torch.equal`, exact |
| **P3** | two runs identical at every logged step | pass — **exact** equality, not a tolerance |
| **P4** | split reproducible across processes | pass — identical test indices in two processes |
| **P5** | seed-to-seed spread | measured — sd ≈ 0.031 (see below) |

## Resolution of the **verify** items

- **Download host.** Confirmed for torchvision 0.26.0+cu128: pinned HuggingFace
  commit `c877bcd43f099cd0196738f714544e355477f3fd`, md5 `c8fa014336c82ac7804f0398fcb19387`.
  The `_check_exists()` trap in §3.1 is real as written — it tests only that
  `data/eurosat/2750` exists, not that extraction completed.
- **N = 27,000**, confirmed directly.
- **Per-class counts**, previously unmeasured: 3,000 each for AnnualCrop, Forest,
  HerbaceousVegetation, Residential, SeaLake; 2,500 each for Highway, Industrial,
  PermanentCrop, River; 2,000 for Pasture. Imbalance 1.5:1.
- **Channel statistics**, over the 18,900 training images only:
  mean `(0.3440, 0.3801, 0.4076)`, std `(0.2024, 0.1370, 0.1158)`.

## Decisions the measurements drove

**Stratified split.** [the hypergeometric argument — sd ≈ 15 on Pasture's 300]

**Plain accuracy is adequate** at 1.5:1. [per-class logged anyway as a bug detector]

## The trap that was not in §6

[`ImageFolder` sorts filenames lexicographically — `sorted(fnames)`. Numeric
sorting silently misaligns every index while both implementations look correct.
This is what P2 caught, and the negative control in `tests/test_data.py`
demonstrates P2 can catch it.]

## P5 — noise floor

[five accuracies, mean, sd, spread; the ±35% sampling error on n=5; the
resolvable-difference arithmetic; the two-epoch caveat]

## What this does not establish

[accuracy is not a criterion; 0.52 cleared as a bug detector only; no accuracy
claim is made]

## On writing verification code

[three checks initially could not fail — the tensor `.size` comparison, a
negative control binding a local, an assert on a list literal. Each ran, printed
something encouraging, and tested nothing.]

## Licence

MIT — see [`LICENSE`](LICENSE).

