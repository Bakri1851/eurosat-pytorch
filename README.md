# eurosat-pytorch

A `Dataset` written from scratch and verified against torchvision's, then trained end to end on GPU.

## What this is

EuroSAT is 27,000 Sentinel-2 images across 10 land-use classes. Classifying them is not the point — `torchvision.datasets.EuroSAT` already exists, and a small CNN handles the task comfortably.

The point is that the data layer is mine and it is checked. `EuroSATRaw` implements `Dataset` directly — class discovery, `__len__`, `__getitem__` — and is asserted against torchvision's `EuroSAT` on three things: identical length, identical `classes` in identical order, and identical `(tensor, label)` pairs over a seeded sample of indices.

That check is the deliverable. It is the difference between using a dataset and having written one.

## Status

**Complete, 20 September 2026.** Protocol and pass criteria are in [`docs/eurosat-pipeline.md`](docs/eurosat-pipeline.md), committed before the first run. Measured outcomes are appended below the line at the end of this file.

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

**Stratified split.** The class counts are uneven and Pasture is the smallest at 2,000. A split taken by shuffling all 27,000 at once would draw Pasture's test images hypergeometrically: expected 300 in a 15% test set, standard deviation 15.4, a 95% range of roughly 270 to 330. That is a ±5% swing in the rarest class's support, decided by nothing but `split_seed`.

It matters here for one specific reason. P5 measures seed-to-seed spread with `split_seed` held fixed and `init_seed` varying, so the number it yields is the noise floor of *initialisation*. Under an unstratified split, moving `split_seed` would move per-class support as well, and the two sources of variance could not be separated afterwards — which is exactly the separation Phase 2 needs in order to attribute a gap between optimisers to the optimiser. `make_splits` therefore shuffles within each class and slices each class separately.

**Plain accuracy is adequate** at 1.5:1. The majority classes hold 3,000 and the minority 2,000, so no single class dominates the aggregate and a macro-averaged metric would buy very little. Per-class accuracy is logged anyway — not as the reported metric, but as a bug detector. A fault that misaligns labels shows up as one class collapsing long before it shows up in the average.

## The trap that was not in §6

`ImageFolder` builds its sample list with `sorted()` on the filenames, which is lexicographic. EuroSAT's filenames carry an unpadded integer, so lexicographic and numeric order diverge at the *second* element and never re-converge:

```
torchvision order (sorted):       AnnualCrop_1, AnnualCrop_10, AnnualCrop_100, AnnualCrop_1000, ...
numeric order (what NOT to use):  AnnualCrop_1, AnnualCrop_2,  AnnualCrop_3,   AnnualCrop_4,    ...
```

First index at which they differ: 1.

Sorting numerically is the more obvious thing to do, because it is what a person means by "in order". Do it and the two implementations still agree on length, on `classes`, on class order, and on all 27,000 files. The labels even stay correct — each path keeps its own label, so a model trained on the numerically-sorted version would be unharmed.

That is what makes it dangerous. Nothing raises, nothing degrades, and the dataset is simply not the same object as torchvision's. Anything travelling by index stops meaning what it meant: a split computed under one ordering and applied under the other selects different images, and a reference comparison at index *i* compares two unrelated samples.

This is what P2 catches, and it is why P2 compares `(tensor, label)` pairs at sampled indices rather than comparing metadata — length, class list and class order all agree here. The negative control in `tests/test_data.py` re-sorts a copy numerically and asserts the comparison *fails*, so the check is shown to be capable of failing rather than assumed to be.

## P5 — noise floor

Five runs at one configuration: `split_seed = 0` held fixed, `init_seed` varying over 0–4, two epochs each. Everything about the data is constant across the five; the only things moving are the weight initialisation and the shuffle order.

| `init_seed` | final validation accuracy |
|---|---|
| 0 | 0.7889 |
| 1 | 0.7314 |
| 2 | 0.7904 |
| 3 | 0.7504 |
| 4 | 0.7267 |

Mean 0.758, sample standard deviation 0.031, and 0.064 between the best and worst run. Raw values in [`results/seed-spread.json`](results/seed-spread.json). The mean is not a performance claim — see the next section.

**Six and a half points of spread from initialisation alone**, at a fixed split, on a fixed configuration, with nothing else varying. That is larger than a good many differences reported in the literature as improvements, and it is the entire reason this measurement comes before any comparison rather than after one.

**The standard deviation is itself poorly determined.** At *n* = 5 the sampling error on an estimate of σ is roughly ±35%, and the exact interval is worse than that makes it sound: the 95% confidence interval on σ runs from 0.018 to 0.088, a factor of 4.8 end to end. So 0.031 is an order-of-magnitude statement, not a constant, and it should not be carried forward as a known quantity.

**What it implies about resolution.** Comparing two methods at *k* seeds each, the standard error on the difference in means is σ√(2/*k*):

| seeds per arm | se(difference) | significant at 95% | detectable at 80% power |
|---|---|---|---|
| 5 | 0.019 | 4.5 pp | 6.2 pp |
| 10 | 0.014 | 2.9 pp | 4.1 pp |
| 20 | 0.010 | 2.0 pp | 2.8 pp |
| 30 | 0.008 | 1.6 pp | 2.3 pp |

At five seeds per arm, a gap below about four and a half percentage points cannot be told from initialisation noise, and a real gap has to reach six before there is a fair chance of catching it at all. This is the seed-budget question stated in run-hours: three optimisers at twenty seeds is sixty runs per configuration, before a single tuning trial is counted. Running five and reporting the winner would be reporting the seed.

**Two caveats, both cutting against reuse of this number.** Two epochs is short — the model is nowhere near converged, so some of this spread is runs caught at different points along their trajectories rather than settling in genuinely different places, and the spread at convergence need not be the same. And this is untuned SGD on a small CNN, which is not what Phase 2 compares. Phase 2 measures its own noise floor, at its own budget, on its own model. What transfers is the procedure, not 0.031.

## What this does not establish

Accuracy is not a pass criterion and no accuracy claim is made here. The model is a small CNN trained for a few epochs at a fixed configuration that was never tuned. Its validation accuracy is a by-product of the pipeline running, not a result about EuroSAT, and it should not be set beside anything published on this dataset.

The 0.52 in the protocol is a 2024 coursework figure from an MLP on flattened 32×32 pixels — a floor set by discarding the spatial structure. It functions here as a bug detector: a convolutional model landing near it has something broken. Clearing it establishes that the pipeline works, which is what P1 already says, and nothing more.

None of P1–P5 says the model is good, the architecture sensible, or the optimiser settings reasonable. They say the data layer is mine and checked, the runs are reproducible, and the split is stable. That is the whole claim.

## On writing verification code

The most useful thing this exercise produced is not in the criteria table.

Three checks written during the build passed on their first run and could not have done otherwise:

- **A comparison of shapes standing in for a comparison of values.** Every sample here is a 3×64×64 tensor, so a check that two samples agree on shape agrees for every pair of samples that will ever be drawn — including pairs that ought to have disagreed. It reported success across the whole sample and tested nothing. `torch.equal` is the fix.
- **A negative control that rebound a local name instead of mutating the object.** The control is meant to break the dataset and confirm the comparison notices. Rebinding inside the function left the object it was supposed to corrupt untouched, so the "broken" dataset was the working one, and the control confirmed that a correct dataset matches a correct dataset.
- **An `assert` on a list literal.** A non-empty list is truthy, so the assertion holds whatever the comparisons inside it evaluate to. It is a syntactically valid way to write a check down and never run it.

Each of them ran, printed something encouraging, and established nothing. The shape they share is that all three fail *open*: the failure mode of a broken check is a pass, so the thing that would have told me was the thing that was broken.

Passing checks are not evidence unless the check has been shown capable of failing. That is why the negative control in `tests/test_data.py` exists, and why the control is now itself checked.

## Licence

MIT — see [`LICENSE`](LICENSE).

