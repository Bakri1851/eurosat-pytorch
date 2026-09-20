# EuroSAT — the data pipeline

**Writing a `Dataset` from scratch and verifying it against torchvision's**

**Open**, pre-registered 17 September 2026. A Phase 0 exercise from [training-dynamics](https://github.com/Bakri1851/training-dynamics), kept in its own repository.

| | |
|---|---|
| **Artefact** | `src/config.py`, `src/data.py`, `src/model.py`, `src/train.py` |
| **Device** | CUDA — this is the first milestone that touches the GPU |
| **Precision** | `float32`. bf16 is a Phase 3 memory problem and is deliberately not solved here |
| **Runtime budget** | < 5 min per run. If a run takes longer, the model is too big for what this establishes |

Protocol and pass criteria below are fixed before the first run. No numbers in this document are measurements yet; the ones marked **verify** are things I have not confirmed and must not assume.

---

## 1. What this establishes — and what it does not

M1 ended in a number I could check against a formula. This one cannot: there is no closed form for a CNN on satellite imagery, and pretending otherwise would be the kind of overclaim the repository is supposed to avoid.

What it establishes is narrower and still worth a milestone:

1. The full data path works on this machine — `Dataset` → `DataLoader` → device → model → optimiser step → evaluation.
2. The split is reproducible, so two runs differ only by what I intended to change.
3. The pieces exist as importable modules with the shape Phase 2's sweep will call.

**What it does not establish: anything about accuracy, and anything about optimisers.** A mediocre classifier that runs the full pipeline on device clears this milestone. A good one obtained by fighting the dataloader for three days does not clear it any harder. Accuracy is not a pass criterion and no accuracy number from this milestone belongs in the README.

## 2. Why this goes in `src/` and not a notebook

I originally argued this on the grounds that the code here *is* the optimiser-comparison harness and would be reused verbatim. That was overstated, and the correction is in §9: the model and the image-specific half of the data layer are throwaway. What survives is narrower — the shape of `RunConfig`, the signature of `train_one_run`, and the seeding discipline. Patterns, not a library.

The reason that survives is smaller and still sufficient: **a notebook cannot establish what ran in what order.** P3 and P4 below are claims about reproducibility, and a notebook whose cells can be executed out of sequence cannot support them. Execution counts are not provenance.

Writing it as modules also means the P2 agreement check is an importable function rather than a cell, which is what makes it a test rather than a thing that was true once.

A thin notebook on top for the figures is fine. The logic does not live there.

## 3. The dataset — what I verified, and what I got wrong before

Checked against `torchvision.datasets.eurosat` source, torchvision 0.29:

- **`EuroSAT` takes no `split` argument.** The signature is `EuroSAT(root, transform, target_transform, download, loader)`. It subclasses `ImageFolder` over `root/eurosat/2750/` and hands you the whole thing. The split is yours to make, which is §6 trap 1.
- **Class order is `sorted()` on the directory names** — `folder.find_classes` does `sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())`. So label indices are alphabetical and stable across machines. Read `ds.classes` anyway rather than hardcoding the mapping; it costs nothing and it is self-documenting.
- **The download is a pinned HuggingFace commit with an md5**, not the DFKI university mirror. **This corrects what I told you previously.** I said the host had been intermittently down and not to debug your own code over it — that advice was about the old mirror and no longer applies to current torchvision. An md5 is checked on extract, so a corrupt download fails loudly rather than silently.
- **Version caveat.** `requirements.txt` pins `torchvision>=0.26`, and I could not confirm which release the URL changed in. One line settles it for whatever you have installed:

  ```python
  import inspect
  from torchvision.datasets import EuroSAT
  print(inspect.getsource(EuroSAT.download))
  ```

- **N = 27,000, confirmed** — `x.shape == (27000, 3072)` in the 2024 coursework notebook, at 32×32×3. Native resolution is 64×64; the 32×32 was that coursework's downscale, not a property of the dataset.
- **Per-class counts are not uniform and remain unmeasured.** Ten classes, unequal sizes. I could not reach the data to count them and will not quote numbers I have not measured. One `Counter` over `ds.samples`, recorded in the notebook. It decides whether the split must be stratified and whether plain accuracy is an adequate metric.

### 3.1 Decision: `torchvision.datasets.EuroSAT`, not the 2024 copy

The 2024 coursework's `EuroSAT_RGB/<ClassName>/*.jpg` is already `ImageFolder` layout, so pointing `ImageFolder` at it would work and would skip a 90 MB download. **Decided against, on 17 September.**

```python
from torchvision.datasets import EuroSAT
ds = EuroSAT(root=cfg.data_root, download=True, transform=...)   # -> data/eurosat/2750/
```

The reasoning, recorded because the cheaper option looks better right up until someone else tries to run this:

- **A clone has to be runnable.** `ImageFolder` on the old copy makes the repo depend on a directory that exists on one laptop. This repository is an application artefact whose stated case is reproducibility; "clone and run" is worth more than a saved download.
- **Integrity.** The archive is md5-checked on extract. The 2024 copy has no such guarantee — it has been through a coursework preprocessing pipeline, a OneDrive sync, and two years.
- **Provenance is pinned**, to HuggingFace commit `c877bcd4`, rather than "downloaded from somewhere in 2024".

`data/` is already in `.gitignore`, so the extracted tree stays out of the history.

> **The trap, and it is specific to this class.** `download()` returns immediately if `_check_exists()` is true, and `_check_exists()` only tests `os.path.exists(root/eurosat/2750)` — it does not check that the extraction completed. So an interrupted first extract leaves a partial tree that `download=True` will never repair, on any later run, silently. If the first download is interrupted, **delete `data/eurosat/` and start again**; do not re-run and assume it healed itself. Confirm the extract with a count before trusting it.

**Cross-check once, then retire the old copy.** After downloading, count both trees. If they agree at 27,000 and per class, the 0.52 floor in §3.2 transfers cleanly. If they disagree, the coursework number describes a different dataset and should not be used as a floor at all.

### 3.1.1 …and then write the `Dataset` yourself anyway

Choosing `EuroSAT` over `ImageFolder` settles where the bytes come from. It settles nothing about what you learn, because **`EuroSAT` *is* an `ImageFolder`** — it subclasses it, adds a download and a path convention, and inherits everything else. Either way a finished `Dataset` is handed to you.

The roadmap's purpose for this milestone is `Dataset`, `DataLoader`, batching and device transfer. Using a ready-made dataset skips the first of those four outright, and it is the one with the most transferable content: Phase 2 and 3 will need a dataset object that yields whatever the scaling study trains on, and that will not be a stock torchvision class.

So write both.

1. `EuroSAT(root, download=True)` fetches the data and is the **reference implementation**.
2. `EuroSATRaw(Dataset)` is yours — class discovery, `__len__`, `__getitem__`, over the same files on disk.
3. **Assert they agree.** Same length; same `classes` list in the same order; and over a random sample of indices, identical tensors and identical labels.

Roughly twenty lines for step 2, and step 3 is the point. It is the same move M1 made — a thing you built, checked against an independent implementation of the same thing — and it is why M1 is worth showing someone and a stock-dataset tutorial is not. Without it this milestone has no verification content at all; it just runs.

The agreement check is **P2** in §7.

## 3.2 The 2024 coursework as a reference — what to take and what to ignore

**Take the floor.** 5-fold CV accuracy was **0.52** (plain and stratified, 10 classes, chance 0.10) for a 50-unit MLP on flattened 32×32 pixels. That is a genuine lower bound and a cheap bug detector: a CNN at 64×64 that *keeps spatial structure* and lands anywhere near 0.52 has something broken in the pipeline — most likely the label mapping, the normalisation, or augmentation leaking into eval. I expect a small CNN to clear it by a wide margin, but I have not measured that and you should not take my figure for it; the floor is the part that's measured.

**Do not take it as a target.** 0.52 is set by the model class, not by the data. Flattening throws away every spatial relationship in a satellite image. Beating it is not an achievement and optimising toward it is the fastest route to turning this milestone into the accuracy chase §1 rules out.

**Two of the traps below are live in that code, which is why they are worth stating.**

- `q2` calls `StandardScaler().fit_transform` on all 27,000 samples, and `q3` splits *afterwards* — trap 3, test statistics in the training normalisation.
- `q3`'s trainer evaluates `m.score(X_test, y_test)` every epoch and then selects `best_model` on `test_c[-1]` — trap 2, model selection on the test set. Any accuracy from that run is a selected number, not a held-out one.

Neither is a criticism of the coursework, which was presumably marked to a spec that asked for those curves. They matter because this project's entire claim to being worth reading is that it does not make protocol errors of that class, and the habits carry unless they are named.

**One more, and it is the one that bears directly on Phase 3.** `q5` compares plain against stratified CV with `scipy.stats.ttest_rel` over 5 folds. A paired *t*-test is not valid across CV folds: the folds share training data, so the scores are not independent and the test's variance estimate is biased downward — it over-rejects. Here it returned *p* = 0.65 and concluded no effect, so nothing rests on it. But Phase 3 puts confidence intervals on α across optimisers, which is the same question with the same structure, and reaching for a paired *t*-test there would be a real error in the part of the work that is supposed to be the differentiator. Nadeau & Bengio's corrected resampled *t*-test, or the bootstrap the roadmap already commits to, is what that needs.

## 4. Configuration

One dataclass, every field that changes a number. Phase 2's sweep varies fields of this object and nothing else, so anything that affects a result and lives outside it is a reproducibility hole.

```python
# src/config.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class RunConfig:
    # data
    data_root: str = "data"
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 0          # fixes which images land where; separate from init_seed
    # model
    base_channels: int = 32      # fixed, not swept (§9) - but it changes a
                                 # number, so it lives here like everything else
    num_classes: int = 10
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
```

**Two seeds, not one.** `split_seed` fixes which images land in which split; `init_seed` fixes weights and batch order. They must be separable, because Phase 2 asks "how much does this vary across seeds *at a fixed split*" and that question is unanswerable if one seed controls both.

## 5. Structure

Signatures and contracts. The bodies are the exercise.

```python
# src/data.py

class EuroSATRaw(Dataset):
    """Your own Dataset over root/eurosat/2750/<ClassName>/*.jpg.

    Written from scratch rather than inherited — see 3.1.1. torchvision's
    EuroSAT is the reference this is checked against, not the thing used.

    Contract:
      - self.classes is sorted(), matching torchvision's find_classes, so
        label indices agree between the two implementations
      - __getitem__ returns (image, label) and applies self.transform
      - the file list is built once in __init__, not per __getitem__
      - no .to(device) anywhere in here (trap 9)
    """
    def __init__(self, root, transform=None): ...
    def __len__(self) -> int: ...
    def __getitem__(self, i): ...


def assert_matches_reference(mine: Dataset, reference: Dataset, n: int, seed: int) -> None:
    """P2. Raise if the two disagree on length, classes, or n sampled items.

    Compare tensors with torch.equal, not ==. Sample the indices rather than
    checking all 27,000: decoding the set twice is minutes you do not need
    to spend, and a seeded sample of a few hundred that agrees exactly is
    already a strong statement.
    """
    ...


def channel_stats(dataset, indices) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-channel mean and std over `indices` only.

    Contract: reads only the indices given. Called with the TRAIN indices.
    Returns values in the same units the ToTensor-ed images are in.
    """
    ...

def make_splits(cfg) -> tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """Return (train_idx, val_idx, test_idx) as index lists into the full dataset.

    Contract:
      - deterministic given cfg.split_seed, across processes and machines
      - the three are disjoint and exhaust the dataset
      - returns INDICES, not Datasets — so the train/val transform split in
        make_loaders is possible at all (see trap 6)
    """
    ...

def make_loaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders.

    Contract:
      - train loader shuffles; val and test do not
      - train applies augmentation, val and test do not
      - normalisation constants come from channel_stats on train indices only
      - shuffle order is governed by cfg.init_seed, not cfg.split_seed
    """
    ...
```

```python
# src/model.py

class SmallCNN(nn.Module):
    """Three conv blocks, global average pool, linear head.

    Args:
        num_classes: read from the dataset, not hardcoded
        base_channels: channel count of block 1; blocks 2 and 3 are 2x and 4x it.
            Pick one that trains in under five minutes. It is not a width ladder
            for anything downstream — see §9.

    Deliberately not a pretrained ResNet: a pretrained backbone makes the numbers
    uninterpretable (you are measuring ImageNet, not your pipeline) and adds a
    second download to debug.
    """
    ...
```

```python
# src/train.py

@dataclass
class RunResult:
    config: RunConfig
    train_loss: list[float]      # per epoch
    val_loss: list[float]
    val_acc: list[float]
    test_acc: float | None       # None unless the run was the final one
    wall_clock_s: float
    gpu_temp_c: list[float]      # §6 thermal commitment; empty list if unavailable

def train_one_run(cfg: RunConfig) -> RunResult:
    """Train one model under one config. No printing, no plotting, no globals.

    This signature is the point of the milestone. Phase 2's sweep is
    `[train_one_run(c) for c in configs]` and nothing more, so anything this
    function reads that is not in `cfg` is a bug that surfaces in six weeks.
    """
    ...
```

## 6. Traps, ordered by what they cost

1. **There is no train/test split.** You make it, with a fixed `torch.Generator` seed recorded in the config — not an unseeded `random_split`, which gives a different test set every run and makes every number unreproducible.

2. **Three-way, not two-way.** Validation drives the stopping rule; test gets touched once, at the end. Phase 2 tunes hyperparameters — if tuning touches test, the comparison is contaminated, and that is precisely the class of protocol error this project exists to expose in other people's benchmarks. Getting the habit wrong here means getting the result wrong there.

3. **Normalisation stats from the train split only.** Channel means and stds over the whole dataset leak test statistics into training. Small effect at this scale, wrong habit, same category as trap 2.

4. **`random_split` returns `Subset`s of the *same* dataset object, which share one `.transform`.** So augmenting "the train set" augments validation too. The symptom is val accuracy that jitters for no reason and sits below train for reasons that look like overfitting and are not. This is why `make_splits` returns indices: build two `EuroSAT` instances with different transforms and `Subset` each by its own indices, or wrap `Subset` so it applies its own. **Most people hit this and never notice.**

5. **`num_workers > 0` on Windows.** Workers re-import the module, so a script needs the `if __name__ == "__main__":` guard and a notebook can hang outright. Start at `num_workers=0`, get it training, then raise it and measure whether it helped.

6. **`pin_memory=True` *and* `.to(device, non_blocking=True)`.** The pair is the point; neither does much alone.

7. **`model.train()` / `model.eval()`,** and `torch.no_grad()` around evaluation. Forgetting `eval()` leaves dropout and BatchNorm in training mode and your validation number is wrong in a direction that looks plausible.

8. **Accumulate `loss.item()`, not `loss`.** Summing the tensor retains the autograd graph for the whole epoch and runs you out of 8 GB.

9. **Never `.to(device)` inside `__getitem__`.** Worker processes cannot; transfer happens in the training loop.

10. **`cudnn.benchmark = True` trades determinism for speed** by picking algorithms per shape at runtime. Pick one and record which. For this milestone, determinism — P2 below depends on it.

## 7. Pass criteria

Fixed now, before the first run.

**P1 — it trains end-to-end on the GPU.** `next(model.parameters()).device` and a fetched batch's device are both `cuda`, and training loss decreases over an epoch. *This is the roadmap's Phase 0 exit criterion and it alone closes the phase.*

**P2 — your `Dataset` agrees with torchvision's.** `EuroSATRaw` and `EuroSAT` return the same length, the same `classes` in the same order, and identical `(tensor, label)` pairs on a seeded sample of indices. This is the milestone's only verification criterion — the one thing here checked against an independent implementation rather than against my own expectations. If it fails, the likely causes are class ordering (`sorted()`, §3), a transform applied in one and not the other, or a different PIL decode path.

**P3 — determinism.** Two runs at identical `cfg` produce identical training loss at every logged step. If they do not, the run is not reproducible and Phase 2 cannot attribute a difference to the optimiser — which is the entire experiment. If exact equality proves unattainable, record the achieved tolerance and the reason rather than relaxing the criterion silently.

**P4 — the split is reproducible across processes.** `make_splits(cfg)` in two separate Python processes returns identical test indices. A `Generator` seeded correctly passes; anything relying on global RNG state may not.

**P5 — optional, and I think worth the evening.** Five runs at fixed `cfg` and fixed `split_seed`, varying only `init_seed`; report the spread in final validation accuracy. This is the noise floor of your setup. Phase 2's claim is "optimiser A differs from B", and without a noise floor that claim has nothing to be measured against — you would have no way to say whether a gap is a result or a seed. The number itself will not transfer to Phase 2's model and dataset, but the harness and the habit do, and measuring it costs five short runs on a dataset small enough to make that free.

Accuracy is not on this list, deliberately. The 0.52 floor from §3.2 is a **bug detector, not a criterion** — if a run lands near it, stop and find the fault; if it clears it, that tells you the pipeline works and nothing more.

## 8. Not in scope

Each of these is a real thing to do and none of them belongs in this milestone.

- **An accuracy target** — §1.
- **Pretrained backbones** — measures ImageNet, not the pipeline.
- **Augmentation or LR schedule search** — that is Phase 2 with a matched budget, not an afternoon of guessing.
- **Mixed precision** — a Phase 3 memory constraint. Adding it here adds a failure mode to a milestone whose job is to have none.
- **Class imbalance handling** — measure the imbalance (§3), then decide in Phase 2 if it matters.

## 9. Closed: this model is throwaway, and that is fine

The question this section used to ask — what the parent project's scaling study actually trains on — is answered by its own roadmap. The ladder is 2M / 8M / 32M / 128M parameters in bf16, fitted against the Kaplan and Hoffmann scaling literature. That is a language-model ladder. Nobody builds a 128M-parameter CNN over 64×64 images.

So the model here is not a small version of that model, and `base_channels` is not a width knob for anything downstream. It stays in `RunConfig` — §4's rule is that everything affecting a number lives there, and width affects a number — but it is a fixed field, not an axis. Pick a value that trains in under five minutes and stop thinking about it.

Two consequences worth stating plainly, because they set how much care this deserves:

- **`model.py` and the image-specific half of `data.py` are throwaway.** They exist to be written once, correctly, by me.
- **What transfers is the discipline, not the code** — the config object holding everything that changes a number, the two separate seeds, the three-way split, the agreement check as an importable test.

This is a fluency exercise with one verification criterion attached. It is not a foundation, and treating it as one would be the most expensive mistake available here.

---

*Pre-registered 17 September 2026. Results, measured numbers and the resolution of the **verify** items above get appended when the milestone closes.*

---

# Results — 20 September 2026

Everything above this line was committed before the first run. Everything below
is measured. Environment: torch 2.11.0+cu128, torchvision 0.26.0+cu128,
Python 3.13, RTX 5070 Laptop (sm_120).

## Criteria

| | Criterion | Outcome |
|---|---|---|
| **P1** | trains end to end on GPU | **pass** — 3 epochs in 63.6 s, train loss 1.069 → 0.612 |
| **P2** | `EuroSATRaw` agrees with torchvision | **pass** — 300 seeded indices, `torch.equal`, exact |
| **P3** | two runs identical at every logged step | **pass** — exact equality, no tolerance required |
| **P4** | split reproducible across processes | **pass** — identical test indices in two separate processes |
| **P5** | seed-to-seed spread | **measured** — sd ≈ 0.031 in final validation accuracy |

P3 deserves the emphasis: the criterion allowed for recording an achieved
tolerance if exact equality proved unattainable. It did not. Two runs at one
config produce bit-identical `train_loss`, `val_loss` and `val_acc`, and the same
first-two-epoch values recur across separate invocations on different days.

## Resolution of the **verify** items

**Download host.** Confirmed for torchvision 0.26.0: pinned HuggingFace commit
`c877bcd43f099cd0196738f714544e355477f3fd`, md5 `c8fa014336c82ac7804f0398fcb19387`.
The trap in §3.1 is real as written — `_check_exists()` tests only that
`data/eurosat/2750` exists, not that extraction completed.

**N = 27,000**, confirmed directly.

**Per-class counts**, previously unmeasured:

| count | classes |
|---|---|
| 3,000 | AnnualCrop, Forest, HerbaceousVegetation, Residential, SeaLake |
| 2,500 | Highway, Industrial, PermanentCrop, River |
| 2,000 | Pasture |

Imbalance 1.5:1. The counts are clean multiples of 500, which is itself evidence
the extraction completed — a partial tree gives ragged numbers.

**Channel statistics**, over the 18,900 training images only:

| channel | mean | std |
|---|---|---|
| R | 0.3440 | 0.2024 |
| G | 0.3801 | 0.1370 |
| B | 0.4076 | 0.1158 |

Computed on the train split rather than all 27,000, per trap 3, so they differ
slightly from constants published over the full dataset. The ordering is
physically sensible — blue highest mean (Rayleigh scattering over land), red
widest spread (chlorophyll absorption separating vegetation from soil and
built-up) — which is a free check on the whole data path.

## Decisions the measurements drove

**The split is stratified.** Not because 1.5:1 is dangerous, but because of the
two-seed design in §4. Test is 4,050 images; under an unstratified split,
Pasture's share is hypergeometric with mean 300 and sd ≈ 15. Harmless for
accuracy, but it would make `split_seed` change both *which* images land in test
and *how many of each class*. Stratifying makes that knob mean exactly one thing.
Always on, and deliberately not a config field — a field would imply an axis to
sweep. Verified: test contains exactly 450 of each 3,000-image class, 375 of each
2,500, and 300 Pasture.

**Plain accuracy is adequate** at this imbalance; micro and macro barely diverge.
Per-class accuracy is logged anyway as a bug detector — one class near 0% is the
signature of a broken label mapping.

## The trap that is not in §6

`ImageFolder.make_dataset` builds its file list with `for fname in sorted(fnames)`
— a plain lexicographic string sort. EuroSAT filenames carry an unpadded integer,
so lexicographic and numeric order diverge from the second element on:

    AnnualCrop_1, AnnualCrop_10, AnnualCrop_100, AnnualCrop_1000, ... AnnualCrop_2

Sorting numerically — the more obviously *correct* thing to do — makes index *i*
a different image in each implementation. Both datasets remain individually
correct; only the correspondence breaks. §7 lists class ordering, transforms and
decode path as the likely causes of P2 failing. This was not among them and was
the one that actually fired.

`tests/test_data.py` carries a negative control: an `EuroSATRaw` whose `samples`
are re-sorted numerically must fail `assert_matches_reference`. Without it, P2
passing would only mean the check ran.

## P5 — the noise floor

Five runs, `epochs=2`, `split_seed=0` fixed, `init_seed` 0–4:

    0.7889, 0.7314, 0.7904, 0.7504, 0.7267

mean 0.7575, sd 0.0306, range 0.7267–0.7904 (spread 0.0637).

**Treat the sd as an order of magnitude.** With n = 5 the sampling error on a
standard deviation is roughly 1/√(2(n−1)) ≈ 35% relative, so this is "about 3
points", not 3.06.

**What it implies for Phase 2.** With five seeds per method, the standard error of
each mean is 0.031/√5 ≈ 0.014, so the SE of a difference is ≈ 0.019 and the
smallest resolvable gap at ~95% is about 4–5 percentage points. Resolvable
difference scales as 1/√n, so halving that threshold costs four times the runs.
This is a seed-budget input, available before the protocol is written rather than
after.

**And it is an upper bound.** Measured at 2 epochs, far from convergence, where
run-to-run divergence is largest. Whatever Phase 2 trains — per §9, not this model
— its own noise floor needs measuring at that protocol's settings before the seed
budget is fixed. The number here does not transfer; the practice of having one
does.

## What this does not establish

Accuracy, as §1 committed. A mediocre classifier running the full pipeline on
device clears every criterion above. The 0.52 floor from §3.2 was cleared
comfortably, which tells us the pipeline is not broken and nothing more. No
accuracy claim is made here or in the README.

## On writing verification code

Three of the checks written for this milestone initially could not fail:

- comparing `mine_image.size` to `ref_image.size` — on tensors `.size` is a bound
  method, so the comparison is `False` even for identical tensors; and had it been
  `.size()`, every EuroSAT image is 64×64×3, so it would have passed for any
  pairing including a fully scrambled index order
- the negative control binding `broken_samples = sorted(...)` to a new local
  instead of assigning to `broken.samples`, leaving the "broken" dataset correct
- `assert [expr]` rather than `assert expr` — a non-empty list is always truthy

Each ran, printed something encouraging, and tested nothing. Two were caught only
because the negative controls raised rather than printed. That is the argument for
`else: raise` over a print, and for writing a control that must fail alongside
every check that must pass.

---

*Milestone closed 20 September 2026. Phase 0 of the roadmap in
[training-dynamics](https://github.com/Bakri1851/training-dynamics) is complete.*