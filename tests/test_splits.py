from src.data import EuroSATRaw, make_splits
from collections import Counter
from src.config import RunConfig
import subprocess
import sys
from collections import Counter


train, val, test = make_splits(RunConfig())

expected_test_distribution = [(0, 450), (1, 450), (2, 450), (3, 375), (4, 375), (5, 300), (6, 375), (7, 450), (8, 375), (9, 450)]

CHILD = """
import hashlib
from src.config import RunConfig
from src.data import make_splits
_, _, test = make_splits(RunConfig())
print(hashlib.sha256(repr(test).encode()).hexdigest())
"""


def split_hash_in_subprocess():
    """sha256 of test_idx, computed in a separate Python process."""
    result = subprocess.run(
        [sys.executable, "-c", CHILD], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(f"child process failed:\n{result.stderr}")
    return result.stdout.strip()


def main():
    cfg = RunConfig()
    train, val, test = make_splits(cfg)
    ds = EuroSATRaw(cfg.data_root, transform=None)

    # --- partition ---------------------------------------------------------
    total = len(train) + len(val) + len(test)
    assert total == len(ds), f"splits sum to {total}, expected {len(ds)}"
    assert set(train).isdisjoint(val), "train and val overlap"
    assert set(train).isdisjoint(test), "train and test overlap"
    assert set(val).isdisjoint(test), "val and test overlap"
    print(f"partition OK — {len(train)} / {len(val)} / {len(test)} = {total}")

    # --- stratification: 15% of each class, not 15% overall ----------------
    per_class = sorted(Counter(ds.samples[i][1] for i in test).items())
    assert per_class == expected_test_distribution, f"not stratified: {per_class}"
    print("stratification OK — per-class test counts are 15% of each class")

    # --- P4: two separate processes agree ----------------------------------
    h1 = split_hash_in_subprocess()
    h2 = split_hash_in_subprocess()
    assert h1 and h2, "subprocess produced no output"
    assert h1 == h2, f"split differs across processes:\n  {h1}\n  {h2}"
    print(f"P4 pass — identical test indices across processes ({h1[:16]}…)")


if __name__ == "__main__":
    main()