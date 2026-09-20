import dataclasses
import json
import statistics
from pathlib import Path

from src.config import RunConfig
from src.train import train_one_run

SEEDS = range(5)
EPOCHS = 2
OUT = Path(__file__).resolve().parents[1] / "results" / "seed-spread.json"


def main():
    base = dataclasses.replace(RunConfig(), epochs=EPOCHS)
    print(f"{len(SEEDS)} runs; split_seed = {base.split_seed} fixed, "
          f"init_seed varying, epochs = {EPOCHS}")

    accs = []
    for seed in SEEDS:
        cfg = dataclasses.replace(base, init_seed=seed)
        result = train_one_run(cfg)
        accs.append(result.val_accuracy[-1])
        print(f"seed {seed} -> val acc = {result.val_accuracy[-1]:.4f}")

    assert len(accs) == len(SEEDS), "Not all runs completed successfully"

    mean = statistics.mean(accs)
    stdev = statistics.stdev(accs)

    print(f"    mean = {mean:.4f} +/- {stdev:.4f}")
    print(f"    range = {min(accs):.4f} .. {max(accs):.4f}")
    print(f"    (spread = {max(accs) - min(accs):.4f})")
    print()
    print(f"    noise floor of order {stdev:.3f} in final validation accuracy, due to random init_seed")
    print(f"    n = 5, so the stdev itself carries about +/- 35% sampling error")

    # Persist it. A number quoted in the README with nothing behind it in the
    # repository is the same failure as a claim with no measurement, one size down.
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "split_seed": base.split_seed,
        "val_accuracy": accs,
        "mean": mean,
        "stdev": stdev,
        "spread": max(accs) - min(accs),
    }, indent=2) + "\n")
    print(f"    wrote {OUT}")


if __name__ == "__main__":
    main()
