import dataclasses
import statistics

from src.config import RunConfig
from src.train import train_one_run

SEEDS = range(5)
EPOCHS = 2


def main():
    base = dataclasses.replace(RunConfig(), epochs=EPOCHS)
    print(f"{len(SEEDS)} runs; split_seed = {base.split_seed} fixed"
          f"init_seed varying, epochs = {EPOCHS}")

    accs = []
    for seed in SEEDS:
        cfg = dataclasses.replace(base, init_seed = seed)
        result = train_one_run(cfg)
        accs.append(result.val_accuracy[-1])
        print(f"seed {seed} -> val acc = {result.val_accuracy[-1]:.4f}")

    assert len(accs) == len(SEEDS), "Not all runs completed successfully"

    mean = statistics.mean(accs)
    stdev = statistics.stdev(accs)

    print(f"    mean = {mean:.4f} ± {stdev:.4f}")
    print(f"    range = {min(accs):.4f} .. {max(accs):.4f}")
    print(f"    (spread = {max(accs) - min(accs):.4f})")
    print()
    print(f"    noise floor of order {stdev:.3f} in final validation accuracy, due to random init_seed")
    print(f"    n = 5 so the stdev carries +/- 35% samplling error")

if __name__ == "__main__":
    main()