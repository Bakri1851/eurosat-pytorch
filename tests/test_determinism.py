
import dataclasses
from src.train import train_one_run
from src.config import RunConfig

EPOCHS = 2



def compare(name, a, b):
    print(f"{name}")
    print(f"    run 1: {[round(v, 6) for v in a]}")
    print(f"    run 2: {[round(v, 6) for v in b]}") 

    if a == b:
        print(" identical")
        return True
    first = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    worst = max(abs(x - y) for x, y in zip(a, b))
    print(f"diverges at epoch {first}, worst difference {worst:.6f}")
    return False

def main():
    cfg = dataclasses.replace(RunConfig(), epochs=EPOCHS)
    print(f"two runs at init_seed={cfg.init_seed}, split_seed={cfg.split_seed}, epochs={cfg.epochs}, deterministic={cfg.deterministic}")  #

    r1 = train_one_run(cfg)
    r2 = train_one_run(cfg)

    print

    ok = all([
        compare("train loss", r1.train_loss, r2.train_loss),
        compare("val loss", r1.val_loss, r2.val_loss),
        compare("val accuracy", r1.val_accuracy, r2.val_accuracy),
    ])

    assert ok, (
        f"runs diverged: "
        f"train loss {r1.train_loss} vs {r2.train_loss}, "
        f"val loss {r1.val_loss} vs {r2.val_loss}, "
        f"val accuracy {r1.val_accuracy} vs {r2.val_accuracy}"
    )

    other = dataclasses.replace(cfg, init_seed=cfg.init_seed + 1)
    print(f"negative test: init_seed= {other.init_seed}")
    r3 = train_one_run(other)
    print(f"    seed {cfg.init_seed}: {[round(v, 6) for v in r1.train_loss]}")
    print(f"    seed {other.init_seed}: {[round(v, 6) for v in r3.train_loss]}")

    if r1.train_loss == r3.train_loss:
        raise SystemExit(f"unexpectedly identical train loss for different seeds: {r1.train_loss} vs {r3.train_loss}")
    print(" differ - the seed is having an effect, as expected")
if __name__ == "__main__":
    main()