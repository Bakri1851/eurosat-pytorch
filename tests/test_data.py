from torchvision import transforms
from torchvision.datasets import EuroSAT

from src.data import EuroSATRaw, assert_matches_reference

N = 300
seed = 0

def numeric_key(sample):
    """Order by (label, integer suffix) — the ordering ImageFolder does NOT use.

    EuroSAT filenames carry an unpadded integer, so numeric and lexicographic
    order diverge from the second element on: _1, _2, _3 against _1, _10, _100.
    """
    path, label = sample
    return label , int(path.stem.rsplit("_",1)[-1])


def main():
    t = transforms.ToTensor()

    reference = EuroSAT(root = "data", transform=t)

    myDataset = EuroSATRaw("data", transform=t)

    # positive - the real dataset agrees with torch vision
    assert_matches_reference(myDataset, reference, n=N, seed=seed)
    print(f"Success: {N} samples from myDataset match reference dataset with seed {seed}.")


    # negative - a broken dataset must not agree
    broken = EuroSATRaw("data", transform=t)
    broken.samples = sorted(broken.samples, key=numeric_key)

    try:
        assert_matches_reference(broken, reference, n=N, seed=seed)
    except AssertionError:
        print("negative control: broken dataset does not match reference dataset as expected.")
    else:
        raise SystemExit("Failure: broken dataset unexpectedly matches reference dataset.")

    

if __name__ == "__main__":
    main()

