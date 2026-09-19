from pathlib import Path
import random
from PIL import Image
from numpy import indices
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from collections import defaultdict
from torchvision import transforms



# A PyTorch "Dataset" is just an object that knows how many samples it has
# (__len__) and how to fetch any one of them by index (__getitem__). PyTorch's
# DataLoader (used later in make_loaders) relies on exactly those two methods to
# read data in batches, so every custom dataset subclasses Dataset and
# implements them. This one wraps the raw EuroSAT image files on disk.
class EuroSATRaw(Dataset):
    def __init__(self, root_dir, transform=None):
        # Required boilerplate when subclassing Dataset.
        super().__init__()

        # EuroSAT images are laid out on disk as:
        #   <root_dir>/eurosat/2750/<class_name>/<image>.jpg
        # e.g. .../eurosat/2750/Forest/Forest_123.jpg
        self.root_dir = Path(root_dir) / "eurosat" / "2750"

        # `transform` is an optional preprocessing function (often a chain of
        # steps, see transforms.Compose elsewhere in this file) applied to each
        # image right before it's returned, e.g. converting it to a tensor,
        # normalizing pixel values, or randomly flipping it.
        self.transform = transform

        # Each subdirectory name is a class label, e.g. "Forest", "River", etc.
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        # Models work with numbers, not strings, so map each class name to an
        # integer id, e.g. {"AnnualCrop": 0, "Forest": 1, ...}.
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # We don't load any actual image data yet - just remember, for every
        # image file, its path and integer label. Images are only opened and
        # read from disk lazily, inside __getitem__, when a sample is requested.
        self.samples = []

        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_path in sorted(cls_dir.glob("*.jpg"), key = lambda p : p.name):
                self.samples.append((img_path, self.class_to_idx[cls]))


    def __len__(self):
        # PyTorch calls this whenever it needs to know the dataset size,
        # e.g. len(dataset).
        return len(self.samples)

    def __getitem__(self, idx):
        # PyTorch calls this to fetch sample number `idx`, e.g. dataset[idx],
        # or automatically many times over inside a DataLoader.
        ipath, label = self.samples[idx]
        image = Image.open(ipath).convert("RGB")

        # Run the preprocessing pipeline (if any) right before returning, so
        # every consumer of this dataset receives already-processed images.
        if self.transform:
            image = self.transform(image)
        return image, label



# Sanity-check helper: compares our EuroSATRaw dataset against some other
# "reference" dataset implementation (e.g. torchvision's built-in EuroSAT
# dataset, assumed correct) to make sure both agree on images and labels.
# Checking every sample would be slow, so instead it spot-checks `n` randomly
# chosen samples, using a fixed `seed` so the check is reproducible.
def assert_matches_reference(mine, reference, n, seed):

    assert len(mine) == len(reference), f"Length mismatch: {len(mine)} vs {len(reference)}"
    assert mine.classes == reference.classes, f"Classes mismatch: {mine.classes} vs {reference.classes}"

    # Pick n random sample indices (deterministic given `seed`) to check.
    indices = random.Random(seed).sample(range(len(mine)), n)

    for i in indices:
        mine_image, mine_label = mine[i]
        ref_image, ref_label = reference[i]

        assert (mine_label == ref_label), f"Label mismatch at index {i}: {mine_label} vs {ref_label}"
        # torch.equal checks every pixel value matches exactly, not just the shape.
        assert torch.equal(mine_image, ref_image), f"Image mismatch at index {i}: {mine_image} vs {ref_image}"



# Decides which samples go into the training, validation, and test sets.
# Returns three lists of integer indices (not the actual images/data) - these
# get used later (e.g. in make_loaders) to build Subsets of the full dataset.
def make_splits(cfg):

    # We only need this to know how many samples exist and which class each
    # belongs to - no images are loaded, since transform=None and we just read
    # the .samples list of (path, label) pairs.
    index  = EuroSATRaw(cfg.data_root, transform=None).samples

    # Group sample indices by class label, e.g. {0: [i1, i2, ...], 1: [...]}.
    # We split within each class separately (a "stratified" split) rather than
    # shuffling the whole dataset at once, so that train/val/test each end up
    # with roughly the same proportion of every class.
    class_to_indices = defaultdict(list)

    for i, (path, label) in enumerate(index):
        class_to_indices[label].append(i)

    # A seeded random generator makes the split reproducible: running this
    # again with the same cfg.split_seed always produces the same train/val/test
    # assignment.
    g = torch.Generator().manual_seed(cfg.split_seed)

    train_idx, val_idx, test_idx = [], [], []

    for label in sorted(class_to_indices):
        indices = class_to_indices[label]

        # Randomly reorder this class's indices, then slice the shuffled list
        # into test/val/train chunks below.
        perm = torch.randperm(len(indices), generator=g)
        shuffled_indices = [indices[i] for i in perm.tolist()]

        # How many of this class's samples go to test/val, based on the
        # fractions in cfg (e.g. test_frac=0.2 means 20% of this class's
        # samples go to the test set). Whatever remains goes to training.
        n_test = int(len(shuffled_indices) * cfg.test_frac)
        n_val = int(len(shuffled_indices) * cfg.val_frac)

        # Slice the shuffled list into three non-overlapping chunks: test gets
        # the first n_test, val gets the next n_val, and train gets the rest.
        test_idx.extend(shuffled_indices[:n_test])
        val_idx.extend(shuffled_indices[n_test:n_test + n_val])
        train_idx.extend(shuffled_indices[n_test + n_val:])

    return (
        train_idx,
        val_idx,
        test_idx,
    )

# Computes the per-channel (R, G, B) mean and standard deviation across a set
# of images (the ones at `indices` in `dataset`), in [0, 1] pixel-value units
# (i.e. after ToTensor, before any normalization). These stats later feed into
# transforms.Normalize so input pixel values get rescaled to roughly zero mean
# / unit variance, which tends to make model training more stable.
def channel_stats(dataset, indices):

    # We accumulate running totals instead of loading every image into memory
    # at once, so this works no matter how many images `indices` covers.
    # float64 is used for precision, since we're summing over potentially
    # millions of individual pixel values.
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)

    for i in indices:
        img, _ = dataset[i]  # img has shape (C, H, W): 3 channels, height, width
        # Sum pixel values over the height/width dimensions, leaving one
        # running total per channel. We also track the sum of *squared* pixel
        # values (total_sq), which the std calculation below needs.
        total += img.sum(dim = (1,2))
        total_sq += (img ** 2).sum(dim = (1,2))

    # mean = (sum of all pixel values) / (number of pixel values), per channel.
    mean = total / (len(indices) * img.shape[1] * img.shape[2])

    # Standard deviation via std = sqrt(E[X^2] - E[X]^2) - a shortcut formula
    # that lets us compute std from running sums, without a second pass over
    # the data to compute (x - mean)^2 directly for every pixel.
    std = (total_sq / len(indices) / img.shape[1] / img.shape[2] - mean ** 2) ** 0.5

    # Unpack each 3-element tensor into plain Python floats, one per channel.
    mean_r, mean_g, mean_b = mean.tolist()
    std_r, std_g, std_b = std.tolist()

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)



# This function creates data loaders for training, validation, and testing based on the provided configuration.
# A data loader is an iterable over a dataset, providing batches of data for training or evaluation.
def make_loaders(cfg):
    # Get which sample indices belong to train/val/test.
    # These are just lists of integer positions into the dataset, no image data yet.
    train_idx, val_idx, test_idx = make_splits(cfg)

    # Build a "plain" version of the dataset that only converts images to tensors
    # (pixel values scaled to [0, 1]) with no normalization applied yet.
    # We need this first so we can measure the *real* pixel statistics below,
    # before we know what values to normalize by.
    plain = EuroSATRaw(cfg.data_root, transform=transforms.ToTensor())

    # Compute the per-channel (R, G, B) mean and standard deviation, using only the
    # training images. We deliberately exclude val/test here so that no information
    # about the validation/test data leaks into how we preprocess the training data.
    mean, std = channel_stats(plain, train_idx)

    # transforms.Compose chains together a sequence of image preprocessing steps that
    # get applied, in order, every time an image is loaded.
    #
    # Training pipeline: randomly flip images horizontally/vertically (this is "data
    # augmentation" - it creates variety in what the model sees each epoch, so it
    # generalizes better instead of memorizing the exact training images), convert
    # to a tensor, then normalize each channel using the stats computed above so
    # pixel values have roughly zero mean / unit variance (this tends to make
    # training more stable).
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Evaluation pipeline: same tensor conversion + normalization, but no random
    # flips. We want val/test evaluation to be consistent and reproducible every
    # time, not randomly altered.
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create two datasets that wrap the exact same underlying images, but each
    # attaches a different transform pipeline: one with augmentation for training,
    # one without for evaluation.
    train_parent = EuroSATRaw(cfg.data_root, transform=train_tf)
    eval_parent = EuroSATRaw(cfg.data_root, transform=eval_tf)

    # Subset picks out just the samples at the given indices from a dataset,
    # without copying any data - it's just a lightweight "view" into train_parent
    # restricted to the training indices.
    #
    # A DataLoader wraps a dataset so you can iterate over it in batches instead of
    # one sample at a time. Key arguments:
    #   - batch_size: how many samples to group into each batch
    #   - shuffle: whether to randomize the order of samples each epoch
    #   - num_workers: number of background processes used to load data in
    #     parallel, so the GPU/CPU doesn't sit idle waiting for images to be read
    #   - pin_memory: speeds up transferring batches to the GPU
    #   - generator: a random number generator seeded with cfg.init_seed, so the
    #     shuffling order is reproducible across runs given the same seed
    train_loader = DataLoader(
        Subset(train_parent, train_idx),
        generator = torch.Generator().manual_seed(cfg.init_seed),
        batch_size = cfg.batch_size,
        num_workers = cfg.num_workers,
        pin_memory = cfg.pin_memory,
        shuffle = True,  # shuffle each epoch during training
         )

    # Validation/test loaders: shuffle=False because the order doesn't matter for
    # evaluation, and keeping it fixed makes results easier to compare/debug.
    val_loader = DataLoader(
        Subset(eval_parent, val_idx),
        batch_size = cfg.batch_size,
        num_workers = cfg.num_workers,
        pin_memory = cfg.pin_memory,
        shuffle = False,
    )

    test_loader = DataLoader(
        Subset(eval_parent, test_idx),
        batch_size = cfg.batch_size,
        num_workers = cfg.num_workers,
        pin_memory = cfg.pin_memory,
        shuffle = False,
    )

    # Return all three loaders, ready to be iterated over in a training/eval loop.
    return (train_loader, val_loader, test_loader)

# quick channel stats test on dataset

