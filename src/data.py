from pathlib import Path
import random
from PIL import Image
from numpy import indices
import torch
from torch.utils.data import Dataset
from collections import defaultdict

# Define a custom dataset class for EuroSAT
class EuroSATRaw(Dataset):
    # Initialize the dataset with the root directory and optional transformations
    def __init__(self, root_dir, transform=None):
        # Call the parent class constructor
        super().__init__()

        # Set the root directory for the dataset, appending the specific path for EuroSAT
        self.root_dir = Path(root_dir) / "eurosat" / "2750" # root_dir is 

        # Set the transformation to be applied to the images, if any
        self.transform = transform

        # Get the list of classes (subdirectories) in the root directory and sort them.
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Create a list to hold the samples (image paths and their corresponding labels)
        self.samples = []

        # Iterate through each class directory and collect image paths along with their labels
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_path in sorted(cls_dir.glob("*.jpg"), key = lambda p : p.name):
                self.samples.append((img_path, self.class_to_idx[cls]))


    # Define the method to get the length of the dataset
    def __len__(self):
        return len(self.samples)  

    # Define the method to get an item from the dataset by index
    def __getitem__(self, idx):
        # Get the image path and label for the given index
        ipath, label = self.samples[idx]
        image = Image.open(ipath).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label



def assert_matches_reference(mine, reference, n, seed):

    assert len(mine) == len(reference), f"Length mismatch: {len(mine)} vs {len(reference)}"
    assert mine.classes == reference.classes, f"Classes mismatch: {mine.classes} vs {reference.classes}"


    indices = random.Random(seed).sample(range(len(mine)), n)

    for i in indices:
        mine_image, mine_label = mine[i]
        ref_image, ref_label = reference[i]

        assert (mine_label == ref_label), f"Label mismatch at index {i}: {mine_label} vs {ref_label}"
        assert torch.equal(mine_image, ref_image), f"Image mismatch at index {i}: {mine_image} vs {ref_image}"



def make_splits(cfg):

    # build the index of all samples in the dataset
    index  = EuroSATRaw(cfg.data_root, transform=None).samples

    # group the indices by class
    class_to_indices = defaultdict(list)

    # initialize the dictionary with empty lists for each class
    for i, (path, label) in enumerate(index):
        class_to_indices[label].append(i)

    # for each class, shuffle the indices and split them into train, val, and test
    # a generator is used to ensure that the shuffling is deterministic based on the provided seed
    g = torch.Generator().manual_seed(cfg.split_seed)

    train_idx, val_idx, test_idx = [], [], []

    for label in sorted(class_to_indices):
        indices = class_to_indices[label]

        # Shuffle the indices for the current class
        perm = torch.randperm(len(indices), generator=g)
        shuffled_indices = [indices[i] for i in perm.tolist()]

        # Calculate the number of samples for test and validation splits based on the provided configuration
        n_test = int(len(shuffled_indices) * cfg.test_frac)
        n_val = int(len(shuffled_indices) * cfg.val_frac)

        # Extend the respective lists with the shuffled indices for test, validation, and training splits
        test_idx.extend(shuffled_indices[:n_test])
        val_idx.extend(shuffled_indices[n_test:n_test + n_val])
        train_idx.extend(shuffled_indices[n_test + n_val:])

    return (
        train_idx,
        val_idx,
        test_idx,
    )


