from pathlib import Path
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

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

