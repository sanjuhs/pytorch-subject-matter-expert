# Day 07 — 30-Second Shorts (Data Loading + Preprocessing + Augmentation)

These are **script-ready** shorts. Each short includes:
- **Title**
- **Hook** (1 sentence)
- **Talk track** (what you say, ~20 seconds)
- **Code** (what you show)
- **Common mistake** (quick pitfall)

---

## Short 01 — Day 7: Great Models Need Great Data Pipelines

**Hook**
Most ML “bugs” aren’t model bugs — they’re data bugs.

**Talk track**
- “Today we master data pipelines: Dataset, DataLoader, transforms, augmentation.”
- “You’ll learn how to load custom folders, speed up loading, and handle imbalance.”

**Code**
```python
# Today: Dataset + DataLoader + transforms + augmentation + imbalance
```

**Common mistake**
- Spending days tuning models before validating the data pipeline.

---

## Short 02 — Dataset vs DataLoader (Who Does What?)

**Hook**
Dataset defines samples; DataLoader defines batches.

**Talk track**
- “Dataset: how to fetch ONE item.”
- “DataLoader: how to batch, shuffle, and load in parallel.”

**Code**
```python
# Dataset: __len__ + __getitem__
# DataLoader: batching + shuffling + workers
```

**Common mistake**
- Trying to do batching logic inside `__getitem__` (keep it per-sample).

---

## Short 03 — Minimal Custom Dataset Skeleton

**Hook**
Every custom dataset needs 3 methods.

**Talk track**
- “Implement `__init__`, `__len__`, `__getitem__`.”
- “Return `(x, y)` from `__getitem__`.”

**Code**
```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None
```

**Common mistake**
- Returning only `x` and forgetting labels (then training breaks later).

---

## Short 04 — SimpleDataset: Synthetic Data Example

**Hook**
Practice Dataset patterns with synthetic data first.

**Talk track**
- “This dataset returns a feature vector and a binary label.”
- “It’s the perfect minimal example.”

**Code**
```python
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, size=100):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

**Common mistake**
- Returning labels as float when you need class indices (e.g., CrossEntropyLoss needs `long`).

---

## Short 05 — DataLoader: Batch + Shuffle + Workers

**Hook**
DataLoader is how you feed your GPU efficiently.

**Talk track**
- “Batch size controls how many samples per step.”
- “Shuffle training data, don’t shuffle validation.”
- “Workers load batches in parallel.”

**Code**
```python
from torch.utils.data import DataLoader

ds = SimpleDataset(100)
loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
```

**Common mistake**
- Using `shuffle=True` with a `sampler` (PyTorch will complain).

---

## Short 06 — Print Batch Shapes (Your Daily Sanity Check)

**Hook**
If you don’t print shapes, you will suffer.

**Talk track**
- “A batch adds a new dimension: `(batch, features)`.”
- “Check shapes early.”

**Code**
```python
for x, y in loader:
    print(x.shape, y.shape)
    break
```

**Common mistake**
- Thinking `x.shape` is `(features,)` inside the loop (it’s batched).

---

## Short 07 — pin_memory=True (Why It Matters)

**Hook**
pin_memory can speed CPU→GPU transfer.

**Talk track**
- “Pinned memory allows faster DMA transfers to GPU.”
- “Useful when training on CUDA with DataLoader workers.”

**Code**
```python
loader = DataLoader(ds, batch_size=128, num_workers=4, pin_memory=True)
```

**Common mistake**
- Expecting `pin_memory` to help on CPU-only training (it mostly helps with GPUs).

---

## Short 08 — The Folder Dataset Problem

**Hook**
Real projects usually store images in folders by class.

**Talk track**
- “Common structure: `root/class_a/*.jpg`, `root/class_b/*.jpg`.”
- “We’ll build a dataset that maps folders to labels.”

**Code**
```python
# root/
#   cats/*.jpg
#   dogs/*.jpg
```

**Common mistake**
- Having non-image files in the folder and crashing during load.

---

## Short 09 — CustomImageDataset: Collect Paths + Labels

**Hook**
Step 1: build an index of file paths and class ids.

**Talk track**
- “List class folders.”
- “Assign an integer id per class.”
- “Store image paths + labels.”

**Code**
```python
import os, glob
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.labels = [], []
        self.class_to_idx = {}

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.class_to_idx[class_name] = idx
            for p in glob.glob(os.path.join(class_dir, "*.jpg")):
                self.image_paths.append(p)
                self.labels.append(idx)
```

**Common mistake**
- Not sorting class names → label ids change run-to-run.

---

## Short 10 — __getitem__: Load Image with PIL and Convert to RGB

**Hook**
Always standardize image mode.

**Talk track**
- “Use PIL to open the image.”
- “Convert to RGB so you always have 3 channels.”
- “Apply transforms if provided.”

**Code**
```python
from PIL import Image

def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    image = Image.open(img_path).convert("RGB")
    label = self.labels[idx]
    if self.transform:
        image = self.transform(image)
    return image, label
```

**Common mistake**
- Forgetting `.convert("RGB")` and getting inconsistent shapes across images.

---

## Short 11 — Transforms: Resize → ToTensor → Normalize

**Hook**
Transforms are your preprocessing pipeline.

**Talk track**
- “Resize to a consistent size.”
- “ToTensor converts to tensor.”
- “Normalize uses dataset mean/std (ImageNet stats are common for transfer learning).”

**Code**
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

**Common mistake**
- Using 1-channel mean/std on RGB images (needs 3 values).

---

## Short 12 — Data Augmentation: The Goal

**Hook**
Augmentation makes your dataset “bigger” without collecting new data.

**Talk track**
- “Augmentation teaches invariances: flips, crops, rotations.”
- “It improves generalization.”

**Code**
```python
# Augmentations: random transforms applied during training
```

**Common mistake**
- Over-augmenting so images no longer match the label.

---

## Short 13 — RandomHorizontalFlip (Most Common Augmentation)

**Hook**
Horizontal flips are free accuracy for many datasets.

**Talk track**
- “Flip left-right with some probability.”
- “Great for many natural images.”

**Code**
```python
from torchvision import transforms

aug = transforms.RandomHorizontalFlip(p=0.5)
```

**Common mistake**
- Flipping when it changes the meaning (e.g., text recognition, digits like 2/5 maybe).

---

## Short 14 — RandomCrop + Resize (Classic Recipe)

**Hook**
Crops force robustness to position.

**Talk track**
- “RandomCrop changes framing.”
- “Resize back to expected input size.”

**Code**
```python
from torchvision import transforms

aug = transforms.Compose([
    transforms.RandomCrop(200),
    transforms.Resize(256),
])
```

**Common mistake**
- Cropping too aggressively and removing the object entirely.

---

## Short 15 — RandomRotation (Small Angles)

**Hook**
Small rotations add viewpoint robustness.

**Talk track**
- “Rotate up to a small degree.”
- “Keeps label semantics but adds variety.”

**Code**
```python
from torchvision import transforms

aug = transforms.RandomRotation(degrees=15)
```

**Common mistake**
- Using huge rotations and destroying label meaning.

---

## Short 16 — ColorJitter (Lighting Robustness)

**Hook**
Teach your model that lighting changes don’t change the label.

**Talk track**
- “Brightness/contrast/saturation/hue changes simulate real-world lighting.”

**Code**
```python
from torchvision import transforms

aug = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
```

**Common mistake**
- Too strong jitter makes images unrealistic and hurts training.

---

## Short 17 — Train vs Val Transforms (Golden Rule)

**Hook**
Augment TRAIN only — never VAL/TEST.

**Talk track**
- “Train transform can be random.”
- “Val/test should be deterministic and stable.”

**Code**
```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
])
```

**Common mistake**
- Using random transforms in validation and getting noisy/unstable metrics.

---

## Short 18 — CIFAR-10 with Augmentation (Practical Example)

**Hook**
This is what augmentation looks like in real code.

**Talk track**
- “CIFAR-10 is 32x32 RGB.”
- “Use padding + random crop + horizontal flip.”

**Code**
```python
from torchvision import datasets, transforms

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
```

**Common mistake**
- Forgetting CIFAR is RGB and using grayscale normalization stats.

---

## Short 19 — Visualize Augmentation (Pro Tip)

**Hook**
Always test augmentations visually first.

**Talk track**
- “Call `dataset[0]` multiple times and see different augmented versions.”
- “If it looks wrong, it is wrong.”

**Code**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 2))
for i in range(5):
    img, _ = train_ds[0]
    plt.subplot(1, 5, i+1)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis("off")
plt.show()
```

**Common mistake**
- Forgetting to de-normalize before plotting and thinking augmentation broke colors.

---

## Short 20 — Imbalanced Datasets: What It Means

**Hook**
If one class dominates, accuracy can be misleading.

**Talk track**
- “Imbalanced data means some classes have far fewer samples.”
- “You need balancing strategies: sampler, loss weights, or resampling.”

**Code**
```python
# imbalance -> sampler or weighted loss or resampling
```

**Common mistake**
- Predicting the majority class always and getting high accuracy.

---

## Short 21 — WeightedRandomSampler (Balance Your Batches)

**Hook**
Sampler can force balanced sampling without changing the dataset.

**Talk track**
- “Compute class counts.”
- “Give minority class higher sampling weight.”
- “Use sampler in DataLoader.”

**Code**
```python
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader

class_counts = torch.bincount(dataset.labels)
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[dataset.labels]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

**Common mistake**
- Setting `shuffle=True` along with `sampler` (don’t).

---

## Short 22 — Weighted Loss (When You Can’t Resample)

**Hook**
Weight the loss so minority mistakes matter more.

**Talk track**
- “For binary classification with logits, use `BCEWithLogitsLoss(pos_weight=...)`.”
- “It changes gradient contributions.”

**Code**
```python
import torch
import torch.nn as nn

pos_weight = torch.tensor([10.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Common mistake**
- Using weighted loss incorrectly for multi-class; for CE use `CrossEntropyLoss(weight=...)`.

---

## Short 23 — Oversample vs Undersample (Tradeoffs)

**Hook**
Sampling strategies always trade bias vs variance.

**Talk track**
- “Oversample minority: more repeats, risk overfitting minority.”
- “Undersample majority: throw away data, but more balanced.”

**Code**
```python
# Oversample minority OR undersample majority
```

**Common mistake**
- Undersampling so hard you lose the majority class diversity.

---

## Short 24 — Custom collate_fn: Variable-Length Sequences

**Hook**
If samples have different sizes, you need a custom collate.

**Talk track**
- “Default collate stacks tensors — it fails on variable lengths.”
- “Collate function can pad sequences and return lengths.”

**Code**
```python
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    max_len = lengths.max()
    padded = torch.zeros(len(sequences), max_len, sequences[0].size(1))
    for i, s in enumerate(sequences):
        padded[i, :len(s)] = s
    return padded, torch.stack(labels), lengths
```

**Common mistake**
- Not returning lengths/mask, then the model can’t distinguish padding from real tokens.

---

## Short 25 — DataLoader Speed Knobs (The 4 Settings)

**Hook**
These 4 flags can drastically speed training.

**Talk track**
- “`num_workers`: parallelism.”
- “`pin_memory`: faster GPU transfer.”
- “`persistent_workers`: reuse workers.”
- “`prefetch_factor`: preload batches.”

**Code**
```python
loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
```

**Common mistake**
- Setting `num_workers` too high on constrained environments → crashes/hangs.

---

## Short 26 — persistent_workers=True (Why It Helps)

**Hook**
Spawning workers every epoch is slow.

**Talk track**
- “Persistent workers keep processes alive between epochs.”
- “Faster epoch-to-epoch iteration.”

**Code**
```python
loader = DataLoader(train_ds, batch_size=128, num_workers=4, persistent_workers=True)
```

**Common mistake**
- Using persistent workers with `num_workers=0` (no effect).

---

## Short 27 — Normalize With Dataset Statistics (Not Random Values)

**Hook**
Bad normalization can slow convergence.

**Talk track**
- “Use correct mean/std for the dataset.”
- “ImageNet stats are common for transfer learning.”

**Code**
```python
# CIFAR-10 mean/std example:
# mean=(0.4914, 0.4822, 0.4465)
# std =(0.2023, 0.1994, 0.2010)
```

**Common mistake**
- Using MNIST normalization for CIFAR-10 (wrong channels and stats).

---

## Short 28 — End-to-End CIFAR-10 Pipeline Skeleton

**Hook**
This is a production-style input pipeline.

**Talk track**
- “Train transform is random augmentation.”
- “Val/test transform is deterministic.”
- “Split train into train/val.”

**Code**
```python
from torch.utils.data import DataLoader, random_split

train_size = int(0.9 * len(train_ds))
val_size = len(train_ds) - train_size
train_ds2, val_ds = random_split(train_ds, [train_size, val_size])

train_loader = DataLoader(train_ds2, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
```

**Common mistake**
- Splitting AFTER augmentation in a way that leaks randomness into validation (keep val transforms deterministic).

---

## Short 29 — When Augmentation Hurts (Know When to Stop)

**Hook**
Augmentation is powerful — but not always helpful.

**Talk track**
- “If augmentations change class identity, you’ll hurt performance.”
- “Use domain knowledge: flipping text is bad; flipping animals is usually fine.”

**Code**
```python
# Rule: augmentations must preserve the label.
```

**Common mistake**
- Applying random perspective/rotation too aggressively for small images like CIFAR-10.

---

## Short 30 — Debugging Data Pipelines (The Checklist)

**Hook**
Debug data before debugging the model.

**Talk track**
- “Print shapes and dtypes.”
- “Visualize a batch.”
- “Verify labels and class mapping.”

**Code**
```python
# Debug checklist:
# - print batch shapes
# - visualize samples
# - confirm label mapping
# - check normalization ranges
```

**Common mistake**
- Training for hours before realizing labels were wrong.

---

## Short 31 — Best Practices Summary (Data Edition)

**Hook**
Good data pipelines make models easier.

**Talk track**
- “Different transforms for train vs val/test.”
- “Use efficient DataLoader settings.”
- “Handle imbalance intentionally.”

**Code**
```python
# Train: random aug
# Val/Test: deterministic
# DataLoader: workers + pin_memory
```

**Common mistake**
- Changing transforms mid-training without tracking experiment settings.

---

## Short 32 — Preview: Day 8 = CNNs (Now Data Makes Sense)

**Hook**
Now that your pipeline is solid, CNNs will shine.

**Talk track**
- “Tomorrow we’ll build CNNs for images.”
- “Data augmentation + CNNs is a powerful combo.”

**Code**
```python
# Day 8: Convolutions + pooling + first CNN
```

**Common mistake**
- Jumping to CNNs before you can trust your data pipeline.


