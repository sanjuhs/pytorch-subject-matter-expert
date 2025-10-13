# Day 7: Data Loading, Preprocessing, and Augmentation

**Duration**: 10 minutes
**Platform**: Google Colab

---

## Video Script

### Introduction (30 seconds)

**"Welcome to Day 7 - mastering data pipelines and augmentation!"**

Great models need great data. Today we'll learn:
- Creating custom datasets with `Dataset` class
- Data augmentation techniques (rotation, flipping, cropping)
- Efficient data loading with `DataLoader`
- Handling images from folders
- Dealing with imbalanced datasets
- Best practices for data preprocessing

By the end, you'll build production-ready data pipelines.

---

### Understanding PyTorch's Dataset and DataLoader (1.5 minutes)

**"The foundation of PyTorch data loading"**

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom Dataset class
class SimpleDataset(Dataset):
    """
    All PyTorch datasets must inherit from Dataset and implement:
    1. __init__: Initialize data
    2. __len__: Return dataset size
    3. __getitem__: Return one sample by index
    """

    def __init__(self, size=100):
        # Generate synthetic data
        self.data = torch.randn(size, 10)  # 100 samples, 10 features
        self.labels = torch.randint(0, 2, (size,))  # Binary labels

    def __len__(self):
        # Return total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Return sample at index idx
        return self.data[idx], self.labels[idx]

# Create dataset
dataset = SimpleDataset(size=100)
print(f"Dataset size: {len(dataset)}")

# Access individual samples
sample, label = dataset[0]
print(f"Sample shape: {sample.shape}")
print(f"Label: {label}")

# DataLoader: Batching, shuffling, parallel loading
dataloader = DataLoader(
    dataset,
    batch_size=16,      # Number of samples per batch
    shuffle=True,       # Shuffle data each epoch
    num_workers=2,      # Parallel data loading
    pin_memory=True     # Faster GPU transfer (if using CUDA)
)

print(f"\nNumber of batches: {len(dataloader)}")

# Iterate through batches
for batch_idx, (data, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: data shape = {data.shape}, labels shape = {labels.shape}")
    if batch_idx == 2:  # Show first 3 batches
        break
```

---

### Creating a Custom Image Dataset (2 minutes)

**"Loading images from a folder structure"**

```python
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob

class CustomImageDataset(Dataset):
    """
    Load images from folder structure:
    root/
        class_a/
            img1.jpg
            img2.jpg
        class_b/
            img3.jpg
            img4.jpg
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Root directory with class subdirectories
            transform: Optional transforms to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # Iterate through subdirectories (each is a class)
        class_names = sorted(os.listdir(root_dir))
        for idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            self.class_to_idx[class_name] = idx

            # Get all images in this class
            for img_path in glob.glob(os.path.join(class_dir, '*.jpg')):
                self.image_paths.append(img_path)
                self.labels.append(idx)

        print(f"Found {len(self.image_paths)} images in {len(self.class_to_idx)} classes")
        print(f"Classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Get label
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage (create dummy data first)
import os
os.makedirs('./dummy_data/cats', exist_ok=True)
os.makedirs('./dummy_data/dogs', exist_ok=True)

# In real scenario, you'd have actual images
# Let's create dummy images for demonstration
from PIL import Image
import numpy as np

for i in range(5):
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(f'./dummy_data/cats/cat_{i}.jpg')
    img.save(f'./dummy_data/dogs/dog_{i}.jpg')

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = CustomImageDataset('./dummy_data', transform=transform)

# Test
image, label = dataset[0]
print(f"\nImage shape: {image.shape}")
print(f"Label: {label}")
```

---

### Data Augmentation Techniques (2.5 minutes)

**"Increase dataset diversity to improve generalization"**

```python
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load a sample image (create one for demo)
img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

# Define various augmentation techniques
augmentation_transforms = {
    'Original': transforms.Compose([
        transforms.ToTensor()
    ]),

    'RandomHorizontalFlip': transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),  # p=1.0 for demo
        transforms.ToTensor()
    ]),

    'RandomRotation': transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor()
    ]),

    'RandomCrop': transforms.Compose([
        transforms.RandomCrop(size=200),
        transforms.Resize(256),
        transforms.ToTensor()
    ]),

    'ColorJitter': transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5,
                               saturation=0.5, hue=0.2),
        transforms.ToTensor()
    ]),

    'RandomAffine': transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                scale=(0.9, 1.1)),
        transforms.ToTensor()
    ]),

    'RandomPerspective': transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.5, p=1.0),
        transforms.ToTensor()
    ]),

    'GaussianBlur': transforms.Compose([
        transforms.GaussianBlur(kernel_size=5),
        transforms.ToTensor()
    ]),
}

# Visualize augmentations
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, (name, transform) in enumerate(augmentation_transforms.items()):
    augmented = transform(img)
    axes[idx].imshow(augmented.permute(1, 2, 0))
    axes[idx].set_title(name)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

print("\nAugmentation Guidelines:")
print("- Use for training data only (not validation/test)")
print("- Apply randomly with probability < 1.0")
print("- Don't over-augment (preserve data integrity)")
print("- Test augmentations visually first")
```

---

### Complete Training Pipeline with Augmentation (2 minutes)

**"Applying augmentation in practice"**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Different transforms for training and validation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),              # Random crop for variety
    transforms.RandomHorizontalFlip(p=0.5),  # 50% chance flip
    transforms.RandomRotation(degrees=15),    # Rotate up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# NO augmentation for validation/test
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # Center crop only
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load datasets with different transforms
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform  # Apply augmentation
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=val_transform  # No augmentation
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Visualize augmented vs original
fig, axes = plt.subplots(2, 8, figsize=(16, 4))

# Get one sample
original_img, label = train_dataset.dataset[0]  # Access original

for i in range(8):
    # Original (with augmentation)
    img, _ = train_dataset[0]  # Different augmentation each call
    axes[0, i].imshow(img.permute(1, 2, 0) * 0.229 + 0.485)  # Denormalize
    axes[0, i].axis('off')

    # Validation (no augmentation)
    img, _ = test_dataset[0]
    axes[1, i].imshow(img.permute(1, 2, 0) * 0.229 + 0.485)
    axes[1, i].axis('off')

axes[0, 0].set_title('Augmented', fontsize=12)
axes[1, 0].set_title('Original', fontsize=12)
plt.tight_layout()
plt.show()
```

---

### Handling Imbalanced Datasets (1.5 minutes)

**"When some classes have far fewer samples"**

```python
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Simulated imbalanced dataset
class ImbalancedDataset(torch.utils.data.Dataset):
    def __init__(self):
        # Class 0: 1000 samples, Class 1: 100 samples
        self.data = torch.randn(1100, 10)
        self.labels = torch.cat([
            torch.zeros(1000, dtype=torch.long),  # 1000 class 0
            torch.ones(100, dtype=torch.long)     # 100 class 1
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = ImbalancedDataset()

# Count samples per class
class_counts = torch.bincount(dataset.labels)
print(f"Class distribution: {class_counts}")

# Method 1: Weighted Random Sampler
# Give minority class higher sampling probability
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[dataset.labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)

# DataLoader with sampler (don't use shuffle=True with sampler)
balanced_loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# Verify balance
batch_labels = []
for _, labels in balanced_loader:
    batch_labels.extend(labels.tolist())

balanced_counts = torch.bincount(torch.tensor(batch_labels))
print(f"Balanced distribution (with sampler): {balanced_counts}")

# Method 2: Weighted Loss Function
# Give minority class higher weight in loss
pos_weight = class_counts[0] / class_counts[1]  # Ratio of majority/minority
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

print(f"\nPositive class weight: {pos_weight:.2f}")

# Method 3: Oversample minority class / Undersample majority
from torch.utils.data import Subset

# Find indices of each class
class_0_indices = (dataset.labels == 0).nonzero(as_tuple=True)[0]
class_1_indices = (dataset.labels == 1).nonzero(as_tuple=True)[0]

# Undersample majority to match minority
undersampled_indices = torch.cat([
    class_0_indices[torch.randperm(len(class_0_indices))[:len(class_1_indices)]],
    class_1_indices
])

balanced_dataset = Subset(dataset, undersampled_indices)
print(f"\nUndersampled dataset size: {len(balanced_dataset)}")
```

---

### Advanced: Custom Collate Function (1 minute)

**"Handle variable-sized inputs in batches"**

```python
import torch
from torch.utils.data import DataLoader

# Dataset with variable-length sequences
class VariableLengthDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.sequences = [
            torch.randn(10, 5),  # Length 10
            torch.randn(15, 5),  # Length 15
            torch.randn(8, 5),   # Length 8
            torch.randn(20, 5),  # Length 20
        ]
        self.labels = torch.tensor([0, 1, 0, 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Custom collate function
def collate_fn(batch):
    """
    Pad sequences to same length in batch
    """
    sequences, labels = zip(*batch)

    # Find max length in batch
    lengths = torch.tensor([len(seq) for seq in sequences])
    max_length = lengths.max()

    # Pad sequences
    padded_sequences = torch.zeros(len(sequences), max_length, sequences[0].size(1))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq

    labels = torch.stack(labels)

    return padded_sequences, labels, lengths

dataset = VariableLengthDataset()
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

for padded_seqs, labels, lengths in loader:
    print(f"Padded sequences shape: {padded_seqs.shape}")
    print(f"Original lengths: {lengths}")
    print(f"Labels: {labels}\n")
```

---

### Efficient Data Loading Best Practices (1 minute)

**"Speed up training with optimized data loading"**

```python
import torch
from torch.utils.data import DataLoader
import time

# Best practices for DataLoader
loader_configs = {
    'Slow (single worker)': {
        'num_workers': 0,
        'pin_memory': False,
        'persistent_workers': False
    },
    'Fast (optimized)': {
        'num_workers': 4,          # Parallel loading
        'pin_memory': True,         # Faster GPU transfer
        'persistent_workers': True, # Reuse workers
        'prefetch_factor': 2        # Prefetch batches
    }
}

dataset = datasets.CIFAR10('./data', train=True, download=True,
                           transform=transforms.ToTensor())

for name, config in loader_configs.items():
    loader = DataLoader(dataset, batch_size=128, **config)

    start_time = time.time()
    for i, (images, labels) in enumerate(loader):
        if i == 50:  # Test first 50 batches
            break
    elapsed = time.time() - start_time

    print(f"{name}: {elapsed:.2f} seconds")

print("\nBest Practices:")
print("1. num_workers=4-8: Parallel data loading")
print("2. pin_memory=True: Faster GPU transfer")
print("3. persistent_workers=True: Reuse worker processes")
print("4. prefetch_factor=2: Preload next batches")
print("5. Use SSD storage for datasets")
print("6. Preprocess data offline when possible")
```

---

### Complete Example: CIFAR-10 with Augmentation (1.5 minutes)

**"Putting it all together"**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Configuration
config = {
    'batch_size': 128,
    'num_workers': 4,
    'learning_rate': 0.001,
    'epochs': 20
}

# Augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])

# Load data
train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                  transform=train_transform)
test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                                 transform=test_transform)

# Split train into train/val
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                          shuffle=True, num_workers=config['num_workers'],
                          pin_memory=True, persistent_workers=True)

val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                        shuffle=False, num_workers=config['num_workers'],
                        pin_memory=True, persistent_workers=True)

test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                         shuffle=False, num_workers=config['num_workers'],
                         pin_memory=True)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Ready for training!")
```

---

## Key Takeaways

1. **Dataset Class**:
   - Inherit from `torch.utils.data.Dataset`
   - Implement `__init__`, `__len__`, `__getitem__`
   - Return (data, label) from `__getitem__`

2. **Data Augmentation** (Training only):
   - `RandomHorizontalFlip`, `RandomRotation`, `RandomCrop`
   - `ColorJitter` for color variations
   - `RandomAffine`, `RandomPerspective` for geometric transforms
   - Never augment validation/test data!

3. **DataLoader Optimization**:
   - `num_workers=4-8`: Parallel loading
   - `pin_memory=True`: Faster GPU transfer
   - `persistent_workers=True`: Reuse workers
   - `prefetch_factor=2`: Preload batches

4. **Imbalanced Data**:
   - Use `WeightedRandomSampler`
   - Or weighted loss function
   - Or oversample/undersample

5. **Best Practices**:
   - Different transforms for train/val/test
   - Verify augmentations visually
   - Use efficient data loading
   - Normalize with dataset statistics

---

## Today's Practice Exercise

**Create a custom dataset with augmentation**

```python
# YOUR TASK:
# 1. Create a custom dataset class for a folder of images
# 2. Implement strong augmentation for training
# 3. Create train/val/test splits
# 4. Use WeightedRandomSampler if data is imbalanced
# 5. Optimize DataLoader settings
# 6. Visualize augmented samples
# 7. Train a model and compare with/without augmentation

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class YourCustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # YOUR CODE HERE
        pass

    def __len__(self):
        # YOUR CODE HERE
        pass

    def __getitem__(self, idx):
        # YOUR CODE HERE
        pass

# Define augmentation pipeline
train_transform = transforms.Compose([
    # YOUR AUGMENTATIONS HERE
])

# Create dataset and loader
dataset = YourCustomDataset('./your_data', transform=train_transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Train and compare results!
```

---

## Tomorrow's Preview

**Day 8: Introduction to Convolutional Neural Networks (CNNs)**

- Understanding convolution operations
- Pooling layers (MaxPool, AvgPool)
- CNN architectures for image classification
- Building your first CNN from scratch
- Visualizing learned features
- Transfer learning basics

---

**"You now master data pipelines! Tomorrow we'll dive into CNNs for computer vision. 🚀"**
