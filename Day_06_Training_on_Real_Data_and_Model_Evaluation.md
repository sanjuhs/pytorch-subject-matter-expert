# Day 6: Training on Real Data and Model Evaluation

**Duration**: 10 minutes
**Platform**: Google Colab

---

## Video Script

### Introduction (30 seconds)

**"Welcome to Day 6 - where we train on real datasets and learn to evaluate models properly!"**

So far we've used synthetic data. Today we'll work with real datasets and learn:
- Loading real datasets (MNIST, CIFAR-10, FashionMNIST)
- Train/validation/test splits
- Evaluation metrics (accuracy, precision, recall, F1)
- Model checkpointing
- Visualizing training progress
- Avoiding overfitting

By the end, you'll know how to train and evaluate models like a professional.

---

### Loading Real Datasets with torchvision (1.5 minutes)

**"torchvision provides popular computer vision datasets"**

```python
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transform: Convert images to tensors and normalize
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor (0-1 range)
    transforms.Normalize((0.5,), (0.5,))  # Normalize to (-1, 1)
])

# Download and load MNIST
train_dataset = datasets.MNIST(
    root='./data',           # Where to save data
    train=True,              # Training set
    download=True,           # Download if not present
    transform=transform      # Apply transformations
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,             # Test set
    download=True,
    transform=transform
)

print(f"Training samples: {len(train_dataset)}")    # 60,000
print(f"Test samples: {len(test_dataset)}")         # 10,000

# Inspect a sample
image, label = train_dataset[0]
print(f"\nImage shape: {image.shape}")  # (1, 28, 28) - 1 channel, 28x28 pixels
print(f"Label: {label}")
print(f"Pixel value range: [{image.min():.2f}, {image.max():.2f}]")

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
for i in range(5):
    image, label = train_dataset[i]
    plt.subplot(1, 5, i+1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Other popular datasets:
# - FashionMNIST: Clothing items
# - CIFAR10: 32x32 color images, 10 classes
# - CIFAR100: 32x32 color images, 100 classes
# - ImageNet (requires download): 1000 classes
```

---

### Creating Train/Validation/Test Splits (1.5 minutes)

**"Never evaluate on training data! Use proper splits"**

```python
import torch
from torch.utils.data import DataLoader, random_split

# Load full training set
train_full = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Split training into train + validation
train_size = int(0.8 * len(train_full))  # 80% for training
val_size = len(train_full) - train_size   # 20% for validation

train_dataset, val_dataset = random_split(
    train_full,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

print(f"Training samples:   {len(train_dataset)}")    # 48,000
print(f"Validation samples: {len(val_dataset)}")      # 12,000
print(f"Test samples:       {len(test_dataset)}")     # 10,000

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,      # Shuffle training data
    num_workers=2      # Parallel data loading
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,     # Don't shuffle validation
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2
)

print(f"\nTraining batches:   {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches:       {len(test_loader)}")

# Why three splits?
# TRAIN: Update model weights
# VALIDATION: Tune hyperparameters, early stopping
# TEST: Final evaluation (touch only once!)
```

---

### Building a Complete Training Pipeline (2 minutes)

**"A professional training loop with validation"**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Progress bars

# Define model
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MNISTNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()  # Set to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# Training loop
epochs = 10
train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")

print("\nTraining complete!")
```

---

### Visualizing Training Progress (1 minute)

**"Plots reveal overfitting, underfitting, and convergence"**

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot loss
ax1.plot(train_losses, label='Train Loss', marker='o')
ax1.plot(val_losses, label='Val Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Plot accuracy
ax2.plot(train_accs, label='Train Acc', marker='o')
ax2.plot(val_accs, label='Val Acc', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# What to look for:
# ✅ Both losses decreasing: Good!
# ❌ Train loss << Val loss: Overfitting (model memorizing training data)
# ❌ Both losses high: Underfitting (model too simple)
# ❌ Val loss increasing: Overfitting (stop training!)
```

---

### Evaluation Metrics Beyond Accuracy (2 minutes)

**"Accuracy isn't always enough - use precision, recall, F1"**

```python
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get predictions on test set
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        all_predictions.extend(predictions.numpy())
        all_labels.extend(labels.numpy())

# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(all_labels, all_predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# Classification report (precision, recall, F1)
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(10)]))

# Confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Understanding metrics:
# ACCURACY: (TP + TN) / Total - Overall correctness
# PRECISION: TP / (TP + FP) - Of predicted positives, how many are correct?
# RECALL: TP / (TP + FN) - Of actual positives, how many did we find?
# F1-SCORE: 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean

# Example: Predicting digit "3"
# - High precision: When model predicts "3", it's usually right
# - High recall: Model finds most of the "3"s in dataset
# - High F1: Good balance of both
```

---

### Model Checkpointing and Saving (1.5 minutes)

**"Save your models to avoid retraining"**

```python
import torch
import os

# Save model checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_accuracy': val_acc,
}

# Save to file
torch.save(checkpoint, 'mnist_checkpoint.pth')
print("Checkpoint saved!")

# Load checkpoint
checkpoint = torch.load('mnist_checkpoint.pth')

# Restore model
model = MNISTNet().to(device)
model.load_state_dict(checkpoint['model_state_dict'])

# Restore optimizer (if continuing training)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Val accuracy: {checkpoint['val_accuracy']:.2f}%")

# Save only the best model
best_val_acc = 0
for epoch in range(epochs):
    # ... training code ...
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # Save if best so far
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_accuracy': val_acc,
        }, 'best_model.pth')
        print(f"New best model saved! Val Acc: {val_acc:.2f}%")
```

---

### Early Stopping (1 minute)

**"Stop training when validation loss stops improving"**

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        """
        patience: How many epochs to wait after last improvement
        min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Use in training loop
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(100):  # Max 100 epochs
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")

    # Check early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Why early stopping?
# - Prevents overfitting
# - Saves training time
# - Automatically finds optimal number of epochs
```

---

### Complete Production-Ready Training Script (1.5 minutes)

**"Putting it all together"**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hyperparameters
CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'epochs': 50,
    'patience': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_full = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size
train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

# Model
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

model = MNISTNet().to(CONFIG['device'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                        weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=0.5, patience=3)

# Training
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0
early_stopping = EarlyStopping(patience=CONFIG['patience'])

for epoch in range(CONFIG['epochs']):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                         optimizer, CONFIG['device'])

    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])

    # Update scheduler
    scheduler.step(val_loss)

    # Record history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
    print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
    print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ✓ Best model saved!")

    # Early stopping
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Load best model and evaluate on test set
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc = validate(model, test_loader, criterion, CONFIG['device'])
print(f"\nTest Accuracy: {test_acc:.2f}%")
```

---

## Key Takeaways

1. **Dataset Splits**:
   - **Train**: Update model weights (80%)
   - **Validation**: Tune hyperparameters, early stopping (10-20%)
   - **Test**: Final evaluation - use only once! (10-20%)

2. **Evaluation Metrics**:
   - **Accuracy**: Overall correctness
   - **Precision**: Quality of positive predictions
   - **Recall**: Coverage of actual positives
   - **F1-Score**: Balance of precision and recall

3. **Training Best Practices**:
   - Always use validation set
   - Monitor both train and val metrics
   - Save checkpoints regularly
   - Use early stopping
   - Plot training curves
   - Test only once at the end

4. **Overfitting Detection**:
   - Train loss << Val loss → Overfitting
   - Use dropout, weight decay, early stopping

5. **Model Persistence**:
   - Save: `torch.save(model.state_dict(), 'model.pth')`
   - Load: `model.load_state_dict(torch.load('model.pth'))`

---

## Today's Practice Exercise

**Train a FashionMNIST classifier with proper evaluation**

```python
# YOUR TASK:
# 1. Load FashionMNIST dataset
# 2. Create train/val/test splits
# 3. Build a neural network
# 4. Train with early stopping
# 5. Plot training curves
# 6. Generate classification report
# 7. Create confusion matrix
# 8. Save the best model
# 9. Load and test the saved model

# Classes: T-shirt, Trouser, Pullover, Dress, Coat,
#          Sandal, Shirt, Sneaker, Bag, Ankle boot

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# YOUR CODE HERE
```

---

## Tomorrow's Preview

**Day 7: Data Loading, Preprocessing, and Augmentation**

- Custom datasets with `Dataset` class
- Data augmentation (rotation, flipping, cropping)
- Handling images of different sizes
- Working with custom image folders
- Efficient data pipelines
- Dealing with imbalanced datasets

---

**"You now know how to train and evaluate models professionally! Tomorrow we'll learn advanced data loading techniques. 🚀"**
