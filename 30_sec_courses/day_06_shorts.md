# Day 06 — 30-Second Shorts (Real Data + Evaluation)

These are **script-ready** shorts. Each short includes:
- **Title**
- **Hook** (1 sentence)
- **Talk track** (what you say, ~20 seconds)
- **Code** (what you show)
- **Common mistake** (quick pitfall)

---

## Short 01 — Day 6: Real Data + Professional Evaluation

**Hook**
Training is easy — evaluating correctly is what makes you a pro.

**Talk track**
- “Today we move from synthetic data to real datasets.”
- “We’ll learn train/val/test splits, evaluation metrics, checkpointing, and plots.”

**Code**
```python
# Today: torchvision datasets + splits + loaders + eval + saving
```

**Common mistake**
- Reporting training accuracy as “model performance”.

---

## Short 02 — torchvision Datasets: MNIST in 5 Lines

**Hook**
torchvision gives you real datasets with one import.

**Talk track**
- “Use `datasets.MNIST` with `download=True`.”
- “Apply transforms so images become normalized tensors.”

**Code**
```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_full = datasets.MNIST("./data", train=True, download=True, transform=transform)
```

**Common mistake**
- Forgetting transforms → model sees raw PIL images or inconsistent scaling.

---

## Short 03 — Transforms: ToTensor + Normalize (What It Means)

**Hook**
Normalize your inputs or training becomes harder.

**Talk track**
- “`ToTensor()` converts to float tensor in [0,1].”
- “Normalize shifts/scales to something like [-1,1].”

**Code**
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

**Common mistake**
- Using the wrong mean/std shape for multi-channel images (CIFAR-10 needs 3 values).

---

## Short 04 — Inspect One Sample (Shape + Label)

**Hook**
Before training, inspect the dataset.

**Talk track**
- “Print image shape, label, and value range.”
- “This catches transform mistakes immediately.”

**Code**
```python
image, label = train_full[0]
print(image.shape, label, float(image.min()), float(image.max()))
```

**Common mistake**
- Forgetting `.squeeze()` when visualizing single-channel images.

---

## Short 05 — Visualize a Few Samples (Quick Sanity Check)

**Hook**
If the data looks wrong, the model will learn the wrong thing.

**Talk track**
- “Plot 5 samples quickly.”
- “Confirm labels match the image.”

**Code**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 3))
for i in range(5):
    img, y = train_full[i]
    plt.subplot(1, 5, i+1)
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(y)
    plt.axis("off")
plt.tight_layout()
plt.show()
```

**Common mistake**
- Visualizing normalized images without un-normalizing (they can look “washed out”; it’s fine).

---

## Short 06 — Train/Val/Test Split (Why 3 Sets?)

**Hook**
Touch the test set once — at the end.

**Talk track**
- “Train: update weights.”
- “Validation: choose hyperparameters / early stopping.”
- “Test: final report.”

**Code**
```python
# TRAIN: learn
# VAL: tune / stop
# TEST: final score (only once)
```

**Common mistake**
- Using the test set repeatedly to tune hyperparameters (data leakage).

---

## Short 07 — random_split() With a Seed (Reproducible Splits)

**Hook**
No seed = your validation set changes every run.

**Talk track**
- “Use `random_split` and set a manual seed.”
- “So your comparisons are fair.”

**Code**
```python
import torch
from torch.utils.data import random_split

train_size = int(0.8 * len(train_full))
val_size = len(train_full) - train_size
train_ds, val_ds = random_split(
    train_full, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
```

**Common mistake**
- Comparing models when the validation split changed (you’re comparing apples to oranges).

---

## Short 08 — DataLoader: Batching + Shuffling

**Hook**
DataLoader is how you feed data efficiently.

**Talk track**
- “Train loader should shuffle.”
- “Val/test loaders should not shuffle.”

**Code**
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
```

**Common mistake**
- Shuffling validation/test and then getting hard-to-debug evaluation behavior.

---

## Short 09 — Device: CPU vs GPU (Always Explicit)

**Hook**
Pick a device once, move everything consistently.

**Talk track**
- “Choose `cuda` if available.”
- “Move model and tensors to the same device.”

**Code**
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```

**Common mistake**
- CPU tensor + GPU model = runtime error (device mismatch).

---

## Short 10 — A Real MNIST Model (MLP Starter)

**Hook**
This is a clean baseline you can beat later.

**Talk track**
- “Flatten → Linear → ReLU → Dropout → Linear → logits.”
- “Keep it simple for your first real dataset.”

**Code**
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10),
)
```

**Common mistake**
- Adding softmax at the end and then using CrossEntropyLoss (don’t; pass logits).

---

## Short 11 — Loss + Optimizer (Standard Pair for MNIST)

**Hook**
CrossEntropy + AdamW is a great default.

**Talk track**
- “Classification uses CrossEntropyLoss.”
- “AdamW is a strong default optimizer.”

**Code**
```python
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

**Common mistake**
- Using BCE for 10-class MNIST (wrong loss for the task).

---

## Short 12 — The Core Training Step (5 Lines)

**Hook**
Memorize this pattern — it’s everywhere.

**Talk track**
- “Forward → loss → zero_grad → backward → step.”

**Code**
```python
# outputs = model(images)
# loss = criterion(outputs, labels)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
```

**Common mistake**
- Forgetting `optimizer.zero_grad()` → gradients accumulate across batches.

---

## Short 13 — model.train(): Turn On Training Behaviors

**Hook**
Dropout and BatchNorm behave differently in train mode.

**Talk track**
- “During training, call `model.train()`.”
- “This enables dropout and updates batchnorm stats.”

**Code**
```python
model.train()
```

**Common mistake**
- Training while in eval mode → dropout disabled and batchnorm frozen.

---

## Short 14 — model.eval() + torch.no_grad(): Correct Validation

**Hook**
Validation should not compute gradients.

**Talk track**
- “Set eval mode.”
- “Wrap in `no_grad()` for speed and memory savings.”

**Code**
```python
model.eval()
with torch.no_grad():
    outputs = model(images)
```

**Common mistake**
- Validating in train mode (dropout noise makes metrics unstable).

---

## Short 15 — Compute Accuracy (Fast + Simple)

**Hook**
Accuracy is the first metric you should track.

**Talk track**
- “Get predicted class with `argmax`.”
- “Compare to labels and average.”

**Code**
```python
pred = outputs.argmax(dim=1)
acc = (pred == labels).float().mean().item()
print(acc)
```

**Common mistake**
- Forgetting to move tensors to CPU before converting to NumPy for some metrics/plots.

---

## Short 16 — tqdm Progress Bars (Instant Training UX Upgrade)

**Hook**
Seeing progress makes debugging training 10x easier.

**Talk track**
- “Wrap your loader with tqdm.”
- “You get ETA and speed instantly.”

**Code**
```python
from tqdm import tqdm

for images, labels in tqdm(train_loader, desc="Training"):
    pass
```

**Common mistake**
- Putting heavy printing inside the tqdm loop (slows everything).

---

## Short 17 — Track Train Loss vs Val Loss (Overfitting Detector)

**Hook**
Curves tell the truth even when metrics lie.

**Talk track**
- “Track train loss and val loss each epoch.”
- “If train keeps improving but val gets worse: overfitting.”

**Code**
```python
train_losses, val_losses = [], []
# append train_loss and val_loss each epoch
```

**Common mistake**
- Only tracking training loss and thinking “lower loss = better model”.

---

## Short 18 — Plot Loss Curves (The Best Debug Tool)

**Hook**
One plot can save you hours.

**Talk track**
- “Plot train vs val loss.”
- “Look for divergence (overfitting).”

**Code**
```python
import matplotlib.pyplot as plt

plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.grid(True)
plt.show()
```

**Common mistake**
- Not saving history, so you can’t compare runs.

---

## Short 19 — Plot Accuracy Curves Too

**Hook**
Loss might drop while accuracy stays flat — check both.

**Talk track**
- “Track train accuracy and val accuracy.”
- “Watch for gaps between them.”

**Code**
```python
plt.plot(train_accs, label="train acc")
plt.plot(val_accs, label="val acc")
plt.legend()
plt.grid(True)
plt.show()
```

**Common mistake**
- Comparing accuracy between runs without consistent data split/seed.

---

## Short 20 — Accuracy Isn’t Enough (Precision/Recall/F1)

**Hook**
Accuracy can lie when classes are imbalanced.

**Talk track**
- “Precision: when you predict positive, how often you’re right.”
- “Recall: how many true positives you found.”
- “F1: balance between precision and recall.”

**Code**
```python
# Accuracy, Precision, Recall, F1
```

**Common mistake**
- Using accuracy only on imbalanced datasets and being misled.

---

## Short 21 — classification_report in 2 Lines

**Hook**
Get precision/recall/F1 per class instantly.

**Talk track**
- “Collect predictions on test set.”
- “Then call sklearn’s report.”

**Code**
```python
from sklearn.metrics import classification_report

print(classification_report(all_labels, all_predictions))
```

**Common mistake**
- Forgetting to `pip install scikit-learn` in some environments (Colab has it often, local may not).

---

## Short 22 — Confusion Matrix: Where Your Model Is Confused

**Hook**
This shows exactly which classes get mixed up.

**Talk track**
- “Confusion matrix rows are true labels, columns are predicted.”
- “It’s the fastest way to see failure modes.”

**Code**
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_predictions)
print(cm.shape)
```

**Common mistake**
- Interpreting axes backwards and drawing the wrong conclusion.

---

## Short 23 — Heatmap Confusion Matrix (Readable)

**Hook**
Turn a matrix into insight in one plot.

**Talk track**
- “Heatmap makes patterns obvious.”

**Code**
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
```

**Common mistake**
- Plotting with too many classes without adjusting figure size (unreadable).

---

## Short 24 — Save the Model: state_dict() (Best Practice)

**Hook**
Save weights, not the whole Python object.

**Talk track**
- “Use `model.state_dict()` for portability.”
- “Then load into the same model class later.”

**Code**
```python
import torch

torch.save(model.state_dict(), "best_model.pth")
```

**Common mistake**
- Saving the entire model object and then failing to load due to code changes.

---

## Short 25 — Load the Model: load_state_dict()

**Hook**
Loading is 2 lines when you save state_dict.

**Talk track**
- “Recreate the model architecture.”
- “Load weights and switch to eval mode.”

**Code**
```python
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
```

**Common mistake**
- Loading on GPU/CPU mismatch; use `torch.load(..., map_location=device)` when needed.

---

## Short 26 — Full Checkpoint: Model + Optimizer + Epoch

**Hook**
Checkpointing lets you resume training exactly.

**Talk track**
- “Save model weights, optimizer state, and metadata like epoch.”
- “This is what ‘resume training’ means.”

**Code**
```python
import torch

ckpt = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}
torch.save(ckpt, "checkpoint.pth")
```

**Common mistake**
- Only saving model weights, then resuming with a fresh optimizer (can change training behavior).

---

## Short 27 — “Save Best Model” Pattern

**Hook**
Always keep the best checkpoint, not just the last.

**Talk track**
- “Track best val accuracy.”
- “If current val acc is best, save the model.”

**Code**
```python
best_val_acc = 0.0
if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), "best_model.pth")
```

**Common mistake**
- Choosing best model based on training accuracy (overfits).

---

## Short 28 — Early Stopping: Stop When Val Loss Stops Improving

**Hook**
The best model often happens before the final epoch.

**Talk track**
- “Early stopping prevents overfitting and saves time.”
- “If validation loss doesn’t improve for `patience` epochs, stop.”

**Code**
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best is None:
            self.best = val_loss
        elif val_loss > self.best - self.min_delta:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True
        else:
            self.best = val_loss
            self.count = 0
```

**Common mistake**
- Early stopping on accuracy that bounces; val loss is often smoother.

---

## Short 29 — ReduceLROnPlateau: Auto-Lower LR When Stuck

**Hook**
If val loss plateaus, lower the learning rate.

**Talk track**
- “This scheduler watches validation loss.”
- “When it stops improving, LR drops automatically.”

**Code**
```python
import torch.optim as optim

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)
# call once per epoch:
# scheduler.step(val_loss)
```

**Common mistake**
- Forgetting `scheduler.step(val_loss)` (scheduler never updates).

---

## Short 30 — Overfitting Checklist (What To Do)

**Hook**
Train loss way lower than val loss? That’s overfitting.

**Talk track**
- “Use dropout, weight decay, data augmentation (later), early stopping.”
- “Or simplify the model.”

**Code**
```python
# Fix overfitting:
# - dropout
# - weight_decay
# - early stopping
# - smaller model
```

**Common mistake**
- Training longer hoping it will fix overfitting (it usually gets worse).

---

## Short 31 — Test Set Rule: Evaluate Once

**Hook**
The test set is sacred.

**Talk track**
- “Don’t tune hyperparameters on test.”
- “Use validation for decisions.”
- “Use test once for final report.”

**Code**
```python
# Tune on val. Report on test.
```

**Common mistake**
- Iterating on test results (you’re effectively training on the test set).

---

## Short 32 — Day 6 Summary: The Professional Workflow

**Hook**
You now know the end-to-end training pipeline.

**Talk track**
- “Load data → split → DataLoader.”
- “Train with `model.train()`.”
- “Validate with `model.eval()` + `no_grad()`.”
- “Plot curves, save best model, and test once.”

**Code**
```python
# Data -> Train -> Validate -> Plot -> Save best -> Test
```

**Common mistake**
- Not saving the best model and losing your best run.


