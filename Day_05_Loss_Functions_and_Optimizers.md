# Day 5: Loss Functions and Optimizers

**Duration**: 10 minutes
**Platform**: Google Colab

---

## Video Script

### Introduction (30 seconds)

**"Welcome to Day 5 - where we learn how neural networks actually learn!"**

We've built neural networks with `nn.Module`. Now we need to understand:
- **Loss functions**: How to measure how wrong our predictions are
- **Optimizers**: How to update weights to improve predictions
- **Learning rate**: The most important hyperparameter
- **Gradient descent variants**: SGD, Adam, RMSprop, and more

By the end, you'll understand how to train any neural network effectively.

---

### What is a Loss Function? (1.5 minutes)

**"Loss functions measure the difference between predictions and truth"**

```python
import torch
import torch.nn as nn

# Example: Predicting house prices
predictions = torch.tensor([250000., 180000., 320000.])  # Model predictions
actual = torch.tensor([240000., 200000., 310000.])       # True prices

# Method 1: Manual calculation
differences = predictions - actual
squared_errors = differences ** 2
mse_manual = squared_errors.mean()

print(f"Predictions: {predictions}")
print(f"Actual:      {actual}")
print(f"Differences: {differences}")
print(f"MSE (manual): {mse_manual.item():.2f}")

# Method 2: Using PyTorch
mse_loss = nn.MSELoss()
loss = mse_loss(predictions, actual)
print(f"MSE (PyTorch): {loss.item():.2f}")

# Why squared error?
# - Penalizes large errors more than small errors
# - Always positive
# - Differentiable (smooth gradients)

# Alternative: Mean Absolute Error (L1 Loss)
mae_loss = nn.L1Loss()
mae = mae_loss(predictions, actual)
print(f"\nMAE: {mae.item():.2f}")

# L1 vs L2:
# L1 (MAE): Less sensitive to outliers, sparser gradients
# L2 (MSE): More sensitive to outliers, smoother gradients
```

---

### Loss Functions for Classification (2 minutes)

#### 1. Binary Cross-Entropy (BCE)

**"For binary classification (yes/no, cat/dog, spam/not spam)"**

```python
import torch
import torch.nn as nn

# Example: Spam detection (1 = spam, 0 = not spam)
predictions = torch.tensor([0.9, 0.2, 0.7, 0.1])  # Model probabilities
targets = torch.tensor([1.0, 0.0, 1.0, 0.0])      # True labels

# Binary Cross-Entropy Loss
bce_loss = nn.BCELoss()
loss = bce_loss(predictions, targets)

print(f"Predictions: {predictions}")
print(f"Targets:     {targets}")
print(f"BCE Loss: {loss.item():.4f}")

# Manual calculation:
# BCE = -[y*log(p) + (1-y)*log(1-p)]
manual_bce = -(targets * torch.log(predictions) +
               (1 - targets) * torch.log(1 - predictions)).mean()
print(f"Manual BCE: {manual_bce.item():.4f}")

# IMPORTANT: Predictions must be probabilities (0 to 1)
# Use sigmoid activation before BCE:
logits = torch.tensor([2.0, -1.5, 1.0, -2.0])  # Raw outputs
probs = torch.sigmoid(logits)
print(f"\nLogits: {logits}")
print(f"Probabilities: {probs}")

# Or use BCEWithLogitsLoss (more numerically stable)
bce_with_logits = nn.BCEWithLogitsLoss()
loss = bce_with_logits(logits, targets)
print(f"BCE with logits: {loss.item():.4f}")
```

#### 2. Cross-Entropy Loss (Multi-class)

**"For multi-class classification (digits, objects, etc.)"**

```python
import torch
import torch.nn as nn

# Example: Digit classification (0-9)
# Batch of 3 samples, 10 classes
logits = torch.tensor([
    [2.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Predicts class 0
    [0.0, 0.0, 0.0, 3.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],  # Predicts class 3
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0],  # Predicts class 7
])

targets = torch.tensor([0, 3, 7])  # True class indices

# CrossEntropyLoss
ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(logits, targets)

print(f"Logits shape: {logits.shape}")
print(f"Targets: {targets}")
print(f"Cross-Entropy Loss: {loss.item():.4f}")

# What CrossEntropyLoss does:
# 1. Applies softmax to logits → probabilities
# 2. Takes negative log of probability for true class
# 3. Averages across batch

# Manual calculation:
probs = torch.softmax(logits, dim=1)
print(f"\nProbabilities:\n{probs}")

# For first sample: true class is 0
print(f"\nSample 1 probability for class 0: {probs[0, 0]:.4f}")
print(f"Sample 1 loss: -log({probs[0, 0]:.4f}) = {-torch.log(probs[0, 0]):.4f}")

# IMPORTANT: No softmax before CrossEntropyLoss!
# It's built-in for numerical stability

# Wrong way:
# probs = torch.softmax(logits, dim=1)
# loss = ce_loss(probs, targets)  # ❌ Double softmax!

# Right way:
# loss = ce_loss(logits, targets)  # ✅ Raw logits
```

---

### Understanding Optimizers (2 minutes)

**"Optimizers update model parameters to minimize loss"**

#### Gradient Descent Basics

```python
import torch

# Simple optimization problem: minimize f(x) = (x - 3)^2
x = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.1
steps = 20

print("Minimizing f(x) = (x - 3)^2 using gradient descent")
print(f"Starting x: {x.item():.4f}\n")

for step in range(steps):
    # Compute loss
    loss = (x - 3) ** 2

    # Backward pass
    loss.backward()

    # Manual gradient descent
    with torch.no_grad():
        x -= learning_rate * x.grad

    # Zero gradient
    x.grad.zero_()

    if step % 5 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}, loss = {loss.item():.4f}")

print(f"\nFinal x: {x.item():.4f} (target: 3.0)")
```

#### Using PyTorch Optimizers

```python
import torch
import torch.optim as optim

# Same problem, using PyTorch optimizer
x = torch.tensor(0.0, requires_grad=True)

# Create optimizer (manages parameter updates)
optimizer = optim.SGD([x], lr=0.1)

print("\nUsing torch.optim.SGD:")

for step in range(20):
    # Compute loss
    loss = (x - 3) ** 2

    # Backward pass
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients

    # Update parameters
    optimizer.step()       # Performs x = x - lr * grad

    if step % 5 == 0:
        print(f"Step {step:2d}: x = {x.item():.4f}, loss = {loss.item():.4f}")
```

---

### Popular Optimizers Compared (2.5 minutes)

#### 1. SGD (Stochastic Gradient Descent)

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# What does momentum do?
# - Accumulates past gradients
# - Accelerates in consistent directions
# - Dampens oscillations

# Formula: v = momentum * v + gradient
#          param = param - lr * v

print("SGD with momentum:")
print(f"Learning rate: 0.01")
print(f"Momentum: 0.9")
```

#### 2. Adam (Adaptive Moment Estimation)

```python
# Adam optimizer (most popular)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# What makes Adam special?
# - Adaptive learning rates for each parameter
# - Combines momentum and RMSprop
# - Works well with minimal tuning

print("\nAdam optimizer:")
print(f"Learning rate: 0.001")
print("- Adapts learning rate per parameter")
print("- Good default choice")
```

#### 3. RMSprop

```python
# RMSprop optimizer
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# What does RMSprop do?
# - Divides learning rate by running average of gradient magnitudes
# - Good for recurrent networks
# - Handles sparse gradients well

print("\nRMSprop:")
print(f"Learning rate: 0.01")
print("- Good for RNNs and non-stationary problems")
```

#### 4. AdamW (Adam with Weight Decay)

```python
# AdamW optimizer (modern best practice)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# What is weight decay?
# - Adds L2 regularization: loss = loss + 0.01 * ||weights||^2
# - Prevents weights from getting too large
# - Reduces overfitting

print("\nAdamW:")
print(f"Learning rate: 0.001")
print(f"Weight decay: 0.01")
print("- Adam with proper weight decay")
print("- Current best practice for transformers")
```

---

### Complete Training Loop with Different Optimizers (2 minutes)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Generate synthetic data
torch.manual_seed(42)
X = torch.randn(1000, 10)
y = (X.sum(dim=1, keepdim=True) > 0).float()  # Binary classification

# Model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1),
    nn.Sigmoid()
)

# Loss function
criterion = nn.BCELoss()

# Compare optimizers
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01),
    'SGD+Momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'Adam': optim.Adam(model.parameters(), lr=0.01),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.01),
}

# Train with each optimizer
results = {}

for name, optimizer in optimizers.items():
    # Reset model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
        nn.Sigmoid()
    )

    # Re-initialize optimizer with new model
    if name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif name == 'SGD+Momentum':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=0.01)

    losses = []

    # Training
    for epoch in range(50):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    results[name] = losses
    print(f"{name:15s} - Final loss: {losses[-1]:.4f}")

# Plot comparison
plt.figure(figsize=(10, 6))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

---

### Learning Rate: The Most Important Hyperparameter (1.5 minutes)

**"Getting the learning rate right is crucial"**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generate data
torch.manual_seed(42)
X = torch.randn(100, 5)
y = torch.randn(100, 1)

model = nn.Linear(5, 1)
criterion = nn.MSELoss()

# Compare different learning rates
learning_rates = [0.001, 0.01, 0.1, 1.0]

for lr in learning_rates:
    # Reset model
    model = nn.Linear(5, 1)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print(f"\nLearning Rate: {lr}")

    for epoch in range(10):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        if epoch % 3 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

# Observations:
# lr = 0.001: Very slow, may not converge
# lr = 0.01:  Good balance
# lr = 0.1:   Fast, might be unstable
# lr = 1.0:   Too large, diverges (loss increases!)
```

#### Learning Rate Scheduling

```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

model = nn.Linear(5, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 1. StepLR: Reduce LR every N epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# 2. ExponentialLR: Multiply LR by gamma each epoch
scheduler = ExponentialLR(optimizer, gamma=0.95)

# 3. CosineAnnealingLR: Cosine decay (popular for transformers)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Usage in training loop:
for epoch in range(50):
    # Training code...
    optimizer.zero_grad()
    # ... forward, backward, optimizer.step()

    # Update learning rate
    scheduler.step()

    # Print current LR
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: LR = {current_lr:.6f}")
```

---

### Gradient Clipping (1 minute)

**"Prevent exploding gradients"**

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with gradient clipping
X = torch.randn(32, 10)
y = torch.randn(32, 1)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()

    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Check gradient norms before and after clipping
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Grad norm = {total_norm:.4f}")

# When to use gradient clipping?
# - RNNs (prone to exploding gradients)
# - Very deep networks
# - Unstable training (loss spikes)
```

---

### Practical Training Loop Template (1 minute)

**"A production-ready training loop"**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters
learning_rate = 0.001
batch_size = 64
epochs = 50
weight_decay = 0.01

# Data
X_train = torch.randn(1000, 20)
y_train = torch.randint(0, 10, (1000,))
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = nn.Sequential(
    nn.Linear(20, 100),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(100, 10)
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Training loop
for epoch in range(epochs):
    model.train()  # Set to training mode
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Track metrics
        epoch_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == batch_y).sum().item()
        total += batch_y.size(0)

    # Update learning rate
    scheduler.step()

    # Print progress
    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100 * correct / total
    lr = optimizer.param_groups[0]['lr']

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, "
              f"Acc = {accuracy:.2f}%, LR = {lr:.6f}")

print("Training complete!")
```

---

## Key Takeaways

1. **Loss Functions**:
   - **MSELoss**: Regression tasks
   - **BCELoss**: Binary classification (with sigmoid)
   - **CrossEntropyLoss**: Multi-class classification (with raw logits)

2. **Optimizers**:
   - **SGD**: Simple, needs tuning
   - **SGD + Momentum**: Better than vanilla SGD
   - **Adam**: Good default choice
   - **AdamW**: Modern best practice (Adam + weight decay)

3. **Learning Rate**:
   - Most important hyperparameter
   - Too small: slow convergence
   - Too large: divergence
   - Use schedulers to adjust during training

4. **Training Loop**:
   ```python
   optimizer.zero_grad()  # 1. Clear gradients
   loss = criterion(...)   # 2. Compute loss
   loss.backward()         # 3. Compute gradients
   optimizer.step()        # 4. Update weights
   ```

5. **Advanced Techniques**:
   - Learning rate scheduling
   - Gradient clipping
   - Weight decay (regularization)

---

## Today's Practice Exercise

**Train a network with different configurations and compare results**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)

# EXPERIMENT 1: Compare optimizers
optimizers_to_test = [
    ('SGD', optim.SGD(model.parameters(), lr=0.01)),
    ('Adam', optim.Adam(model.parameters(), lr=0.001)),
    ('AdamW', optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01))
]

# EXPERIMENT 2: Compare learning rates
learning_rates = [0.0001, 0.001, 0.01, 0.1]

# EXPERIMENT 3: Compare loss functions for same task
# (Try MSE vs CrossEntropy for classification - see what happens!)

# YOUR TASK:
# 1. Train with different optimizers and compare final accuracy
# 2. Try different learning rates with Adam
# 3. Add learning rate scheduling
# 4. Visualize training loss curves
# 5. Find the best configuration!
```

---

## Tomorrow's Preview

**Day 6: Training on Real Data and Model Evaluation**

- Loading and preprocessing datasets
- Data augmentation techniques
- Train/validation/test splits
- Evaluation metrics (accuracy, precision, recall, F1)
- Model checkpointing and saving
- Visualizing training progress

---

**"You now understand how neural networks learn! Tomorrow we'll train on real data and learn how to evaluate model performance. 🚀"**
