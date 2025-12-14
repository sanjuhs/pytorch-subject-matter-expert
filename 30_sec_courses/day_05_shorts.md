# Day 05 — 30-Second Shorts (Loss Functions + Optimizers)

These are **script-ready** shorts. Each short includes:
- **Title**
- **Hook** (1 sentence)
- **Talk track** (what you say, ~20 seconds)
- **Code** (what you show)
- **Common mistake** (quick pitfall)

---

## Short 01 — Day 5: How Neural Networks Actually Learn

**Hook**
Loss tells you “how wrong”; optimizer tells you “how to fix it.”

**Talk track**
- “You already built models with `nn.Module`.”
- “Now we add: a loss function + an optimizer + a learning rate.”
- “This is the training recipe for basically every neural network.”

**Code**
```python
# Training recipe:
# logits = model(x)
# loss = criterion(logits, y)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
```

**Common mistake**
- Training without understanding if your model output is logits or probabilities (loss choice depends on it).

---

## Short 02 — What Is a Loss Function?

**Hook**
Loss is a single number that measures how wrong your predictions are.

**Talk track**
- “Loss compares predictions to ground truth.”
- “Lower loss is better.”
- “Backprop uses the loss to compute gradients.”

**Code**
```python
import torch
import torch.nn as nn

pred = torch.tensor([2.0, 3.0, 4.0])
true = torch.tensor([1.5, 3.5, 3.0])
loss = nn.MSELoss()(pred, true)
print(loss.item())
```

**Common mistake**
- Comparing losses across different batch sizes without knowing whether your loss is mean or sum.

---

## Short 03 — MSE (L2) for Regression

**Hook**
MSE punishes big mistakes more than small ones.

**Talk track**
- “Regression = predict a real number.”
- “MSE squares the error so large errors get punished heavily.”

**Code**
```python
import torch
import torch.nn as nn

pred = torch.tensor([250000., 180000., 320000.])
true = torch.tensor([240000., 200000., 310000.])
print(nn.MSELoss()(pred, true).item())
```

**Common mistake**
- Using MSE for classification labels (usually wrong; use CrossEntropy / BCE variants).

---

## Short 04 — L1 (MAE) for Regression

**Hook**
MAE is more robust to outliers than MSE.

**Talk track**
- “L1 loss uses absolute error.”
- “It doesn’t blow up as hard on big outliers.”

**Code**
```python
import torch
import torch.nn as nn

pred = torch.tensor([250000., 180000., 320000.])
true = torch.tensor([240000., 200000., 310000.])
print(nn.L1Loss()(pred, true).item())
```

**Common mistake**
- Expecting L1 to have perfectly smooth gradients (it’s less smooth around 0).

---

## Short 05 — Manual MSE in 3 Lines (Build Intuition)

**Hook**
If you can compute it manually, you can debug it.

**Talk track**
- “Manual MSE: error → square → mean.”
- “This is what `nn.MSELoss` does.”

**Code**
```python
import torch

pred = torch.tensor([2.0, 3.0, 4.0])
true = torch.tensor([1.5, 3.5, 3.0])
mse = ((pred - true) ** 2).mean()
print(mse.item())
```

**Common mistake**
- Forgetting `.mean()` and accidentally scaling gradients by batch size.

---

## Short 06 — Binary Classification Loss: BCE

**Hook**
BCE expects probabilities between 0 and 1.

**Talk track**
- “Binary classification = yes/no.”
- “BCE measures how close predicted probabilities are to true labels.”

**Code**
```python
import torch
import torch.nn as nn

probs = torch.tensor([0.9, 0.2, 0.7, 0.1])
targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
print(nn.BCELoss()(probs, targets).item())
```

**Common mistake**
- Feeding raw logits into `BCELoss` (it expects probabilities).

---

## Short 07 — Sigmoid + BCE (Correct Pair)

**Hook**
Logits → sigmoid → probabilities → BCE.

**Talk track**
- “Many models output logits (any real number).”
- “Use sigmoid to convert logits to probabilities before BCE.”

**Code**
```python
import torch
import torch.nn as nn

logits = torch.tensor([2.0, -1.5, 1.0, -2.0])
probs = torch.sigmoid(logits)
targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
print(nn.BCELoss()(probs, targets).item())
```

**Common mistake**
- Applying sigmoid twice (once in model, once outside) and getting weird gradients.

---

## Short 08 — BCEWithLogitsLoss (Best Practice)

**Hook**
This is BCE + sigmoid in a numerically stable package.

**Talk track**
- “Instead of sigmoid + BCE, use `BCEWithLogitsLoss`.”
- “It’s more stable and safer.”

**Code**
```python
import torch
import torch.nn as nn

logits = torch.tensor([2.0, -1.5, 1.0, -2.0])
targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
print(nn.BCEWithLogitsLoss()(logits, targets).item())
```

**Common mistake**
- Using `BCEWithLogitsLoss` but still putting `Sigmoid()` in the model output.

---

## Short 09 — Multi-Class Classification: CrossEntropyLoss

**Hook**
CrossEntropyLoss expects raw logits and integer class indices.

**Talk track**
- “For multi-class problems: digits 0–9, objects, etc.”
- “Provide logits shape `(batch, classes)` and targets shape `(batch,)`.”

**Code**
```python
import torch
import torch.nn as nn

logits = torch.randn(3, 10)     # (batch, classes)
targets = torch.tensor([0, 3, 7])
print(nn.CrossEntropyLoss()(logits, targets).item())
```

**Common mistake**
- Using one-hot targets with `CrossEntropyLoss` (it expects class indices by default).

---

## Short 10 — No Softmax Before CrossEntropyLoss

**Hook**
CrossEntropyLoss already includes log-softmax internally.

**Talk track**
- “Don’t apply softmax before CrossEntropyLoss.”
- “Pass raw logits for numerical stability.”

**Code**
```python
import torch
import torch.nn as nn

logits = torch.randn(4, 3)
targets = torch.tensor([0, 2, 1, 0])
loss = nn.CrossEntropyLoss()(logits, targets)
print(loss.item())
```

**Common mistake**
- “Double softmax”: `softmax(logits)` then CrossEntropyLoss → worse training.

---

## Short 11 — What CrossEntropyLoss Does (Conceptually)

**Hook**
It’s softmax + negative log likelihood of the true class.

**Talk track**
- “Step 1: softmax to probabilities.”
- “Step 2: take probability of the true class.”
- “Step 3: take `-log` and average.”

**Code**
```python
import torch

logits = torch.tensor([[2.0, 1.0, 0.1]])
probs = torch.softmax(logits, dim=1)
loss = -torch.log(probs[0, 0])
print(probs, loss.item())
```

**Common mistake**
- Confusing logits with probabilities when printing model outputs.

---

## Short 12 — Optimizers: What They Actually Do

**Hook**
Optimizers update parameters using gradients.

**Talk track**
- “After backprop, each parameter has a gradient.”
- “Optimizer applies an update rule like gradient descent.”

**Code**
```python
# Conceptually:
# param = param - lr * param.grad
```

**Common mistake**
- Forgetting to call `optimizer.step()` and wondering why the model never changes.

---

## Short 13 — Gradient Descent Toy Example: Minimize (x-3)^2

**Hook**
Watch gradient descent converge in 10 seconds.

**Talk track**
- “We start at x=0.”
- “Backprop gives grad.”
- “Update x toward 3.”

**Code**
```python
import torch

x = torch.tensor(0.0, requires_grad=True)
lr = 0.1
for _ in range(10):
    loss = (x - 3) ** 2
    loss.backward()
    with torch.no_grad():
        x -= lr * x.grad
    x.grad.zero_()
print(x.item())
```

**Common mistake**
- Not zeroing grads → updates become incorrect due to accumulation.

---

## Short 14 — Same Toy Example Using torch.optim.SGD

**Hook**
torch.optim automates updates safely.

**Talk track**
- “Optimizers manage parameter updates.”
- “You still do: zero_grad → backward → step.”

**Code**
```python
import torch
import torch.optim as optim

x = torch.tensor(0.0, requires_grad=True)
opt = optim.SGD([x], lr=0.1)

for _ in range(10):
    loss = (x - 3) ** 2
    opt.zero_grad()
    loss.backward()
    opt.step()
print(x.item())
```

**Common mistake**
- Calling `loss.backward()` before `opt.zero_grad()`.

---

## Short 15 — SGD + Momentum (Why Momentum Helps)

**Hook**
Momentum smooths updates and speeds you up.

**Talk track**
- “Momentum accumulates gradients like a rolling ball.”
- “It reduces zig-zagging and helps convergence.”

**Code**
```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

**Common mistake**
- Using a high learning rate and high momentum together and getting instability.

---

## Short 16 — Adam: The Default “Good Enough” Optimizer

**Hook**
Adam works well with minimal tuning.

**Talk track**
- “Adam adapts learning rates per parameter.”
- “Great default choice for many problems.”

**Code**
```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Common mistake**
- Copying SGD learning rates to Adam (Adam typically uses smaller lr like 1e-3).

---

## Short 17 — RMSprop: Good for Some Non-Stationary Problems

**Hook**
RMSprop normalizes updates by gradient magnitude history.

**Talk track**
- “RMSprop divides lr by a running average of grad magnitudes.”
- “Historically common for RNNs.”

**Code**
```python
import torch.optim as optim

optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
```

**Common mistake**
- Thinking RMSprop is always better; try Adam/AdamW first unless you have a reason.

---

## Short 18 — AdamW: Modern Best Practice + Weight Decay

**Hook**
AdamW is Adam with proper weight decay.

**Talk track**
- “Weight decay is a regularizer that discourages large weights.”
- “AdamW is especially common in transformer training.”

**Code**
```python
import torch.optim as optim

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

**Common mistake**
- Setting `weight_decay` too high and underfitting badly.

---

## Short 19 — Weight Decay Is Not “Just Another LR”

**Hook**
Weight decay changes the optimization behavior.

**Talk track**
- “Weight decay pushes weights toward zero.”
- “It can improve generalization.”

**Code**
```python
# Typical starting point:
# AdamW(lr=1e-3, weight_decay=1e-2)
```

**Common mistake**
- Applying weight decay to bias / batchnorm parameters without thought (many setups exclude them).

---

## Short 20 — The Learning Rate Is the #1 Hyperparameter

**Hook**
If training diverges, suspect learning rate first.

**Talk track**
- “Too small: painfully slow.”
- “Too big: loss explodes/diverges.”
- “Try a few values quickly.”

**Code**
```python
learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
```

**Common mistake**
- Not resetting the model when comparing learning rates (you’re not running a fair test).

---

## Short 21 — Quick LR Comparison Loop

**Hook**
Try 4 learning rates in under a minute.

**Talk track**
- “Reset model each time.”
- “Train a few steps and see which one decreases loss fastest without exploding.”

**Code**
```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
X = torch.randn(100, 5)
y = torch.randn(100, 1)
criterion = nn.MSELoss()

for lr in [0.001, 0.01, 0.1]:
    model = nn.Linear(5, 1)
    opt = optim.SGD(model.parameters(), lr=lr)
    for _ in range(5):
        opt.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        opt.step()
    print(lr, loss.item())
```

**Common mistake**
- Comparing losses from different random initializations without setting a seed.

---

## Short 22 — Learning Rate Schedulers: Why Use Them?

**Hook**
Good schedules can improve final accuracy.

**Talk track**
- “Start with a larger LR to learn fast.”
- “Then reduce LR to refine and converge.”

**Code**
```python
# LR schedule idea:
# big steps early, small steps later
```

**Common mistake**
- Using a scheduler but forgetting to call `scheduler.step()`.

---

## Short 23 — StepLR (Drop LR Every N Epochs)

**Hook**
StepLR is the simplest scheduler.

**Talk track**
- “Every `step_size` epochs, multiply LR by `gamma`.”
- “Easy and effective baseline.”

**Code**
```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
```

**Common mistake**
- Stepping the scheduler at the wrong time (be consistent: end of epoch is common).

---

## Short 24 — ExponentialLR (Decay a Little Every Epoch)

**Hook**
Smooth decay with one parameter.

**Talk track**
- “Multiply LR by `gamma` every epoch.”
- “Use when you want gradual decay.”

**Code**
```python
from torch.optim.lr_scheduler import ExponentialLR

scheduler = ExponentialLR(optimizer, gamma=0.95)
```

**Common mistake**
- Setting gamma too small and killing learning too early.

---

## Short 25 — CosineAnnealingLR (Popular for Modern Training)

**Hook**
Cosine decay is common in transformer recipes.

**Talk track**
- “LR follows a cosine curve down to a minimum.”
- “Often helps smooth training.”

**Code**
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=50)
```

**Common mistake**
- Misunderstanding `T_max` (it’s the cycle length in epochs/steps, depending on how you step).

---

## Short 26 — How to Print the Current LR

**Hook**
Always log LR when debugging training.

**Talk track**
- “If training stalls, check LR.”
- “This one line tells you what LR is right now.”

**Code**
```python
current_lr = optimizer.param_groups[0]["lr"]
print(current_lr)
```

**Common mistake**
- Using multiple param groups and only logging the first one unintentionally.

---

## Short 27 — Gradient Clipping: Prevent Exploding Gradients

**Hook**
If loss spikes hard, clipping might save you.

**Talk track**
- “Exploding gradients can happen in deep nets or unstable setups.”
- “Clip the gradient norm before optimizer.step().”

**Code**
```python
import torch.nn.utils as U

U.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Common mistake**
- Clipping after `optimizer.step()` (too late).

---

## Short 28 — The “Golden” Training Loop (Production Template)

**Hook**
This is the exact loop pattern you’ll reuse forever.

**Talk track**
- “Forward pass produces logits.”
- “Loss compares to labels.”
- “Zero grads, backward, optional clip, step.”

**Code**
```python
# outputs = model(batch_X)
# loss = criterion(outputs, batch_y)
# optimizer.zero_grad()
# loss.backward()
# torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # optional
# optimizer.step()
```

**Common mistake**
- Forgetting `model.train()` during training (matters with dropout/batchnorm).

---

## Short 29 — Compare Optimizers (The Right Way)

**Hook**
Fair optimizer comparisons require resetting everything.

**Talk track**
- “Same model init, same data order, same epochs.”
- “Otherwise you’re comparing randomness, not optimizers.”

**Code**
```python
# Fair test checklist:
# - same seed
# - same initialization
# - same batches
# - same number of updates
```

**Common mistake**
- Reusing the same model instance across optimizer tests (unfair).

---

## Short 30 — Loss Selection Cheat Sheet

**Hook**
Pick the right loss and training gets 10x easier.

**Talk track**
- “Regression: MSE / L1.”
- “Binary classification: BCEWithLogitsLoss.”
- “Multi-class: CrossEntropyLoss.”

**Code**
```python
# Regression: nn.MSELoss() / nn.L1Loss()
# Binary: nn.BCEWithLogitsLoss()
# Multi-class: nn.CrossEntropyLoss()
```

**Common mistake**
- Using BCE for multi-class problems (use CE), or using CE for multi-label (often BCEWithLogits).

---

## Short 31 — Output Layer Cheat Sheet (What Your Model Should Return)

**Hook**
The loss you use decides whether you apply sigmoid/softmax.

**Talk track**
- “For CrossEntropyLoss: return logits (no softmax).”
- “For BCEWithLogitsLoss: return logits (no sigmoid).”
- “Only apply sigmoid/softmax for inference/visualization.”

**Code**
```python
# CE: model -> logits
# BCEWithLogits: model -> logits
# Inference: probs = softmax/logits or sigmoid(logits)
```

**Common mistake**
- Baking softmax/sigmoid into the model and then using the “with logits” loss.

---

## Short 32 — Day 5 Summary: If Training Is Bad, Check These 3

**Hook**
Most training failures come down to 3 things.

**Talk track**
- “1) Wrong loss for the task.”
- “2) Learning rate too high/low.”
- “3) Output mismatch (logits vs probabilities).”

**Code**
```python
# Debug order:
# 1) loss/task match
# 2) logits vs probs match
# 3) learning rate
```

**Common mistake**
- Changing 5 things at once—change one variable at a time when debugging.


