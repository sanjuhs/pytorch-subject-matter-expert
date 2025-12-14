# Day 03 — 30-Second Shorts (Autograd + Backprop + Activations)

These are **script-ready** shorts. Each short includes:
- **Title**
- **Hook** (1 sentence)
- **Talk track** (what you say, ~20 seconds)
- **Code** (what you show)
- **Common mistake** (quick pitfall)

---

## Short 01 — PyTorch “Magic”: Autograd in One Word

**Hook**
Autograd is the reason PyTorch feels like cheating.

**Talk track**
- “Autograd means PyTorch automatically computes derivatives for you.”
- “If a tensor has `requires_grad=True`, PyTorch tracks operations and builds a graph.”
- “Then one call to `.backward()` gives gradients.”

**Code**
```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print("y:", y.item(), "dy/dx:", x.grad.item())
```

**Common mistake**
- Forgetting `requires_grad=True` and wondering why `.grad` is `None`.

---

## Short 02 — What Does requires_grad Actually Do?

**Hook**
This flag decides whether PyTorch tracks your math.

**Talk track**
- “`requires_grad=True` tells PyTorch: keep a history of ops for backprop.”
- “It’s usually ON for model parameters, OFF for input data.”

**Code**
```python
import torch

a = torch.tensor(2.0, requires_grad=False)
b = torch.tensor(2.0, requires_grad=True)
print(a.requires_grad, b.requires_grad)
```

**Common mistake**
- Setting `requires_grad=True` on huge input batches unnecessarily (slows training, uses more memory).

---

## Short 03 — Backprop in One Line: .backward()

**Hook**
This is the line that makes networks learn.

**Talk track**
- “`.backward()` computes gradients of a scalar output with respect to everything that requires grad.”
- “Gradients get stored in `.grad` on leaf tensors.”

**Code**
```python
import torch

w = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(2.0)  # data
y = w * x
y.backward()
print("dy/dw:", w.grad.item())
```

**Common mistake**
- Calling `.backward()` on a non-scalar tensor without supplying `gradient=...`.

---

## Short 04 — The Computational Graph (Concept Only)

**Hook**
Every op you do becomes a node in a graph.

**Talk track**
- “PyTorch builds a computational graph as you compute forward.”
- “Backward walks the graph in reverse using the chain rule.”

**Code**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = (x + 1) * (x + 2)
print("y.grad_fn:", y.grad_fn)
```

**Common mistake**
- Thinking the graph is static; it’s **built dynamically each forward pass**.

---

## Short 05 — Multiple Parameters: Gradients for x, w, b

**Hook**
One backward call can compute multiple gradients.

**Talk track**
- “If `loss` depends on many parameters, `.backward()` fills all their `.grad` fields.”
- “That’s the basis of training.”

**Code**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

target = 10.0
y = w * x + b
loss = (y - target) ** 2
loss.backward()

print(x.grad, w.grad, b.grad)
```

**Common mistake**
- Forgetting to reset grads between steps (they accumulate).

---

## Short 06 — Gradients Accumulate (This Surprises Everyone)

**Hook**
PyTorch doesn’t overwrite gradients — it adds them.

**Talk track**
- “By default, gradients accumulate into `.grad`.”
- “So in training, you must zero gradients each iteration.”

**Code**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
(x**2).backward()
print("after 1st:", x.grad.item())

(x**3).backward()
print("after 2nd:", x.grad.item())
```

**Common mistake**
- Not clearing grads → training explodes or behaves randomly.

---

## Short 07 — The Fix: Zero Gradients

**Hook**
Zero grads = your “reset” button for training.

**Talk track**
- “Before computing a new gradient, clear the old one.”
- “With raw tensors: `param.grad.zero_()` (if it exists).”

**Code**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
(x**2).backward()
print(x.grad.item())

x.grad.zero_()
(x**2).backward()
print(x.grad.item())
```

**Common mistake**
- Calling `.zero_()` when `.grad` is `None` (first iteration). Check `if x.grad is not None:`.

---

## Short 08 — Why Activations Exist (Linear Layers Collapse)

**Hook**
Without activations, “deep” networks are secretly shallow.

**Talk track**
- “Two linear layers with no nonlinearity collapse into one linear layer.”
- “Activations add nonlinearity so the model can learn complex patterns.”

**Code**
```python
import torch

x = torch.randn(1, 3)
W1 = torch.randn(3, 4)
W2 = torch.randn(4, 2)

y1 = (x @ W1) @ W2
y2 = x @ (W1 @ W2)
print(torch.allclose(y1, y2))
```

**Common mistake**
- Stacking `nn.Linear` layers and wondering why performance doesn’t improve without activations.

---

## Short 09 — ReLU: The Default Activation

**Hook**
ReLU is the workhorse of deep learning.

**Talk track**
- “ReLU is `max(0, x)`.”
- “It’s simple, fast, and avoids vanishing gradients on positive values.”

**Code**
```python
import torch

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(torch.relu(x))
```

**Common mistake**
- Using ReLU at the final layer when you need outputs that can be negative.

---

## Short 10 — ReLU Gradient: 0 or 1

**Hook**
ReLU either passes gradient or blocks it.

**Talk track**
- “For negative inputs, gradient is 0.”
- “For positive inputs, gradient is 1.”

**Code**
```python
import torch

x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
y = torch.relu(x).sum()
y.backward()
print(x.grad)
```

**Common mistake**
- “Dying ReLU”: if a neuron stays negative, it can stop learning (gradient 0).

---

## Short 11 — Leaky ReLU: Fix Dying ReLUs

**Hook**
Give negative inputs a tiny gradient.

**Talk track**
- “Leaky ReLU uses `max(alpha*x, x)`.”
- “So negatives still get a small gradient and can recover.”

**Code**
```python
import torch
import torch.nn.functional as F

x = torch.tensor([-3.0, -1.0, 0.0, 2.0])
print(F.leaky_relu(x, negative_slope=0.01))
```

**Common mistake**
- Forgetting to tune `negative_slope` when you actually care about negative-region behavior.

---

## Short 12 — Sigmoid: Probabilities for Binary Output

**Hook**
Sigmoid is great — but mostly at the output.

**Talk track**
- “Sigmoid maps numbers into (0,1).”
- “Use it for binary classification probabilities.”
- “But avoid in deep hidden layers due to vanishing gradients.”

**Code**
```python
import torch

logits = torch.tensor([-2.0, 0.0, 2.0])
print(torch.sigmoid(logits))
```

**Common mistake**
- Using sigmoid in hidden layers and seeing slow learning (vanishing gradients).

---

## Short 13 — Vanishing Gradient: Sigmoid Saturation

**Hook**
Big positive/negative inputs kill gradient.

**Talk track**
- “At large |x|, sigmoid becomes flat.”
- “Flat means derivative ~ 0, learning slows down.”

**Code**
```python
import torch

x = torch.tensor([-5.0, 0.0, 5.0], requires_grad=True)
torch.sigmoid(x).sum().backward()
print(x.grad)
```

**Common mistake**
- Feeding unnormalized data into sigmoid-heavy networks.

---

## Short 14 — Tanh: Like Sigmoid But Zero-Centered

**Hook**
Tanh outputs -1 to 1 instead of 0 to 1.

**Talk track**
- “Tanh is zero-centered, which can help optimization.”
- “Still can saturate and vanish gradients for large |x|.”

**Code**
```python
import torch

x = torch.linspace(-3, 3, 7)
print(torch.tanh(x))
```

**Common mistake**
- Thinking tanh “solves” vanishing gradients (it helps centering, but still saturates).

---

## Short 15 — Softmax: Multi-Class Probabilities

**Hook**
Softmax turns logits into a probability distribution.

**Talk track**
- “Softmax outputs sum to 1.”
- “Use it for multi-class probabilities.”

**Code**
```python
import torch

logits = torch.tensor([2.0, 1.0, 0.1])
probs = torch.softmax(logits, dim=0)
print(probs, probs.sum())
```

**Common mistake**
- Wrong `dim` for batched logits: for `(batch, classes)` use `dim=1`.

---

## Short 16 — Backprop = Chain Rule (Tiny Example)

**Hook**
Backprop is just the chain rule repeated.

**Talk track**
- “We do forward: h → g → f.”
- “Backward computes \(\frac{df}{dx}\) automatically.”

**Code**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
h = x**2
g = h + 3
f = g**2
f.backward()
print(x.grad.item())
```

**Common mistake**
- Doing `.item()` too early and breaking the graph (only convert to Python scalars for logging).

---

## Short 17 — A Full Forward: Linear → ReLU → Loss

**Hook**
This is the “training pipeline” skeleton.

**Talk track**
- “Forward pass produces predictions.”
- “Loss compares predictions to target.”
- “Backward computes gradients for parameters.”

**Code**
```python
import torch

X = torch.randn(3, 4)                    # data
W = torch.randn(4, 2, requires_grad=True)
b = torch.randn(2, requires_grad=True)

Z = X @ W + b
A = torch.relu(Z)
target = torch.ones_like(A)
loss = ((A - target) ** 2).mean()
loss.backward()

print(W.grad.shape, b.grad.shape)
```

**Common mistake**
- In-place ops on tensors that require grad can break autograd in subtle ways.

---

## Short 18 — Gradient Descent Update (Manual)

**Hook**
This is literally how learning happens.

**Talk track**
- “After backward, update parameters: \( \theta \leftarrow \theta - \alpha \nabla_\theta L \).”
- “Use `torch.no_grad()` so the update isn’t tracked.”

**Code**
```python
import torch

W = torch.randn(4, 2, requires_grad=True)
loss = (W**2).mean()
loss.backward()

with torch.no_grad():
    W -= 0.1 * W.grad
```

**Common mistake**
- Updating parameters without `no_grad()` and accidentally building a giant graph over steps.

---

## Short 19 — Training Loop Template (The 5 Steps)

**Hook**
Memorize this loop and you can train anything.

**Talk track**
- “Forward → loss → backward → update → zero grads.”
- “That’s 90% of deep learning training.”

**Code**
```python
# 1) forward
# 2) loss
# 3) backward
# 4) update
# 5) zero grads
```

**Common mistake**
- Skipping step 5 (zero grads) and getting accumulated gradients.

---

## Short 20 — Minimal Linear Regression Training (Tensor-Only)

**Hook**
Train a model without nn.Module just to understand it.

**Talk track**
- “We’ll fit y = 2x + 1 from synthetic data.”
- “Watch W and b converge.”

**Code**
```python
import torch

torch.manual_seed(42)
X = torch.randn(100, 1)
y = 2 * X + 1 + 0.1 * torch.randn(100, 1)

W = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
lr = 0.01

for _ in range(50):
    y_pred = X @ W + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
    W.grad.zero_()
    b.grad.zero_()

print(W.item(), b.item())
```

**Common mistake**
- Calling `.backward()` twice on the same graph without `retain_graph=True`.

---

## Short 21 — torch.no_grad(): Inference Mode

**Hook**
When you’re not training, don’t track gradients.

**Talk track**
- “Inference and evaluation should be faster and use less memory.”
- “Wrap it in `torch.no_grad()`.”

**Code**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    y = x**2
print(y.requires_grad)
```

**Common mistake**
- Forgetting `no_grad()` during evaluation → unnecessary memory use.

---

## Short 22 — torch.set_grad_enabled(): Toggle Training/Eval

**Hook**
One context manager to rule training vs eval.

**Talk track**
- “Sometimes you want conditional grad tracking.”
- “Use `torch.set_grad_enabled(is_training)`.”

**Code**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
is_training = False
with torch.set_grad_enabled(is_training):
    y = x**2
print(y.requires_grad)
```

**Common mistake**
- Confusing this with `model.train()` / `model.eval()` (they are related but not the same thing).

---

## Short 23 — detach(): Stop Gradients Through a Tensor

**Hook**
Detach is how you “cut” the graph.

**Talk track**
- “`detach()` returns a tensor with same values but no gradient history.”
- “Useful for logging, targets, or stop-gradient tricks.”

**Code**
```python
import torch

x = torch.tensor(3.0, requires_grad=True)
z = x**2
z_detached = z.detach()
print(z.requires_grad, z_detached.requires_grad)
```

**Common mistake**
- Detaching something you still want to learn through (your model stops training).

---

## Short 24 — .grad Exists Only for Leaf Tensors

**Hook**
Why is `.grad` None even though you called backward?

**Talk track**
- “Only leaf tensors (like parameters) store grads by default.”
- “Intermediate tensors have `grad_fn`, but no `.grad` unless you call `.retain_grad()`.”

**Code**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print("x.grad:", x.grad)
print("y.grad:", y.grad)
```

**Common mistake**
- Expecting `.grad` on non-leaf tensors like activations.

---

## Short 25 — retain_grad() for Debugging Intermediates

**Hook**
Want gradients of activations? Tell PyTorch.

**Talk track**
- “For debugging, you can keep grads on intermediates with `.retain_grad()`.”

**Code**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.retain_grad()
y.backward()
print("y.grad:", y.grad)
```

**Common mistake**
- Using `retain_grad()` everywhere — it increases memory usage.

---

## Short 26 — In-Place Ops Can Break Autograd

**Hook**
One underscore can ruin your gradients.

**Talk track**
- “In-place ops like `relu_()` or `add_()` modify tensors.”
- “That can break autograd if the original values are needed for backward.”

**Code**
```python
import torch

x = torch.tensor([-1.0, 2.0], requires_grad=True)
y = torch.relu(x)     # safe
z = y.sum()
z.backward()
print(x.grad)
```

**Common mistake**
- Using in-place ops during training and getting “one of the variables needed for gradient computation has been modified”.

---

## Short 27 — Softmax dim for Batches (Most Common Bug)

**Hook**
Softmax is correct… until you pick the wrong dimension.

**Talk track**
- “For shape `(batch, classes)`, softmax across classes: `dim=1`.”

**Code**
```python
import torch

batch_logits = torch.tensor([[2.0, 1.0, 0.1],
                             [0.5, 2.1, 0.3]])
probs = torch.softmax(batch_logits, dim=1)
print(probs.sum(dim=1))
```

**Common mistake**
- Using `dim=0` and accidentally normalizing across the batch.

---

## Short 28 — Why We Don’t Use Softmax + CrossEntropy Manually

**Hook**
PyTorch already does the stable version for you.

**Talk track**
- “In real training: use `nn.CrossEntropyLoss` on raw logits.”
- “It combines log-softmax + NLL in a numerically stable way.”

**Code**
```python
import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()
logits = torch.randn(4, 3)          # (batch, classes)
targets = torch.tensor([0, 2, 1, 0])
loss = loss_fn(logits, targets)
print(loss.item())
```

**Common mistake**
- Applying `softmax` before `CrossEntropyLoss` (hurts numerical stability and gradients).

---

## Short 29 — Backward Twice? Use retain_graph=True

**Hook**
Why does the second backward crash?

**Talk track**
- “By default, PyTorch frees the graph after backward to save memory.”
- “If you need backward twice, retain it explicitly.”

**Code**
```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward(retain_graph=True)
y.backward()
print(x.grad)
```

**Common mistake**
- Retaining graphs unnecessarily (memory blow-up). Only do it when needed.

---

## Short 30 — The Real Summary: Forward → Loss → Backward → Update

**Hook**
You now understand how “learning” actually happens.

**Talk track**
- “Forward computes predictions.”
- “Loss says how wrong we are.”
- “Backward computes how to change parameters.”
- “Update applies that change.”

**Code**
```python
# Forward → Loss → Backward → Update → Zero grads
```

**Common mistake**
- Updating parameters but forgetting to set a learning rate that’s not crazy (too high explodes, too low stalls).

---

## Short 31 — Mini Challenge: Predict y = 2x + 1

**Hook**
If you can do this, you understand Day 3.

**Talk track**
- “Generate data, train W and b, print final values.”
- “Aim: W≈2, b≈1.”

**Code**
```python
# Try it: copy Short 20 and increase epochs to 200.
```

**Common mistake**
- Not setting a seed and thinking “my code is wrong” because numbers differ each run.

---

## Short 32 — Preview Day 4: nn.Module Makes This Clean

**Hook**
Tomorrow we stop doing everything manually.

**Talk track**
- “Day 4: we’ll use `nn.Module`, `nn.Linear`, losses, optimizers.”
- “Same ideas, cleaner code.”

**Code**
```python
# Day 4: nn.Module + nn.Linear + optimizers
```


