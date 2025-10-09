# Day 3: Autograd, Backpropagation, and Activation Functions

**Duration**: 10 minutes
**Platform**: Google Colab

---

## Video Script

### Introduction (30 seconds)

**"Welcome to Day 3 - where PyTorch becomes magical!"**

Today we're diving into **autograd** - PyTorch's automatic differentiation engine. This is what makes deep learning possible. We'll understand:
- How gradients are computed automatically
- What backpropagation really does
- Why activation functions are crucial
- How neural networks learn

By the end, you'll understand the math that makes neural networks learn from data.

---

### What is Autograd? (1 minute)

**"Autograd tracks every operation and computes gradients automatically"**

```python
import torch

# requires_grad=True tells PyTorch to track operations
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}")
print(f"Tracking gradients: {x.requires_grad}")

# Perform operations
y = x ** 2 + 2 * x + 1
print(f"y = x² + 2x + 1 = {y}")

# Compute gradient dy/dx
y.backward()  # This is the magic!
print(f"dy/dx = {x.grad}")  # Should be 2x + 2 = 2(3) + 2 = 8
```

**What just happened?**
- PyTorch built a **computational graph** tracking: `x → x² → + 2x → + 1 → y`
- `.backward()` computed derivatives using **chain rule**
- Result stored in `.grad` attribute

---

### The Computational Graph (1.5 minutes)

**"PyTorch builds a graph of operations behind the scenes"**

```python
# More complex example
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Linear transformation: y = wx + b
y = w * x + b
print(f"y = wx + b = {y}")

# Squared error loss: L = (y - target)²
target = 10.0
loss = (y - target) ** 2
print(f"Loss = (y - 10)² = {loss}")

# Compute all gradients
loss.backward()

print(f"\n∂L/∂x = {x.grad}")  # How loss changes with x
print(f"∂L/∂w = {w.grad}")    # How loss changes with w
print(f"∂L/∂b = {b.grad}")    # How loss changes with b
```

**Understanding gradients:**
- Positive gradient = increasing parameter increases loss
- Negative gradient = increasing parameter decreases loss
- Magnitude = how sensitive loss is to that parameter

```python
# Visualizing the computational graph
# L = (wx + b - target)²
# ∂L/∂w = 2(wx + b - target) * x = 2(7 - 10) * 2 = -12
# ∂L/∂b = 2(wx + b - target) * 1 = 2(7 - 10) = -6
# ∂L/∂x = 2(wx + b - target) * w = 2(7 - 10) * 3 = -18
```

---

### Multiple Backward Passes and Gradient Accumulation (1 minute)

**"Important: gradients accumulate by default!"**

```python
# Gradient accumulation
x = torch.tensor(2.0, requires_grad=True)

# First computation
y1 = x ** 2
y1.backward()
print(f"After first backward: x.grad = {x.grad}")  # 4

# Second computation (gradients accumulate!)
y2 = x ** 3
y2.backward()
print(f"After second backward: x.grad = {x.grad}")  # 4 + 12 = 16

# Always zero gradients before new backward pass!
x.grad.zero_()
y3 = x ** 2
y3.backward()
print(f"After zeroing and backward: x.grad = {x.grad}")  # 4
```

**Key lesson:** Always call `.zero_grad()` before computing new gradients in training loops!

---

### Activation Functions: Why Do We Need Them? (2 minutes)

**"Without activation functions, neural networks are just linear regression"**

**The problem with pure linear layers:**

```python
# Two linear layers without activation
x = torch.randn(1, 3)
W1 = torch.randn(3, 4)
W2 = torch.randn(4, 2)

# Two layer network
h = x @ W1      # Hidden layer
y = h @ W2      # Output layer

# This is equivalent to single layer!
W_combined = W1 @ W2  # Can collapse to single matrix
y_collapsed = x @ W_combined

print(f"Two layers: {y}")
print(f"One layer: {y_collapsed}")
print(f"Equal? {torch.allclose(y, y_collapsed)}")  # True!
```

**Activation functions introduce non-linearity:**

---

### Activation Functions Deep Dive (2.5 minutes)

#### 1. ReLU (Rectified Linear Unit) - The Workhorse

**"Most popular activation function in modern deep learning"**

```python
# ReLU: f(x) = max(0, x)
x = torch.linspace(-3, 3, 10)
relu_output = torch.relu(x)

print(f"Input:  {x}")
print(f"ReLU:   {relu_output}")

# Why ReLU?
# ✓ Simple and fast
# ✓ No vanishing gradient for positive values
# ✓ Sparse activation (many zeros)
# ✗ "Dying ReLU" problem (neurons can get stuck at 0)

# ReLU gradient
x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
y = torch.relu(x)
y.sum().backward()

print(f"\nReLU gradient: {x.grad}")  # [0, 0, 1] - gradient is 0 or 1
```

#### 2. Leaky ReLU - Fixing Dying ReLU

```python
# Leaky ReLU: f(x) = max(0.01x, x)
x = torch.linspace(-3, 3, 10)
leaky_relu = torch.nn.functional.leaky_relu(x, negative_slope=0.01)

print(f"Input:       {x}")
print(f"Leaky ReLU:  {leaky_relu}")

# Advantage: negative values get small gradient (0.01) instead of 0
# This prevents "dying neurons"
```

#### 3. Sigmoid - Squashing to (0, 1)

```python
# Sigmoid: f(x) = 1 / (1 + e^(-x))
x = torch.linspace(-5, 5, 11)
sigmoid_output = torch.sigmoid(x)

print(f"Input:   {x}")
print(f"Sigmoid: {sigmoid_output}")

# When to use:
# ✓ Binary classification output (probability)
# ✓ Gate mechanisms (LSTM, GRU)
# ✗ Hidden layers (vanishing gradient problem)

# Visualize the vanishing gradient problem
x = torch.tensor([-5.0, 0.0, 5.0], requires_grad=True)
y = torch.sigmoid(x)
y.sum().backward()

print(f"\nSigmoid gradient at x={x.data}: {x.grad}")
# Notice: gradients near 0 for large |x| (vanishing gradient!)
```

#### 4. Tanh - Centered Sigmoid

```python
# Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
x = torch.linspace(-3, 3, 11)
tanh_output = torch.tanh(x)

print(f"Input: {x}")
print(f"Tanh:  {tanh_output}")

# Advantage over sigmoid: outputs centered at 0 (range: -1 to 1)
# Still suffers from vanishing gradient
```

#### 5. Softmax - Converting to Probabilities

```python
# Softmax: converts logits to probabilities (sum = 1)
logits = torch.tensor([2.0, 1.0, 0.1])
probabilities = torch.softmax(logits, dim=0)

print(f"Logits:        {logits}")
print(f"Probabilities: {probabilities}")
print(f"Sum:           {probabilities.sum()}")  # 1.0

# Multi-class classification example
batch_logits = torch.tensor([[2.0, 1.0, 0.1],
                               [0.5, 2.1, 0.3],
                               [1.0, 1.0, 1.0]])

batch_probs = torch.softmax(batch_logits, dim=1)
print(f"\nBatch probabilities:\n{batch_probs}")
print(f"Row sums: {batch_probs.sum(dim=1)}")  # All 1.0
```

---

### Complete Neural Network Layer with Gradients (1.5 minutes)

**"Let's build a layer and track gradients through everything"**

```python
# Input data
torch.manual_seed(42)
X = torch.randn(3, 4, requires_grad=False)  # 3 samples, 4 features

# Parameters (learnable)
W = torch.randn(4, 2, requires_grad=True)   # 4 inputs -> 2 outputs
b = torch.randn(2, requires_grad=True)       # Bias for 2 outputs

print(f"Initial W:\n{W.data}")
print(f"Initial b: {b.data}")

# Forward pass
Z = X @ W + b           # Linear transformation
A = torch.relu(Z)       # Activation
print(f"\nActivations:\n{A}")

# Loss: mean squared error with target
target = torch.ones_like(A)
loss = ((A - target) ** 2).mean()
print(f"\nLoss: {loss.item()}")

# Backward pass
loss.backward()

# Gradients
print(f"\n∂L/∂W:\n{W.grad}")
print(f"∂L/∂b: {b.grad}")

# Gradient descent update (learning rate = 0.1)
with torch.no_grad():  # Don't track these operations
    W -= 0.1 * W.grad
    b -= 0.1 * b.grad

print(f"\nUpdated W:\n{W.data}")
print(f"Updated b: {b.data}")
```

---

### Understanding Backpropagation: The Chain Rule (1 minute)

**"Backpropagation is just the chain rule applied systematically"**

```python
# Simple example: f(g(h(x)))
x = torch.tensor(2.0, requires_grad=True)

# Forward pass
h = x ** 2          # h(x) = x²
g = h + 3           # g(h) = h + 3
f = g ** 2          # f(g) = g²

print(f"x = {x.item()}")
print(f"h = x² = {h.item()}")
print(f"g = h + 3 = {g.item()}")
print(f"f = g² = {f.item()}")

# Backward pass
f.backward()

print(f"\n∂f/∂x = {x.grad.item()}")

# Manual calculation:
# ∂f/∂x = (∂f/∂g) * (∂g/∂h) * (∂h/∂x)
# ∂f/∂g = 2g = 2(7) = 14
# ∂g/∂h = 1
# ∂h/∂x = 2x = 2(2) = 4
# ∂f/∂x = 14 * 1 * 4 = 56
print(f"Manual calculation: 2g * 1 * 2x = 2({g.item()}) * 1 * 2({x.item()}) = {2 * g.item() * 2 * x.item()}")
```

---

### Practical Training Loop (1.5 minutes)

**"Let's put it all together: a complete training loop"**

```python
import torch
import torch.nn as nn

# Generate synthetic data: y = 2x + 1 (with noise)
torch.manual_seed(42)
X = torch.randn(100, 1)
y_true = 2 * X + 1 + 0.1 * torch.randn(100, 1)

# Initialize parameters
W = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Hyperparameters
learning_rate = 0.01
epochs = 100

print("Training a simple linear model: y = Wx + b")
print(f"Initial W: {W.item():.4f}, b: {b.item():.4f}\n")

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = X @ W + b

    # Compute loss (MSE)
    loss = ((y_pred - y_true) ** 2).mean()

    # Backward pass
    loss.backward()

    # Update parameters
    with torch.no_grad():
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    # Zero gradients
    W.grad.zero_()
    b.grad.zero_()

    # Print progress
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.6f}, W = {W.item():.4f}, b = {b.item():.4f}")

print(f"\nFinal W: {W.item():.4f} (target: 2.0)")
print(f"Final b: {b.item():.4f} (target: 1.0)")
```

---

### Context Managers: torch.no_grad() and torch.set_grad_enabled() (45 seconds)

**"Control when gradients are tracked"**

```python
# torch.no_grad() - disable gradient tracking
x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x ** 2  # No graph built
    print(f"y.requires_grad: {y.requires_grad}")  # False

# Useful for:
# - Inference (testing)
# - Parameter updates
# - Operations that shouldn't affect gradients

# torch.set_grad_enabled() - conditional tracking
is_training = False

with torch.set_grad_enabled(is_training):
    y = x ** 2
    print(f"In training mode: {y.requires_grad}")  # False

# .detach() - detach from computation graph
z = x ** 2
z_detached = z.detach()  # Same values, no gradients
print(f"z.requires_grad: {z.requires_grad}")           # True
print(f"z_detached.requires_grad: {z_detached.requires_grad}")  # False
```

---

## Key Takeaways

1. **Autograd** - PyTorch automatically computes gradients using chain rule
2. **Computational Graph** - Operations are tracked to enable backpropagation
3. **`.backward()`** - Computes all gradients in one call
4. **`.zero_grad()`** - Always clear gradients before new backward pass
5. **Activation Functions**:
   - **ReLU** - Default choice, fast and effective
   - **Leaky ReLU** - Fixes dying ReLU problem
   - **Sigmoid** - For binary outputs and gates
   - **Tanh** - Zero-centered alternative to sigmoid
   - **Softmax** - For multi-class probabilities
6. **Why activations matter** - Enable neural networks to learn non-linear patterns
7. **Training loop**: Forward → Loss → Backward → Update → Zero grads

---

## Today's Practice Exercise

```python
# Build and train a 2-layer neural network from scratch
import torch

# Generate spiral dataset (non-linearly separable)
torch.manual_seed(42)
n_samples = 100

# Class 0: points in inner circle
theta0 = torch.rand(n_samples) * 2 * 3.14159
r0 = torch.rand(n_samples) * 2
X0 = torch.stack([r0 * torch.cos(theta0), r0 * torch.sin(theta0)], dim=1)
y0 = torch.zeros(n_samples, 1)

# Class 1: points in outer circle
theta1 = torch.rand(n_samples) * 2 * 3.14159
r1 = torch.rand(n_samples) * 2 + 3
X1 = torch.stack([r1 * torch.cos(theta1), r1 * torch.sin(theta1)], dim=1)
y1 = torch.ones(n_samples, 1)

# Combine
X = torch.cat([X0, X1], dim=0)
y = torch.cat([y0, y1], dim=0)

# Initialize 2-layer network: 2 -> 8 -> 1
W1 = torch.randn(2, 8, requires_grad=True) * 0.5
b1 = torch.zeros(8, requires_grad=True)
W2 = torch.randn(8, 1, requires_grad=True) * 0.5
b2 = torch.zeros(1, requires_grad=True)

# Training
learning_rate = 0.1
epochs = 200

for epoch in range(epochs):
    # Forward pass
    z1 = X @ W1 + b1
    a1 = torch.relu(z1)  # Hidden layer activation
    z2 = a1 @ W2 + b2
    y_pred = torch.sigmoid(z2)  # Output activation

    # Binary cross-entropy loss
    loss = -(y * torch.log(y_pred + 1e-8) + (1 - y) * torch.log(1 - y_pred + 1e-8)).mean()

    # Backward pass
    loss.backward()

    # Update parameters
    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad

    # Zero gradients
    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()

    if (epoch + 1) % 50 == 0:
        accuracy = ((y_pred > 0.5).float() == y).float().mean()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.4f}")

# Final test
with torch.no_grad():
    z1 = X @ W1 + b1
    a1 = torch.relu(z1)
    z2 = a1 @ W2 + b2
    y_pred = torch.sigmoid(z2)
    final_accuracy = ((y_pred > 0.5).float() == y).float().mean()
    print(f"\nFinal Accuracy: {final_accuracy.item():.4f}")
```

---

## Tomorrow's Preview

**Day 4: Building Neural Networks with nn.Module**

- Using `torch.nn.Module` for clean architecture
- Built-in layers: `nn.Linear`, `nn.Conv2d`
- Loss functions: `nn.MSELoss`, `nn.CrossEntropyLoss`
- Optimizers: `torch.optim.SGD`, `torch.optim.Adam`

---

**"You now understand the magic behind neural network training! Tomorrow we'll learn PyTorch's high-level APIs to make this even easier. 🚀"**
