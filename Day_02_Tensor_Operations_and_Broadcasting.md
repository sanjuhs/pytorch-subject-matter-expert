# Day 2: Tensor Operations and Broadcasting

**Duration**: 10 minutes
**Platform**: Google Colab

---

## Video Script

### Introduction (30 seconds)

**"Welcome back to Day 2!"**

Yesterday we learned Python essentials and created our first tensors. Today we're leveling up with **tensor operations** and **broadcasting** - the secret sauce that makes PyTorch so powerful.

By the end of today, you'll understand how neural networks perform matrix operations efficiently and why shape matters so much in deep learning.

---

### Recap & Setup (30 seconds)

```python
import torch

# Quick recap from Day 1
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(f"Shape: {x.shape}")  # torch.Size([2, 3])
print(f"2 rows, 3 columns")
```

---

### Essential Tensor Operations (2.5 minutes)

**"Let's explore the operations you'll use daily in neural networks"**

**Element-wise operations**

```python
# Element-wise operations (same shape)
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"Add: {a + b}")        # [5., 7., 9.]
print(f"Subtract: {a - b}")   # [-3., -3., -3.]
print(f"Multiply: {a * b}")   # [4., 10., 18.]
print(f"Divide: {a / b}")     # [0.25, 0.4, 0.5]
print(f"Power: {a ** 2}")     # [1., 4., 9.]
```

**Reduction operations**

```python
# Reduce tensors to single values
tensor = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])

print(f"Sum all: {tensor.sum()}")           # 21.0
print(f"Mean all: {tensor.mean()}")         # 3.5
print(f"Max: {tensor.max()}")               # 6.0
print(f"Min: {tensor.min()}")               # 1.0

# Reduce along specific dimension
print(f"Sum rows (dim=0): {tensor.sum(dim=0)}")     # [5., 7., 9.]
print(f"Sum columns (dim=1): {tensor.sum(dim=1)}") # [6., 15.]
print(f"Mean per row: {tensor.mean(dim=1)}")       # [2., 5.]
```

**Activation functions (used in neural networks)**

```python
# ReLU - most common activation function
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
relu = torch.relu(x)  # Max(0, x)
print(f"ReLU: {relu}")  # [0., 0., 0., 1., 2.]

# Sigmoid - squashes to (0, 1)
sigmoid = torch.sigmoid(x)
print(f"Sigmoid: {sigmoid}")

# Softmax - converts to probabilities (sums to 1)
logits = torch.tensor([2.0, 1.0, 0.1])
probs = torch.softmax(logits, dim=0)
print(f"Softmax: {probs}")
print(f"Sum: {probs.sum()}")  # 1.0
```

---

### Broadcasting - The Magic of Shape Compatibility (2.5 minutes)

**"Broadcasting lets you operate on tensors of different shapes"**

**Rule 1: Scalar to any tensor**

```python
# Scalar broadcasts to any shape
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])

result = tensor + 10  # Add 10 to every element
print(f"Add scalar:\n{result}")

result = tensor * 2  # Multiply every element by 2
print(f"Multiply by scalar:\n{result}")
```

**Rule 2: 1D to 2D broadcasting**

```python
# Broadcasting a vector to a matrix
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])  # Shape: (2, 3)

vector = torch.tensor([10, 20, 30])  # Shape: (3,)

# vector broadcasts across rows
result = matrix + vector
print(f"Matrix + Vector:\n{result}")
# [[11, 22, 33],
#  [14, 25, 36]]
```

**Rule 3: Column vector broadcasting**

```python
# Broadcasting a column vector
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])  # Shape: (2, 3)

column = torch.tensor([[10],
                       [20]])  # Shape: (2, 1)

result = matrix + column
print(f"Matrix + Column:\n{result}")
# [[11, 12, 13],
#  [24, 25, 26]]
```

**Rule 4: Broadcasting rules**

```python
# Broadcasting works when:
# 1. Dimensions are equal, OR
# 2. One dimension is 1, OR
# 3. One dimension doesn't exist

# Examples that work:
a = torch.randn(3, 1, 5)   # Shape: (3, 1, 5)
b = torch.randn(1, 4, 5)   # Shape: (1, 4, 5)
c = a + b                   # Result: (3, 4, 5)
print(f"Broadcast result shape: {c.shape}")

# Real neural network example: adding bias
batch_size = 32
features = 10
activations = torch.randn(batch_size, features)  # (32, 10)
bias = torch.randn(features)                      # (10,)

output = activations + bias  # bias broadcasts to (32, 10)
print(f"Activations + bias shape: {output.shape}")
```

---

### Matrix Multiplication - The Heart of Neural Networks (2 minutes)

**"This is how neural networks compute outputs from inputs"**

**Basic matrix multiplication**

```python
# Matrix multiplication with @
A = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6]])  # Shape: (3, 2)

B = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])  # Shape: (2, 3)

# A @ B requires: A's cols = B's rows
C = A @ B  # or torch.matmul(A, B)
print(f"A @ B shape: {C.shape}")  # (3, 3)
print(f"Result:\n{C}")
```

**Neural network layer simulation**

```python
# Simulating a neural network layer
batch_size = 4      # 4 samples
input_features = 3  # 3 input features
output_features = 5 # 5 output neurons

# Input data (batch)
X = torch.randn(batch_size, input_features)  # (4, 3)

# Weight matrix (learned parameters)
W = torch.randn(input_features, output_features)  # (3, 5)

# Bias (one per output neuron)
b = torch.randn(output_features)  # (5,)

# Forward pass: Y = X @ W + b
Y = X @ W + b  # Broadcasting adds bias
print(f"Input shape: {X.shape}")
print(f"Weight shape: {W.shape}")
print(f"Output shape: {Y.shape}")  # (4, 5)
print(f"Output:\n{Y}")
```

**Batch matrix multiplication**

```python
# Multiple matrix multiplications at once
batch = 3
A = torch.randn(batch, 2, 3)  # 3 matrices of size (2, 3)
B = torch.randn(batch, 3, 4)  # 3 matrices of size (3, 4)

C = torch.bmm(A, B)  # Batch matrix multiply
print(f"Batch matmul shape: {C.shape}")  # (3, 2, 4)
```

---

### Indexing and Slicing (1.5 minutes)

**"Access and manipulate specific parts of tensors"**

```python
# Create a sample tensor
tensor = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12]])

# Indexing (similar to NumPy)
print(f"Element [0, 0]: {tensor[0, 0]}")        # 1
print(f"First row: {tensor[0]}")                 # [1, 2, 3, 4]
print(f"Last column: {tensor[:, -1]}")          # [4, 8, 12]

# Slicing
print(f"First 2 rows:\n{tensor[:2]}")
print(f"Columns 1-3:\n{tensor[:, 1:3]}")

# Advanced indexing
print(f"Specific elements: {tensor[[0, 2], [1, 3]]}")  # [2, 12]

# Boolean masking (super useful!)
mask = tensor > 6
print(f"Mask:\n{mask}")
print(f"Elements > 6: {tensor[mask]}")  # [7, 8, 9, 10, 11, 12]

# Replace elements based on condition
tensor_copy = tensor.clone()
tensor_copy[tensor_copy < 6] = 0
print(f"Replace < 6 with 0:\n{tensor_copy}")
```

---

### Practical Example: Batch Normalization (1 minute)

**"Let's combine everything for a real neural network operation"**

```python
# Simulate a batch of data from a neural network layer
batch_size = 8
features = 4

# Random activations from a layer
activations = torch.randn(batch_size, features)
print(f"Original activations:\n{activations}")

# Batch normalization (normalize across batch dimension)
mean = activations.mean(dim=0, keepdim=True)  # (1, 4)
std = activations.std(dim=0, keepdim=True)    # (1, 4)

# Normalize (broadcasting at work!)
normalized = (activations - mean) / (std + 1e-5)
print(f"\nNormalized:\n{normalized}")

# Verify: mean should be ~0, std should be ~1
print(f"\nNew mean: {normalized.mean(dim=0)}")
print(f"New std: {normalized.std(dim=0)}")
```

---

## Key Takeaways

1. **Element-wise ops** - Add, multiply, power tensors element by element
2. **Reductions** - Sum, mean, max along dimensions with `dim=`
3. **Broadcasting** - Operate on different shapes automatically
4. **Matrix multiplication** - Use `@` or `torch.matmul()` for neural network layers
5. **Indexing & slicing** - Access specific elements with `[]`, use boolean masks
6. **Batch operations** - Process multiple samples simultaneously

---

## Today's Practice Exercise

```python
# Neural network layer from scratch
import torch

# Create input data (batch of 5 samples, 3 features each)
X = torch.randn(5, 3)

# Create random weights and bias
W = torch.randn(3, 4)  # 3 inputs -> 4 outputs
b = torch.randn(4)

# Forward pass
Z = X @ W + b  # Linear transformation

# Apply ReLU activation
A = torch.relu(Z)

print(f"Input shape: {X.shape}")
print(f"Weight shape: {W.shape}")
print(f"Pre-activation shape: {Z.shape}")
print(f"Output shape: {A.shape}")
print(f"\nOutput:\n{A}")

# Calculate mean activation per neuron
mean_activation = A.mean(dim=0)
print(f"\nMean activation per neuron: {mean_activation}")

# Find which samples have all activations > 0
all_positive = (A > 0).all(dim=1)
print(f"Samples with all positive activations: {all_positive}")
```

---

## Tomorrow's Preview

**Day 3: Autograd and Backpropagation**

- Automatic differentiation with `torch.autograd`
- Computing gradients automatically
- Understanding `.backward()`
- Building a simple training loop

---

**"You're building serious PyTorch skills! See you tomorrow! 🔥"**
