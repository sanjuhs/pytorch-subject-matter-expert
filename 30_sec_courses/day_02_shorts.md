# Day 02 — 30-Second Shorts (Tensor Ops + Broadcasting)

These are **script-ready** shorts. Each short includes:
- **Title**
- **Hook** (1 sentence)
- **Talk track** (what you say, ~20 seconds)
- **Code** (what you show)
- **Common mistake** (quick pitfall)

---

## Short 01 — Day 2 Mission: Shape + Ops + Broadcasting

**Hook**
If you can predict shapes, you can debug 80% of PyTorch.

**Talk track**
- “Today is all about tensor operations you’ll use daily.”
- “We’ll focus on shapes, broadcasting, and matrix multiplication.”
- “Always print shapes when you’re unsure.”

**Code**
```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(x.shape)
```

**Common mistake**
- Guessing shapes instead of printing them.

---

## Short 02 — Elementwise Add/Subtract (Same Shape)

**Hook**
Elementwise ops are the default when shapes match.

**Talk track**
- “Elementwise means position-by-position.”
- “Same indices combine together.”

**Code**
```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)
print(a - b)
```

**Common mistake**
- Expecting mismatched shapes to “just work” without broadcasting rules.

---

## Short 03 — Elementwise Multiply/Divide

**Hook**
`*` is elementwise — not matrix multiplication.

**Talk track**
- “Multiply and divide happen element-by-element.”
- “This is super common for scaling, masking, and normalization.”

**Code**
```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a * b)
print(a / b)
```

**Common mistake**
- Using `*` when you meant matmul (`@`).

---

## Short 04 — Power: Square a Tensor

**Hook**
Squaring is everywhere in losses and norms.

**Talk track**
- “`x ** 2` is a quick way to square every element.”
- “Used in MSE loss, L2 regularization, distances.”

**Code**
```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
print(a ** 2)
```

**Common mistake**
- Confusing `a ** 2` (power) with `a ^ 2` (bitwise XOR, wrong).

---

## Short 05 — Reductions: sum() and mean()

**Hook**
Reductions collapse many values into fewer values.

**Talk track**
- “`sum()` and `mean()` turn a tensor into a smaller tensor, often a scalar.”
- “Loss functions usually end with a reduction.”

**Code**
```python
import torch

t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(t.sum())
print(t.mean())
```

**Common mistake**
- Forgetting whether your loss should be a sum or mean (changes gradient scale).

---

## Short 06 — Reductions: max() and min()

**Hook**
max/min are your fastest sanity checks.

**Talk track**
- “If training explodes, print max/min.”
- “This helps you catch NaNs and huge values quickly.”

**Code**
```python
import torch

t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
print(t.max())
print(t.min())
```

**Common mistake**
- Only checking the mean (NaNs can hide until too late).

---

## Short 07 — dim=0 vs dim=1 (Most Confusing Concept)

**Hook**
dim tells you “which axis you collapse”.

**Talk track**
- “`dim=0` reduces down rows → you get per-column results.”
- “`dim=1` reduces across columns → you get per-row results.”

**Code**
```python
import torch

t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(t.sum(dim=0))  # per column
print(t.sum(dim=1))  # per row
```

**Common mistake**
- Mixing up “rows vs columns” because you don’t print the shape after reducing.

---

## Short 08 — keepdim=True (Broadcast-Friendly)

**Hook**
keepdim=True saves you from shape headaches.

**Talk track**
- “When you reduce, dimensions can disappear.”
- “`keepdim=True` keeps them so broadcasting works later.”

**Code**
```python
import torch

x = torch.randn(8, 4)
mean = x.mean(dim=0, keepdim=True)  # (1, 4)
print(x.shape, mean.shape)
```

**Common mistake**
- Forgetting `keepdim=True`, then subtracting mean and getting a mismatch later.

---

## Short 09 — ReLU in One Line

**Hook**
ReLU is the default activation for hidden layers.

**Talk track**
- “ReLU is `max(0, x)`.”
- “It’s fast and works well in practice.”

**Code**
```python
import torch

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(torch.relu(x))
```

**Common mistake**
- Using ReLU on the final layer when you need negative outputs.

---

## Short 10 — Sigmoid: Map to (0, 1)

**Hook**
Sigmoid is great for binary probabilities.

**Talk track**
- “Sigmoid squashes values into (0,1).”
- “Often used at the output of binary classifiers.”

**Code**
```python
import torch

x = torch.tensor([-2.0, 0.0, 2.0])
print(torch.sigmoid(x))
```

**Common mistake**
- Using sigmoid in deep hidden layers and getting vanishing gradients.

---

## Short 11 — Softmax: Probabilities That Sum to 1

**Hook**
Softmax converts logits into a probability distribution.

**Talk track**
- “Softmax outputs sum to 1.”
- “Use it for multi-class probability outputs.”

**Code**
```python
import torch

logits = torch.tensor([2.0, 1.0, 0.1])
probs = torch.softmax(logits, dim=0)
print(probs, probs.sum())
```

**Common mistake**
- Wrong dimension: for `(batch, classes)` use `dim=1`.

---

## Short 12 — Broadcasting Rule #1: Scalar to Tensor

**Hook**
Scalars broadcast to any tensor shape.

**Talk track**
- “Add 10 to every element with one operation.”
- “This is the simplest form of broadcasting.”

**Code**
```python
import torch

t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(t + 10)
```

**Common mistake**
- Forgetting broadcasting exists and writing slow Python loops.

---

## Short 13 — Broadcasting Rule #2: Vector Across Rows

**Hook**
Add a (3,) vector to a (2,3) matrix in one line.

**Talk track**
- “The vector expands across the missing dimension.”
- “It gets added to every row.”

**Code**
```python
import torch

matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
vector = torch.tensor([10, 20, 30])

print(matrix + vector)
```

**Common mistake**
- Confusing whether the vector applies per-row or per-column — check shapes.

---

## Short 14 — Broadcasting Rule #3: Column Vector

**Hook**
Shape (2,1) broadcasts across columns.

**Talk track**
- “A column vector adds a different value per row.”
- “It stretches across columns automatically.”

**Code**
```python
import torch

matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
column = torch.tensor([[10],
                       [20]])

print(matrix + column)
```

**Common mistake**
- Accidentally making a row vector when you meant a column vector.

---

## Short 15 — Broadcasting Cheat Sheet (Compare From the End)

**Hook**
Broadcasting is easiest if you compare shapes from the right.

**Talk track**
- “Two dims are compatible if they’re equal or one is 1.”
- “Missing dims act like 1.”

**Code**
```python
import torch

a = torch.randn(3, 1, 5)
b = torch.randn(1, 4, 5)
c = a + b
print(c.shape)
```

**Common mistake**
- Trying to broadcast incompatible middle dimensions (like (3,2,5) + (1,4,5)).

---

## Short 16 — Bias Add is Broadcasting (Real NN Example)

**Hook**
Bias is broadcasting in disguise.

**Talk track**
- “Activations are `(batch, features)`.”
- “Bias is `(features,)` and broadcasts across the batch.”

**Code**
```python
import torch

batch_size, features = 32, 10
activations = torch.randn(batch_size, features)
bias = torch.randn(features)

out = activations + bias
print(out.shape)
```

**Common mistake**
- Bias shaped `(features, 1)` causing unintended broadcasting patterns.

---

## Short 17 — Matrix Multiply: Use @ (Not *)

**Hook**
`@` is the heart of neural networks.

**Talk track**
- “Matrix multiplication is how layers combine features.”
- “Rule: inner dims must match.”

**Code**
```python
import torch

A = torch.tensor([[1, 2],
                  [3, 4],
                  [5, 6]])        # (3, 2)
B = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])  # (2, 3)

C = A @ B
print(C.shape)
```

**Common mistake**
- Using `*` and getting elementwise behavior instead of matmul.

---

## Short 18 — Matmul Shape Rule: (m,n) @ (n,p) = (m,p)

**Hook**
Predict shapes before you run code.

**Talk track**
- “If `A` is `(m,n)` and `B` is `(n,p)` result is `(m,p)`.”
- “This prevents 90% of matmul bugs.”

**Code**
```python
import torch

m, n, p = 3, 2, 4
A = torch.randn(m, n)
B = torch.randn(n, p)
print((A @ B).shape)
```

**Common mistake**
- Swapping dims and getting: “mat1 and mat2 shapes cannot be multiplied”.

---

## Short 19 — Simulate a Linear Layer: Y = X @ W + b

**Hook**
This is literally what `nn.Linear` computes (conceptually).

**Talk track**
- “Batch input `X` times weights `W` plus bias `b`.”
- “Bias uses broadcasting.”

**Code**
```python
import torch

batch_size, input_features, output_features = 4, 3, 5
X = torch.randn(batch_size, input_features)          # (4, 3)
W = torch.randn(input_features, output_features)     # (3, 5)
b = torch.randn(output_features)                     # (5,)

Y = X @ W + b
print(Y.shape)
```

**Common mistake**
- Getting weight shape backwards and fighting dimension errors.

---

## Short 20 — Batch Matmul: torch.bmm()

**Hook**
Do many matrix multiplications in parallel.

**Talk track**
- “`bmm` is for 3D tensors: `(batch, m, n)` and `(batch, n, p)`.”
- “Great for attention-like patterns and batched ops.”

**Code**
```python
import torch

batch = 3
A = torch.randn(batch, 2, 3)
B = torch.randn(batch, 3, 4)
C = torch.bmm(A, B)
print(C.shape)
```

**Common mistake**
- Passing 2D tensors into `bmm` (it requires 3D).

---

## Short 21 — Index a Single Element

**Hook**
PyTorch indexing feels like NumPy — because it is.

**Talk track**
- “Use `tensor[row, col]` to grab one value.”
- “This is basic but crucial for debugging.”

**Code**
```python
import torch

t = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
print(t[0, 0])
```

**Common mistake**
- Forgetting indices are 0-based.

---

## Short 22 — Slice Rows and Columns

**Hook**
Slicing is the fastest way to inspect a tensor.

**Talk track**
- “`:` means ‘all’ along that dimension.”
- “Use slices to pull submatrices.”

**Code**
```python
import torch

t = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

print(t[:2])        # first 2 rows
print(t[:, 1:3])    # columns 1..2
```

**Common mistake**
- Thinking the end index is inclusive (it’s exclusive).

---

## Short 23 — Grab the Last Column Fast

**Hook**
Negative indices are a cheat code.

**Talk track**
- “`-1` always means ‘last element’.”
- “This is super handy for last token / last feature patterns later.”

**Code**
```python
import torch

t = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
print(t[:, -1])
```

**Common mistake**
- Confusing `t[-1]` (last row) with `t[:, -1]` (last column).

---

## Short 24 — Fancy Indexing: Pick Specific Elements

**Hook**
Pick multiple scattered elements in one line.

**Talk track**
- “You can index with lists of positions.”
- “This is useful for sampling and selecting specific entries.”

**Code**
```python
import torch

t = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

print(t[[0, 2], [1, 3]])  # tensor([ 2, 12])
```

**Common mistake**
- Expecting this to return a 2D block (it returns paired coordinates).

---

## Short 25 — Boolean Masking: Filter Values

**Hook**
Masks are everywhere in deep learning.

**Talk track**
- “A boolean mask selects elements that match a condition.”
- “This is used for thresholds, padding masks, and filtering.”

**Code**
```python
import torch

t = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

mask = t > 6
print(t[mask])
```

**Common mistake**
- Mask shape must match the tensor shape you’re indexing.

---

## Short 26 — Replace Values Using a Mask (Safely)

**Hook**
Clone first to avoid accidental mutation.

**Talk track**
- “If you need to modify based on a condition, clone first.”
- “Then assign into the clone with a boolean condition.”

**Code**
```python
import torch

t = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

t2 = t.clone()
t2[t2 < 6] = 0
print(t2)
```

**Common mistake**
- Forgetting `.clone()` and overwriting the original tensor you still needed.

---

## Short 27 — BatchNorm Core: Mean/Std Per Feature

**Hook**
BatchNorm is just normalization across the batch.

**Talk track**
- “We compute mean and std per feature dimension.”
- “Then normalize with broadcasting.”

**Code**
```python
import torch

activations = torch.randn(8, 4)  # (batch, features)
mean = activations.mean(dim=0, keepdim=True)
std = activations.std(dim=0, keepdim=True)
print(mean.shape, std.shape)
```

**Common mistake**
- Reducing over the wrong dimension and normalizing across features instead of batch.

---

## Short 28 — BatchNorm Formula (Broadcasting in Action)

**Hook**
Broadcasting makes normalization one line.

**Talk track**
- “Normalize: `(x - mean) / (std + eps)`.”
- “Epsilon prevents divide-by-zero.”

**Code**
```python
import torch

activations = torch.randn(8, 4)
mean = activations.mean(dim=0, keepdim=True)
std = activations.std(dim=0, keepdim=True)

normalized = (activations - mean) / (std + 1e-5)
print(normalized.mean(dim=0))
print(normalized.std(dim=0))
```

**Common mistake**
- Forgetting epsilon `1e-5` and getting numerical instability.

---

## Short 29 — Mini Exercise: One Layer + ReLU + Stats

**Hook**
Build a tiny forward pass like a real model.

**Talk track**
- “Linear layer math: `Z = X @ W + b`.”
- “Activation: `ReLU(Z)`.”
- “Then compute some stats per neuron.”

**Code**
```python
import torch

X = torch.randn(5, 3)
W = torch.randn(3, 4)
b = torch.randn(4)

Z = X @ W + b
A = torch.relu(Z)
print(A.shape)
print(A.mean(dim=0))
```

**Common mistake**
- Forgetting bias broadcasting and trying to manually expand shapes.

---

## Short 30 — The Debug Habit: Print Shapes at Every Step

**Hook**
Shape checks are the fastest debugging tool.

**Talk track**
- “In PyTorch, most runtime errors are shape errors.”
- “Print shapes after each major step.”

**Code**
```python
import torch

X = torch.randn(4, 3)
W = torch.randn(3, 5)
b = torch.randn(5)

print("X", X.shape)
print("W", W.shape)
print("b", b.shape)
print("Y", (X @ W + b).shape)
```

**Common mistake**
- Waiting until the end to check shapes (debugging becomes painful).


