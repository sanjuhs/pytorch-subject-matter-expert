# Day 01 — 30-Second Shorts (PyTorch + Essential Python)

These are **script-ready** shorts. Each short includes:

- **Title**
- **Hook** (1 sentence)
- **Talk track** (what you say, ~20 seconds)
- **Code** (what you show)
- **Common mistake** (quick pitfall)

---

## Short 01 — The Only Rule: Everything Is a Tensor

**Hook**
If you understand tensors, you understand PyTorch.

**Talk track**

- “PyTorch is basically: tensors + operations + autograd.”
- “Every model is just tensor math under the hood.”
- “Today we’ll do essential Python patterns + tensor basics.”

**Code**

```python
import torch

x = torch.tensor([1, 2, 3])
print(x, type(x))
```

**Common mistake**

- Confusing Python lists with tensors: lists don’t have `.shape`, `.device`, or GPU support.

---

## Short 02 — Fast Colab Setup for PyTorch

**Hook**
One line to confirm PyTorch is ready.

**Talk track**

- “Open Colab, new notebook, name it `Day_01_Tensors`.”
- “First thing: import torch and print the version.”

**Code**

```python
import torch
print(torch.__version__)
```

**Common mistake**

- Installing random CUDA wheels manually in Colab before even trying `import torch`.

---

## Short 03 — Python Lists: Your Go-To Container

**Hook**
Lists are everywhere in ML code — get comfortable fast.

**Talk track**

- “Lists store multiple values: examples, configs, layer stacks.”
- “Indexing is the same skill you’ll use for batches and outputs.”

**Code**

```python
numbers = [1, 2, 3, 4, 5]
print(numbers[0], numbers[-1])
```

**Common mistake**

- Off-by-one indexing: Python is 0-based, and `-1` means the last element.

---

## Short 04 — List Comprehensions = Cleaner Training Code

**Hook**
One-liners that replace noisy loops.

**Talk track**

- “Comprehensions are common in data prep and quick transforms.”
- “They’re readable if you keep them simple.”

**Code**

```python
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print(squares)
```

**Common mistake**

- Over-nesting comprehensions until nobody can debug them.

---

## Short 05 — Tuples + Unpacking (x, y) = (3, 4)

**Hook**
Unpacking is a power move you’ll use daily.

**Talk track**

- “Tuples are fixed collections (often used for shapes).”
- “Unpacking is the clean way to split values: `x, y = point`.”

**Code**

```python
point = (3, 4)
x, y = point
print(x, y)
```

**Common mistake**

- Unpacking the wrong number of values (tuple length must match variables).

---

## Short 06 — Dicts for Config (Like Real Training Scripts)

**Hook**
Every serious training script starts with a config.

**Talk track**

- “Dictionaries store hyperparameters and settings.”
- “You’ll see `lr`, `batch_size`, `epochs` everywhere.”

**Code**

```python
config = {"learning_rate": 1e-3, "batch_size": 32, "epochs": 10}
print(config["learning_rate"])
```

**Common mistake**

- Inconsistent key names (`"learning_rate"` vs `"lr"`) causing KeyErrors or silent bugs.

---

## Short 07 — zip(): Pair Things Up (Karpathy Style)

**Hook**
`zip()` is the cleanest way to loop over paired data.

**Talk track**

- “Use `zip()` to pair names/scores or inputs/labels.”
- “It keeps your loop code tidy.”

**Code**

```python
names = ["Alice", "Bob", "Charlie"]
scores = [95, 87, 92]

for name, score in zip(names, scores):
    print(name, score)
```

**Common mistake**

- `zip()` stops at the shortest list, which can silently drop items.

---

## Short 08 — dict(zip()) = Instant Lookup Table

**Hook**
Turn two lists into a dict in one line.

**Talk track**

- “This is perfect for quick mappings: id → label, name → score.”

**Code**

```python
names = ["Alice", "Bob", "Charlie"]
scores = [95, 87, 92]

score_dict = dict(zip(names, scores))
print(score_dict["Bob"])
```

**Common mistake**

- Duplicate keys overwrite earlier values (only the last one survives).

---

## Short 09 — enumerate(): Track Indices Without Manual Counters

**Hook**
Stop writing `i = 0; i += 1`.

**Talk track**

- “`enumerate()` gives you index + value cleanly.”
- “Super common for steps, batches, and logging.”

**Code**

```python
fruits = ["apple", "banana", "cherry"]
for idx, fruit in enumerate(fruits):
    print(idx, fruit)
```

**Common mistake**

- Forgetting you can set a start index: `enumerate(items, start=1)`.

---

## Short 10 — The \* Operator: “Spread” Lists

**Hook**
This is how you build lists like a pro.

**Talk track**

- “`*` unpacks a list into another list.”
- “Great for composing configs or building sequences.”

**Code**

```python
numbers = [1, 2, 3]
more_numbers = [*numbers, 4, 5, 6]
print(more_numbers)
```

**Common mistake**

- Expecting `*numbers` to deep-copy nested lists (it doesn’t).

---

## Short 11 — \* for Function Arguments (Super Useful Pattern)

**Hook**
Turn a list into function args instantly.

**Talk track**

- “If a function expects separate args, you can unpack a list with `*`.”
- “This shows up in lots of utility code.”

**Code**

```python
def compute_stats(a, b, c):
    return a + b + c

values = [10, 20, 30]
print(compute_stats(*values))
```

**Common mistake**

- Length mismatch: the list must match the function’s required arguments (unless you use `*args`).

---

## Short 12 — \*\* for Dict Merges (and Overrides)

**Hook**
Merge configs cleanly — and override on purpose.

**Talk track**

- “`{**a, **b}` merges dictionaries.”
- “If keys collide, later values overwrite earlier ones.”

**Code**

```python
config = {"lr": 1e-3, "batch_size": 32}
extra = {**config, "epochs": 10, "lr": 1e-4}
print(extra)
```

**Common mistake**

- Overriding `lr` accidentally — always print the final config you train with.

---

## Short 13 — Lambda in 5 Seconds

**Hook**
Tiny functions for quick transforms.

**Talk track**

- “A lambda is an anonymous one-line function.”
- “Use it when it improves clarity — don’t abuse it.”

**Code**

```python
square = lambda x: x**2
print(square(5))
```

**Common mistake**

- Writing complex lambdas that nobody can read; use `def` instead.

---

## Short 14 — map(): Apply a Function to Every Element

**Hook**
One function, many items.

**Talk track**

- “`map()` applies a function across an iterable.”
- “In Python 3 it returns an iterator, so wrap with `list()` if you want to print.”

**Code**

```python
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)
```

**Common mistake**

- Forgetting `list(...)` and being confused by the `map(...)` object.

---

## Short 15 — map() + Tensors: Print Shapes Fast

**Hook**
Quick debugging trick: map tensors → shapes.

**Talk track**

- “When you have a list of tensors, print their shapes in one line.”
- “This catches shape bugs early.”

**Code**

```python
import torch

tensors = [torch.randn(2, 2) for _ in range(3)]
shapes = list(map(lambda t: t.shape, tensors))
print(shapes)
```

**Common mistake**

- Printing full tensors when you only need shapes (it’s noisy and slow).

---

## Short 16 — @property: Looks Like a Field, Runs Like Code

**Hook**
Clean APIs in model code often use `@property`.

**Talk track**

- “`@property` lets you access a computed value like an attribute.”
- “You’ll see it in clean ML codebases.”

**Code**

```python
class NeuralNetwork:
    def __init__(self, input_size):
        self._input_size = input_size
        self._hidden_size = input_size * 2

    @property
    def hidden_size(self):
        return self._hidden_size

net = NeuralNetwork(128)
print(net.hidden_size)
```

**Common mistake**

- Calling it like a method: `net.hidden_size()` (don’t).

---

## Short 17 — What Is a Tensor (In One Line)?

**Hook**
Tensor = multi-dimensional array with GPU + autograd support.

**Talk track**

- “Like NumPy arrays, but built for deep learning.”
- “They can live on CPU or GPU and track gradients.”

**Code**

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x.shape)
```

**Common mistake**

- Assuming tensors are always float — dtype matters for math and models.

---

## Short 18 — Create a Tensor from a Python List

**Hook**
The simplest tensor constructor.

**Talk track**

- “`torch.tensor([...])` converts a Python list into a tensor.”
- “Great for quick examples and small constants.”

**Code**

```python
import torch

tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(tensor_1d)
```

**Common mistake**

- Mixing types in the list can produce unexpected dtype or errors.

---

## Short 19 — 2D Tensors + .shape (Your Daily Habit)

**Hook**
If you’re stuck, print `.shape`.

**Talk track**

- “2D tensors represent matrices.”
- “Neural nets are basically a bunch of matrix operations.”

**Code**

```python
import torch

tensor_2d = torch.tensor([[1, 2, 3],
                          [4, 5, 6]])
print(tensor_2d)
print(tensor_2d.shape)
```

**Common mistake**

- Assuming the first dim is always features; usually it’s batch.

---

## Short 20 — torch.zeros() and torch.ones()

**Hook**
Masks and placeholders start here.

**Talk track**

- “Zeros and ones are used for initialization, masks, and debugging.”
- “Always verify shape.”

**Code**

```python
import torch

zeros = torch.zeros(3, 3)
ones = torch.ones(2, 4)
print(zeros.shape, ones.shape)
```

**Common mistake**

- Creating tensors on CPU but expecting them on GPU; later you’ll want `.to(device)`.

---

## Short 21 — torch.randn(): Random Tensors for Neural Nets

**Hook**
Random init is fundamental to training.

**Talk track**

- “`randn` samples from a normal distribution.”
- “It’s used constantly for initialization and synthetic examples.”

**Code**

```python
import torch

random_tensor = torch.randn(3, 3)
print(random_tensor)
```

**Common mistake**

- Comparing runs without setting a seed and thinking results are “buggy”.

---

## Short 22 — torch.arange(): Ranges for Testing + Reshaping

**Hook**
The fastest way to create a predictable tensor.

**Talk track**

- “`arange` is great for debugging indexing and reshape.”
- “You get predictable values instead of random noise.”

**Code**

```python
import torch

range_tensor = torch.arange(0, 10, 2)
print(range_tensor)
```

**Common mistake**

- Forgetting the step argument and creating a much larger tensor than intended.

---

## Short 23 — Elementwise Tensor Math (Same Shape)

**Hook**
This is the baseline operation type in PyTorch.

**Talk track**

- “If tensors are same shape, ops are elementwise: add, multiply, etc.”
- “Later, broadcasting extends this idea.”

**Code**

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(a + b)
print(a * b)
```

**Common mistake**

- Expecting `*` to do matrix multiplication (that’s `@`).

---

## Short 24 — Dot Product in One Line

**Hook**
Dot product turns two vectors into one number.

**Talk track**

- “`torch.dot(a, b)` is for 1D vectors.”
- “It’s a building block for similarity and linear models.”

**Code**

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(torch.dot(a, b))
```

**Common mistake**

- Using `torch.dot` on 2D tensors (use `@` / `matmul` for matrices).

---

## Short 25 — reshape(): Turn Flat Into (Rows, Cols)

**Hook**
Reshaping is everywhere in deep learning.

**Talk track**

- “Reshape changes the view of the same data.”
- “Total number of elements must stay the same.”

**Code**

```python
import torch

x = torch.arange(12)
reshaped = x.reshape(3, 4)
print(reshaped)
```

**Common mistake**

- Wrong total size: reshape must keep `numel()` constant.

---

## Short 26 — Tensor Vitals: shape, dtype, device, ndim

**Hook**
Print these 4 and you’ll solve most bugs.

**Talk track**

- “Shape: what it looks like.”
- “dtype: what it’s made of.”
- “device: where it lives.”
- “ndim: how many dimensions.”

**Code**

```python
import torch

t = torch.randn(3, 4, 5)
print(t.shape)
print(t.dtype)
print(t.device)
print(t.ndim)
```

**Common mistake**

- dtype mismatch (float vs long) causing errors in losses or embeddings later.

---

## Short 27 — Mini Batch Example (Like Training)

**Hook**
This is what “batch data” really looks like.

**Talk track**

- “We simulate a batch: `(batch_size, feature_dim)`.”
- “Loop with `enumerate(zip(data, labels))` like real training code.”

**Code**

```python
import torch

batch_size = 4
feature_dim = 3

data = torch.randn(batch_size, feature_dim)
labels = torch.tensor([0, 1, 0, 1])

for idx, (features, label) in enumerate(zip(data, labels)):
    print(idx, features.shape, int(label))
```

**Common mistake**

- Iterating a 2D tensor yields rows (samples), not individual scalars.

---

## Short 28 — GPU Check + Move Tensor to GPU

**Hook**
The safest GPU pattern in PyTorch.

**Talk track**

- “Pick device: cuda if available else cpu.”
- “Move tensors with `.to(device)`.”

**Code**

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

t = torch.randn(3, 3).to(device)
print(t.device)
```

**Common mistake**

- Mixing CPU and GPU tensors in one operation → device mismatch runtime error.

---

## Short 29 — Tiny Exercise: BMI with Tensors + zip()

**Hook**
Use tensors for a real calculation in 20 seconds.

**Talk track**

- “We’ll compute BMI for 4 people using tensors.”
- “This mixes Python + tensors the way ML scripts do.”

**Code**

```python
import torch

heights = torch.tensor([170, 165, 180, 175])  # cm
weights = torch.tensor([70, 65, 80, 75])      # kg
names = ["Alice", "Bob", "Charlie", "Diana"]

for name, h, w in zip(names, heights, weights):
    h_m = h / 100
    bmi = w / (h_m ** 2)
    print(f"{name}: BMI = {bmi:.2f}")
```

**Common mistake**

- Integer vs float surprises: if needed, cast `heights = heights.float()` / `weights = weights.float()`.

---

## Short 30 — Preview: Day 2 = Broadcasting + Indexing + Matmul

**Hook**
Tomorrow tensors start feeling powerful.

**Talk track**

- “Day 2: operations, broadcasting, indexing, matrix multiplication.”
- “That’s the core of how neural nets compute.”

**Code**

```python
# Tomorrow: broadcasting + indexing + matrix multiplication
```

**Common mistake**

- Skipping Day 2: broadcasting + shapes are where most real bugs come from.
