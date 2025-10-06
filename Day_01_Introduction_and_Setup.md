# Day 1: Welcome to PyTorch - Tensors and Essential Python

**Duration**: 10 minutes
**Platform**: Google Colab

---

## Video Script

### Introduction (1 minute)

**"Hello and welcome to Day 1 of our PyTorch journey!"**

Today marks the beginning of an exciting 100-day adventure where we'll master PyTorch, build neural networks, create large language models, and even write custom CUDA kernels.

We'll start with the fundamental building block of PyTorch: **tensors**, and learn essential Python patterns used in deep learning courses by experts like Andrej Karpathy.

---

### Quick Colab Setup (30 seconds)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Create new notebook → rename to "Day_01_Tensors"
3. Let's dive in!

---

### Essential Python Patterns from NanoGPT (3.5 minutes)

**"Let's cover the Python structures you'll see everywhere in deep learning"**

**Lists - Your go-to container**

```python
# Lists hold multiple values
numbers = [1, 2, 3, 4, 5]
names = ["Alice", "Bob", "Charlie"]

# List comprehensions - powerful one-liners
squares = [x**2 for x in numbers]
print(squares)  # [1, 4, 9, 16, 25]
```

**Tuples - Immutable pairs**

```python
# Tuples for fixed collections
point = (3, 4)
x, y = point  # Unpacking
print(f"x: {x}, y: {y}")
```

**Dictionaries - Key-value storage**

```python
# Dicts for configuration and parameters
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10
}
print(config['learning_rate'])
```

**The Mighty zip() - Karpathy's favorite**

```python
# zip combines multiple iterables
names = ["Alice", "Bob", "Charlie"]
scores = [95, 87, 92]

# Pair them up elegantly
for name, score in zip(names, scores):
    print(f"{name}: {score}")

# Create dict from two lists
score_dict = dict(zip(names, scores))
print(score_dict)
```

**enumerate() - Track indices**

```python
# Get both index and value
fruits = ["apple", "banana", "cherry"]

for idx, fruit in enumerate(fruits):
    print(f"Position {idx}: {fruit}")
```

**Unpacking with * (spread operator) - NanoGPT style**

```python
# Unpacking lists/tuples with *
numbers = [1, 2, 3]
more_numbers = [*numbers, 4, 5, 6]  # [1, 2, 3, 4, 5, 6]
print(more_numbers)

# Unpacking in function calls
def compute_stats(a, b, c):
    return a + b + c

values = [10, 20, 30]
result = compute_stats(*values)  # Unpacks list as arguments
print(f"Sum: {result}")

# Unpacking dictionaries with **
config = {'lr': 0.001, 'batch_size': 32}
extra_config = {**config, 'epochs': 10, 'lr': 0.0001}  # Override lr
print(extra_config)
```

**Lambda functions and map() - Quick transforms**

```python
# Lambda: inline anonymous functions
square = lambda x: x**2
print(square(5))  # 25

# map: apply function to all elements
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# Often used with tensor operations
import torch
tensors = [torch.randn(2, 2) for _ in range(3)]
shapes = list(map(lambda t: t.shape, tensors))
print(shapes)
```

**@property decorator - Clean class attributes**

```python
# Used in model classes (you'll see this in nn.Module)
class NeuralNetwork:
    def __init__(self, input_size):
        self._input_size = input_size
        self._hidden_size = input_size * 2

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._input_size // 2

# Use like attributes, not methods
net = NeuralNetwork(128)
print(f"Hidden: {net.hidden_size}")  # No parentheses!
print(f"Output: {net.output_size}")
```

---

### Introduction to Tensors (3.5 minutes)

**"Now the star of the show: TENSORS!"**

**What are tensors?**
- Tensors are multi-dimensional arrays (like NumPy arrays, but better!)
- They can run on GPUs for massive speedup
- They're the fundamental data structure in PyTorch

**Creating tensors**

```python
import torch

# From a Python list
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(f"1D Tensor: {tensor_1d}")

# 2D tensor (matrix)
tensor_2d = torch.tensor([[1, 2, 3],
                          [4, 5, 6]])
print(f"2D Tensor:\n{tensor_2d}")

# Check the shape
print(f"Shape: {tensor_2d.shape}")  # torch.Size([2, 3])
```

**Common tensor creation methods**

```python
# Zeros and ones
zeros = torch.zeros(3, 3)
ones = torch.ones(2, 4)

# Random tensors (important for neural networks!)
random_tensor = torch.randn(3, 3)  # Normal distribution
print(f"Random tensor:\n{random_tensor}")

# Range of values
range_tensor = torch.arange(0, 10, 2)  # 0 to 10, step 2
print(f"Range: {range_tensor}")
```

**Tensor operations**

```python
# Basic math
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(f"Addition: {a + b}")
print(f"Multiplication: {a * b}")  # Element-wise
print(f"Dot product: {torch.dot(a, b)}")

# Reshaping - crucial for neural networks
x = torch.arange(12)
reshaped = x.reshape(3, 4)
print(f"Reshaped:\n{reshaped}")
```

**Tensor attributes you'll use constantly**

```python
tensor = torch.randn(3, 4, 5)

print(f"Shape: {tensor.shape}")
print(f"Data type: {tensor.dtype}")
print(f"Device: {tensor.device}")  # cpu or cuda
print(f"Number of dimensions: {tensor.ndim}")
```

---

### Practical Example: Combining Everything (1 minute)

**"Let's use what we learned together"**

```python
# Simulating batch of data (like in real training)
batch_size = 4
feature_dim = 3

# Create random batch of data
data = torch.randn(batch_size, feature_dim)
labels = torch.tensor([0, 1, 0, 1])

# Use zip and enumerate like Karpathy does
for idx, (features, label) in enumerate(zip(data, labels)):
    print(f"Sample {idx}: features shape {features.shape}, label {label}")
```

---

### GPU Check (30 seconds)

```python
# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Move tensor to GPU (if available)
tensor = torch.randn(3, 3)
tensor = tensor.to(device)
print(f"Tensor is on: {tensor.device}")
```

---

## Key Takeaways

1. **Lists, tuples, dicts** - Core Python data structures
2. **zip()** and **enumerate()** - Combine and track iterables
3. **Unpacking with * and *** - Spread operators for lists/dicts
4. **Lambda and map()** - Quick inline transformations
5. **@property** - Clean class attribute access
6. **Tensors** - PyTorch's multi-dimensional arrays
7. **torch.randn()** - Create random tensors
8. **.shape, .dtype, .device** - Essential tensor attributes

---

## Today's Practice Exercise

```python
# Create a mini dataset
heights = torch.tensor([170, 165, 180, 175])  # cm
weights = torch.tensor([70, 65, 80, 75])      # kg
names = ["Alice", "Bob", "Charlie", "Diana"]

# Calculate BMI and print with zip
for name, h, w in zip(names, heights, weights):
    h_m = h / 100  # convert to meters
    bmi = w / (h_m ** 2)
    print(f"{name}: BMI = {bmi:.2f}")

# Try creating a random tensor and check its properties
random_data = torch.randn(5, 10)
print(f"Shape: {random_data.shape}, dtype: {random_data.dtype}")
```

---

## Tomorrow's Preview

**Day 2: Tensor Operations and Broadcasting**

- Advanced tensor operations
- Broadcasting rules
- Indexing and slicing tensors
- Matrix multiplication for neural networks

---

**See you tomorrow! Keep coding! 🚀**
