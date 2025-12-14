# Day 04 — 30-Second Shorts (nn.Module + “PyTorch Way”)

These are **script-ready** shorts. Each short includes:
- **Title**
- **Hook** (1 sentence)
- **Talk track** (what you say, ~20 seconds)
- **Code** (what you show)
- **Common mistake** (quick pitfall)

---

## Short 01 — Day 4: Stop Managing Parameters Manually

**Hook**
Today you graduate from “raw tensors” to real PyTorch models.

**Talk track**
- “Yesterday we trained with raw tensors: W, b, manual updates.”
- “Today we use `nn.Module`, which tracks parameters automatically.”
- “This is the foundation of every PyTorch model you’ll see in the wild.”

**Code**
```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.layer2(nn.functional.relu(self.layer1(x)))
```

**Common mistake**
- Forgetting `super().__init__()` and then parameters won’t register correctly.

---

## Short 02 — What Is nn.Module (In One Sentence)?

**Hook**
`nn.Module` is the base class for every neural network component.

**Talk track**
- “A Module is a container that can hold layers, parameters, and submodules.”
- “PyTorch automatically finds parameters inside it.”

**Code**
```python
import torch.nn as nn

print(isinstance(nn.Linear(3, 4), nn.Module))
```

**Common mistake**
- Storing tensors as plain attributes instead of `nn.Parameter` / layers, so they don’t train.

---

## Short 03 — The Biggest Benefit: model.parameters()

**Hook**
This is how optimizers know what to update.

**Talk track**
- “Optimizers take `model.parameters()`.”
- “That’s the list of learnable tensors.”

**Code**
```python
import torch.nn as nn

model = nn.Sequential(nn.Linear(4, 2))
params = list(model.parameters())
print(len(params), params[0].shape)
```

**Common mistake**
- Passing an empty parameter list to the optimizer because you didn’t register layers properly.

---

## Short 04 — Your First Custom Model Class

**Hook**
Custom `nn.Module` = clean, reusable code.

**Talk track**
- “Define layers in `__init__`.”
- “Define data flow in `forward`.”
- “Then you call `model(x)` like a function.”

**Code**
```python
import torch
import torch.nn as nn

class MyFirstNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

**Common mistake**
- Creating layers inside `forward()` (they won’t be registered correctly and you’ll re-init every call).

---

## Short 05 — Calling model(x) Automatically Runs forward(x)

**Hook**
You rarely call `forward()` directly.

**Talk track**
- “In PyTorch, `model(x)` calls `forward(x)` under the hood.”
- “That also integrates with hooks, scripting, and other tooling.”

**Code**
```python
import torch

model = MyFirstNetwork()
x = torch.randn(4, 784)
out = model(x)
print(out.shape)
```

**Common mistake**
- Confusing logits with probabilities; raw outputs from `Linear` are logits.

---

## Short 06 — Print Your Model Like a Pro

**Hook**
Printing the model is instant architecture documentation.

**Talk track**
- “`print(model)` shows layers and shapes intent.”
- “Do this before training to catch mistakes early.”

**Code**
```python
model = MyFirstNetwork()
print(model)
```

**Common mistake**
- Training without checking the model definition first.

---

## Short 07 — named_parameters(): See What Learns

**Hook**
Know exactly what tensors are trainable.

**Talk track**
- “`named_parameters()` lets you inspect parameter names and shapes.”
- “Great for debugging and sanity checks.”

**Code**
```python
model = MyFirstNetwork()
for name, p in model.named_parameters():
    print(name, tuple(p.shape), p.requires_grad)
```

**Common mistake**
- Accidentally freezing parameters (`requires_grad=False`) and wondering why nothing improves.

---

## Short 08 — nn.Linear: What It Actually Computes

**Hook**
`nn.Linear` is a matrix multiply plus bias.

**Talk track**
- “Conceptually: `y = x @ W + b`.”
- “In PyTorch, `Linear` manages its own weight and bias tensors.”

**Code**
```python
import torch
import torch.nn as nn

linear = nn.Linear(4, 2)
x = torch.randn(3, 4)
y = linear(x)
print(y.shape)
```

**Common mistake**
- Feeding wrong input feature size (shape mismatch error).

---

## Short 09 — nn.Linear Weight Shape Gotcha (It’s (out, in))

**Hook**
The weight matrix is stored transposed vs the math you write.

**Talk track**
- “`linear.weight.shape` is `(out_features, in_features)`.”
- “So internally it uses `x @ W.T + b`.”

**Code**
```python
linear = nn.Linear(4, 2)
print(linear.weight.shape)  # (2, 4)
print(linear.bias.shape)    # (2,)
```

**Common mistake**
- Trying to manually copy weights without transposing and getting mismatched outputs.

---

## Short 10 — Verify Linear Math: x @ W.T + b

**Hook**
Let’s prove nn.Linear is just matmul + bias.

**Talk track**
- “We’ll compute the same output manually.”
- “This builds intuition for debugging.”

**Code**
```python
import torch
import torch.nn as nn

torch.manual_seed(0)
linear = nn.Linear(4, 2)
x = torch.randn(3, 4)

with torch.no_grad():
    y1 = linear(x)
    y2 = x @ linear.weight.T + linear.bias
print(torch.allclose(y1, y2))
```

**Common mistake**
- Forgetting `.T` and wondering why the matmul fails.

---

## Short 11 — Logits vs Probabilities (Classification Core)

**Hook**
Your model outputs logits — don’t softmax too early.

**Talk track**
- “For multi-class classification, return logits.”
- “Use `CrossEntropyLoss` on logits directly.”

**Code**
```python
import torch
import torch.nn as nn

logits = torch.randn(4, 10)
targets = torch.tensor([1, 2, 3, 0])
loss = nn.CrossEntropyLoss()(logits, targets)
print(loss.item())
```

**Common mistake**
- Applying `softmax` before `CrossEntropyLoss` (hurts numerical stability).

---

## Short 12 — Flatten Inputs: (batch, 28, 28) → (batch, 784)

**Hook**
Most MLPs need flattened images.

**Talk track**
- “Linear layers expect shape `(batch, features)`.”
- “So we flatten images before passing them to `Linear`.”

**Code**
```python
import torch

images = torch.randn(32, 28, 28)
flat = images.view(images.size(0), -1)
print(flat.shape)
```

**Common mistake**
- Using `.view` on a non-contiguous tensor; if it errors, use `.reshape(...)`.

---

## Short 13 — Build a Model Dynamically (Loop Hidden Layers)

**Hook**
This is how you make architectures configurable.

**Talk track**
- “We can build layers from a list of hidden sizes.”
- “Then wrap the whole thing in `nn.Sequential`.”

**Code**
```python
import torch.nn as nn

hidden_sizes = [256, 128]
layers = []
prev = 784
for h in hidden_sizes:
    layers += [nn.Linear(prev, h), nn.ReLU()]
    prev = h
layers += [nn.Linear(prev, 10)]

net = nn.Sequential(*layers)
print(net)
```

**Common mistake**
- Forgetting to update `prev`, causing shape mismatch at the next layer.

---

## Short 14 — nn.Sequential: Fast Prototyping

**Hook**
For simple feed-forward models, Sequential is perfect.

**Talk track**
- “`nn.Sequential` runs layers in order.”
- “Great for MLPs and simple stacks.”

**Code**
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
print(model)
```

**Common mistake**
- Trying to use `Sequential` for complex graphs (skip connections, multiple inputs/outputs).

---

## Short 15 — Named Sequential (OrderedDict)

**Hook**
Names help debugging and readability.

**Talk track**
- “You can name layers inside Sequential.”
- “Then access them as attributes.”

**Code**
```python
import torch.nn as nn
from collections import OrderedDict

model = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(784, 256)),
    ("relu1", nn.ReLU()),
    ("fc2", nn.Linear(256, 10)),
]))
print(model.fc1)
```

**Common mistake**
- Reusing the same name twice (later layers overwrite earlier ones).

---

## Short 16 — Access Layers by Index

**Hook**
Sequential models are indexable like a list.

**Talk track**
- “Layer 0 is your first layer.”
- “Great for quick inspection.”

**Code**
```python
import torch.nn as nn

model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
print(model[0])
```

**Common mistake**
- Forgetting the order changed after you edited the model.

---

## Short 17 — Dropout: Regularize by Randomly Zeroing Neurons

**Hook**
Dropout fights overfitting by adding noise during training.

**Talk track**
- “Dropout randomly sets activations to 0 during training.”
- “It’s disabled during evaluation.”

**Code**
```python
import torch.nn as nn

drop = nn.Dropout(p=0.5)
```

**Common mistake**
- Forgetting `.eval()` at inference time, leaving dropout ON.

---

## Short 18 — BatchNorm1d: Stabilize Activations

**Hook**
BatchNorm can make training faster and more stable.

**Talk track**
- “BatchNorm normalizes activations using batch statistics.”
- “It behaves differently in train vs eval.”

**Code**
```python
import torch.nn as nn

bn = nn.BatchNorm1d(256)
```

**Common mistake**
- Using BatchNorm but never calling `model.train()` / `model.eval()` correctly.

---

## Short 19 — train() vs eval(): Critical Mode Switch

**Hook**
This single line changes your model’s behavior.

**Talk track**
- “`.train()` enables dropout and updates batchnorm running stats.”
- “`.eval()` disables dropout and uses running stats.”

**Code**
```python
model = nn.Sequential(nn.Dropout(0.5), nn.Linear(4, 2))
model.train()
print(model.training)
model.eval()
print(model.training)
```

**Common mistake**
- Evaluating accuracy with the model still in training mode.

---

## Short 20 — A Cleaner “ImprovedNet” With Dropout + BatchNorm

**Hook**
This is what “real-world” MLPs start to look like.

**Talk track**
- “We add BatchNorm + Dropout around linear layers.”
- “This improves robustness and generalization.”

**Code**
```python
import torch.nn as nn

class ImprovedNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10, dropout_prob=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
```

**Common mistake**
- Putting dropout before batchnorm in a way that changes intended statistics (keep a consistent pattern).

---

## Short 21 — The Training Trio: Model + Loss + Optimizer

**Hook**
Training always starts with these three lines.

**Talk track**
- “Model defines computations.”
- “Loss defines what ‘wrong’ means.”
- “Optimizer defines how we update parameters.”

**Code**
```python
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(nn.Linear(784, 10))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

**Common mistake**
- Forgetting to pass `model.parameters()` into the optimizer.

---

## Short 22 — The 5-Step Training Loop (Again, But Cleaner)

**Hook**
Same loop as Day 3 — just cleaner with nn.Module.

**Talk track**
- “Forward → loss → zero_grad → backward → step.”
- “Memorize this and you can train anything.”

**Code**
```python
# outputs = model(X_batch)
# loss = criterion(outputs, y_batch)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
```

**Common mistake**
- Calling `loss.backward()` before `optimizer.zero_grad()` and accumulating grads.

---

## Short 23 — Why optimizer.zero_grad() Matters

**Hook**
Gradients accumulate unless you clear them.

**Talk track**
- “PyTorch adds gradients by default.”
- “So every iteration you clear old grads before backward.”

**Code**
```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(1, 1)
opt = optim.SGD(model.parameters(), lr=0.1)

opt.zero_grad()  # do this every step
```

**Common mistake**
- Using `model.zero_grad()` and `optimizer.zero_grad()` inconsistently; pick one and be consistent.

---

## Short 24 — Inference Mode: Use torch.no_grad()

**Hook**
Evaluation should not track gradients.

**Talk track**
- “`no_grad` makes inference faster and uses less memory.”
- “Always use it for validation/testing.”

**Code**
```python
import torch

model.eval()
with torch.no_grad():
    y = model(torch.randn(4, 784))
print(y.shape)
```

**Common mistake**
- Forgetting `model.eval()` so dropout/batchnorm behave incorrectly.

---

## Short 25 — Count Parameters in One Line

**Hook**
Quickly estimate model size and complexity.

**Talk track**
- “Parameter count affects memory and overfitting risk.”
- “Use `numel()` to count total elements.”

**Code**
```python
total = sum(p.numel() for p in model.parameters())
print(total)
```

**Common mistake**
- Comparing models without considering parameter count (bigger isn’t always better).

---

## Short 26 — Trainable vs Total Parameters

**Hook**
Not all parameters are necessarily trainable.

**Talk track**
- “Sometimes you freeze layers by setting `requires_grad=False`.”
- “Count trainable parameters separately.”

**Code**
```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total, trainable)
```

**Common mistake**
- Freezing a layer and then forgetting to unfreeze it later.

---

## Short 27 — Inspect Model State: model.training

**Hook**
One boolean tells you if you’re in train or eval mode.

**Talk track**
- “`model.training` is True in train mode, False in eval mode.”
- “Use this to confirm before measuring metrics.”

**Code**
```python
print(model.training)
model.eval()
print(model.training)
```

**Common mistake**
- Thinking `no_grad()` automatically sets eval mode (it doesn’t).

---

## Short 28 — Debug Intermediate Activations With Forward Hooks

**Hook**
Hooks let you peek inside the model without changing forward().

**Talk track**
- “Register a forward hook on a layer.”
- “Capture its output during a forward pass.”

**Code**
```python
import torch

activations = {}

def hook_fn(name):
    def hook(module, inp, out):
        activations[name] = out.detach()
    return hook

layer = model[0] if hasattr(model, "__getitem__") else model
layer.register_forward_hook(hook_fn("layer0"))

_ = model(torch.randn(2, 784))
print(activations["layer0"].shape)
```

**Common mistake**
- Forgetting `.detach()` and accidentally holding onto graphs (memory leak).

---

## Short 29 — Mini Classifier Skeleton (End-to-End)

**Hook**
This is the simplest “real” classifier template.

**Talk track**
- “Define model, criterion, optimizer.”
- “Train in mini-batches.”
- “Evaluate with `eval()` + `no_grad()`.”

**Code**
```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)
```

**Common mistake**
- Returning probabilities instead of logits for CrossEntropyLoss.

---

## Short 30 — Day 4 Summary: Professional PyTorch Workflow

**Hook**
You now write PyTorch like the real world does.

**Talk track**
- “Use `nn.Module` to define models.”
- “Use `optimizer` to update parameters.”
- “Use `.train()` / `.eval()` correctly.”
- “Tomorrow: losses + optimizers in depth.”

**Code**
```python
# nn.Module + forward() + optimizer + train/eval
```

**Common mistake**
- Skipping eval mode and reporting inaccurate metrics.


