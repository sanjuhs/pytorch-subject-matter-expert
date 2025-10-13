# Day 12: Model Debugging and Optimization

**Duration**: 10 minutes
**Platform**: Google Colab

---

## Video Script

### Introduction (30 seconds)

**"Welcome to Day 12 - where we learn to diagnose and fix training problems!"**

Training deep learning models can be frustrating when things go wrong. Today we'll learn:
- Diagnosing common training problems
- Visualizing gradient flow
- Finding optimal learning rates
- Mixed precision training for speed
- Profiling and optimizing models
- Common pitfalls and how to avoid them

By the end, you'll debug models like a pro.

---

### Common Training Problems and Solutions (2 minutes)

**"The troubleshooting checklist"**

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("Common Training Problems:\n")

# 1. Loss not decreasing
print("=" * 60)
print("Problem 1: Loss Not Decreasing")
print("=" * 60)
print("Symptoms:")
print("- Loss stays flat or increases")
print("- Training accuracy stays low\n")

print("Possible causes and solutions:")
print("❌ Learning rate too low")
print("   ✓ Solution: Increase LR (try 10x higher)")

print("❌ Learning rate too high")
print("   ✓ Solution: Decrease LR (try 10x lower)")

print("❌ Bad initialization")
print("   ✓ Solution: Use Kaiming init for ReLU")

print("❌ Gradient vanishing")
print("   ✓ Solution: Check gradient flow, add BatchNorm, use ResNet")

print("❌ Wrong loss function")
print("   ✓ Solution: CrossEntropy for classification, not MSE")

print("❌ Labels incorrectly formatted")
print("   ✓ Solution: Check label shape and range\n")

# 2. Overfitting
print("=" * 60)
print("Problem 2: Overfitting (Train acc >> Val acc)")
print("=" * 60)
print("Symptoms:")
print("- Training accuracy: 95%")
print("- Validation accuracy: 70%\n")

print("Solutions:")
print("✓ Add dropout (0.3-0.5)")
print("✓ Add weight decay (0.01)")
print("✓ Use data augmentation")
print("✓ Get more training data")
print("✓ Reduce model size")
print("✓ Early stopping\n")

# 3. Underfitting
print("=" * 60)
print("Problem 3: Underfitting (Low train & val acc)")
print("=" * 60)
print("Symptoms:")
print("- Training accuracy: 60%")
print("- Validation accuracy: 58%\n")

print("Solutions:")
print("✓ Increase model capacity (more layers/channels)")
print("✓ Train longer")
print("✓ Reduce regularization")
print("✓ Check data quality")
print("✓ Try different architecture\n")

# 4. Exploding gradients
print("=" * 60)
print("Problem 4: Exploding Gradients")
print("=" * 60)
print("Symptoms:")
print("- Loss becomes NaN")
print("- Gradients are very large (>100)")
print("- Training is unstable\n")

print("Solutions:")
print("✓ Lower learning rate")
print("✓ Gradient clipping")
print("✓ Check for bugs in custom layers")
print("✓ Use BatchNorm")
print("✓ Check input normalization\n")

# 5. Vanishing gradients
print("=" * 60)
print("Problem 5: Vanishing Gradients")
print("=" * 60)
print("Symptoms:")
print("- Early layers not learning")
print("- Gradients very small (<1e-7)")
print("- Deep networks don't help\n")

print("Solutions:")
print("✓ Use ReLU instead of sigmoid/tanh")
print("✓ Add skip connections (ResNet)")
print("✓ Use BatchNorm")
print("✓ Check initialization")
print("✓ Try LSTM/GRU for RNNs\n")

# Quick diagnostic function
def diagnose_training(train_loss_history, val_loss_history,
                      train_acc_history, val_acc_history):
    """Quick diagnostic of training problems"""

    latest_train_loss = train_loss_history[-1]
    latest_val_loss = val_loss_history[-1]
    latest_train_acc = train_acc_history[-1]
    latest_val_acc = val_acc_history[-1]

    print("Training Diagnosis:")
    print("=" * 60)

    # Check if loss is decreasing
    if train_loss_history[-1] > train_loss_history[0]:
        print("⚠️  Training loss not decreasing!")
        print("   → Check learning rate, initialization, gradients")

    # Check overfitting
    if latest_train_acc - latest_val_acc > 15:
        print("⚠️  Overfitting detected!")
        print(f"   Train acc: {latest_train_acc:.1f}%, Val acc: {latest_val_acc:.1f}%")
        print("   → Add regularization (dropout, weight decay, augmentation)")

    # Check underfitting
    if latest_train_acc < 70 and latest_val_acc < 70:
        print("⚠️  Underfitting detected!")
        print(f"   Train acc: {latest_train_acc:.1f}%, Val acc: {latest_val_acc:.1f}%")
        print("   → Increase model capacity or train longer")

    # Check if converged
    if len(val_loss_history) > 5:
        recent_improvement = val_loss_history[-5] - val_loss_history[-1]
        if abs(recent_improvement) < 0.01:
            print("✓ Model converged (no recent improvement)")

    print("=" * 60)

# Example usage
train_losses = [2.3, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2]
val_losses = [2.2, 1.7, 1.3, 1.0, 0.9, 0.9, 0.9]
train_accs = [20, 40, 60, 75, 85, 92, 96]
val_accs = [25, 45, 62, 72, 78, 78, 77]

diagnose_training(train_losses, val_losses, train_accs, val_accs)
```

---

### Visualizing Gradient Flow (1.5 minutes)

**"See what's happening inside your network"**

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model and dummy data
model = SimpleCNN()
x = torch.randn(4, 3, 32, 32)
target = torch.randint(0, 10, (4,))

# Forward and backward pass
criterion = nn.CrossEntropyLoss()
output = model(x)
loss = criterion(output, target)
loss.backward()

# Visualize gradients
def plot_grad_flow(named_parameters):
    """Plot gradient flow through network"""
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(ave_grads)), ave_grads, alpha=0.5, label='Average gradient')
    plt.bar(range(len(max_grads)), max_grads, alpha=0.5, label='Max gradient')
    plt.hlines(0, 0, len(ave_grads), linewidth=2, color='k')
    plt.xticks(range(len(ave_grads)), layers, rotation=90)
    plt.xlim(-1, len(ave_grads))
    plt.xlabel('Layers')
    plt.ylabel('Gradient magnitude')
    plt.title('Gradient Flow')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("Gradient Flow Visualization:")
plot_grad_flow(model.named_parameters())

print("\nInterpretation:")
print("- Healthy: Gradients roughly same magnitude across layers")
print("- Vanishing: Gradients get smaller in early layers")
print("- Exploding: Gradients get very large")

# Check for vanishing/exploding gradients
def check_gradients(model):
    """Check for gradient problems"""
    print("\nGradient Statistics:")
    print("=" * 60)

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_max = param.grad.abs().max().item()

            print(f"{name:20s} | mean: {grad_mean:>10.6f} | "
                  f"std: {grad_std:>10.6f} | max: {grad_max:>10.6f}")

            # Warning flags
            if grad_max > 100:
                print(f"   ⚠️  Gradient exploding!")
            elif grad_max < 1e-7:
                print(f"   ⚠️  Gradient vanishing!")

check_gradients(model)

# Hook to monitor activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))
model.conv3.register_forward_hook(get_activation('conv3'))

# Forward pass
_ = model(x)

# Check activation statistics
print("\n" + "=" * 60)
print("Activation Statistics:")
print("=" * 60)

for name, activation in activations.items():
    mean = activation.mean().item()
    std = activation.std().item()
    zeros = (activation == 0).float().mean().item()

    print(f"{name:10s} | mean: {mean:>8.4f} | std: {std:>8.4f} | "
          f"% zeros: {zeros*100:>6.2f}%")

    if zeros > 0.9:
        print(f"   ⚠️  Most neurons dead (ReLU dying?)")
```

---

### Learning Rate Finder (1.5 minutes)

**"Find the optimal learning rate automatically"**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class LRFinder:
    """
    Learning Rate Finder (Smith, 2017)
    Gradually increase LR and plot loss
    Optimal LR is usually where loss decreases fastest
    """

    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Save initial state
        self.model_state = model.state_dict()
        self.optimizer_state = optimizer.state_dict()

    def find(self, train_loader, start_lr=1e-7, end_lr=10, num_iter=100):
        """Find optimal learning rate"""

        self.model.train()
        lrs = []
        losses = []

        # Generate learning rates exponentially
        lr_schedule = np.geomspace(start_lr, end_lr, num_iter)

        iterator = iter(train_loader)
        for i, lr in enumerate(lr_schedule):
            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Get batch
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Record
            lrs.append(lr)
            losses.append(loss.item())

            # Stop if loss explodes
            if i > 0 and loss.item() > 4 * min(losses):
                break

        # Restore initial state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        return lrs, losses

    def plot(self, lrs, losses, skip_start=10, skip_end=5):
        """Plot learning rate vs loss"""

        # Skip start and end
        lrs = lrs[skip_start:-skip_end]
        losses = losses[skip_start:-skip_end]

        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)

        # Find minimum
        min_loss_idx = np.argmin(losses)
        min_lr = lrs[min_loss_idx]

        plt.axvline(min_lr, color='r', linestyle='--',
                   label=f'Min loss LR: {min_lr:.2e}')

        # Suggested LR (typically 1/10th of min)
        suggested_lr = min_lr / 10
        plt.axvline(suggested_lr, color='g', linestyle='--',
                   label=f'Suggested LR: {suggested_lr:.2e}')

        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"\nOptimal Learning Rate: {suggested_lr:.2e}")
        print("This is typically where the loss decreases fastest")
        print("(not necessarily the minimum loss point)")

# Example usage
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                  transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Create model
model = SimpleCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# Run LR finder
print("Running Learning Rate Finder...")
lr_finder = LRFinder(model, optimizer, criterion, device)
lrs, losses = lr_finder.find(train_loader, start_lr=1e-6, end_lr=10, num_iter=100)
lr_finder.plot(lrs, losses)

print("\nHow to read the plot:")
print("- Look for where loss decreases fastest (steepest slope)")
print("- Choose LR at that point or slightly lower")
print("- Avoid the minimum (usually too high)")
```

---

### Mixed Precision Training (1.5 minutes)

**"Train 2-3x faster with automatic mixed precision"**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time

print("Mixed Precision Training:\n")
print("What is it?")
print("- Use float16 (half precision) for faster computation")
print("- Use float32 for numerical stability")
print("- PyTorch automatically decides which operations use which")
print("- 2-3x speedup on modern GPUs (Volta, Turing, Ampere)")
print("- Reduces memory usage by ~50%\n")

# Standard training (FP32)
def train_standard(model, train_loader, optimizer, criterion, device):
    model.train()
    start_time = time.time()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i >= 10:  # Only do 10 batches for demo
            break

    elapsed = time.time() - start_time
    return elapsed

# Mixed precision training (FP16 + FP32)
def train_mixed_precision(model, train_loader, optimizer, criterion, device):
    model.train()
    scaler = GradScaler()  # Gradient scaler to prevent underflow
    start_time = time.time()

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Automatic mixed precision context
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()

        # Unscale gradients and step
        scaler.step(optimizer)
        scaler.update()

        if i >= 10:  # Only do 10 batches for demo
            break

    elapsed = time.time() - start_time
    return elapsed

# Compare speeds
if torch.cuda.is_available():
    print("Comparing Training Speeds:\n")

    device = torch.device('cuda')
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                      transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Standard FP32
    time_fp32 = train_standard(model, train_loader, optimizer, criterion, device)
    print(f"FP32 Training: {time_fp32:.3f} seconds")

    # Mixed Precision
    time_mixed = train_mixed_precision(model, train_loader, optimizer, criterion, device)
    print(f"Mixed Precision: {time_mixed:.3f} seconds")

    speedup = time_fp32 / time_mixed
    print(f"\nSpeedup: {speedup:.2f}x")
    print("(Speedup varies by GPU model and network architecture)")

else:
    print("Mixed precision requires CUDA GPU")

# Complete training loop with mixed precision
print("\n" + "=" * 60)
print("Complete Mixed Precision Training Loop:")
print("=" * 60)

code = """
from torch.cuda.amp import autocast, GradScaler

# Setup
model = MyModel().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

# Training loop
for epoch in range(epochs):
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()

        # Forward with autocast
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
"""

print(code)

print("\nKey Changes:")
print("1. Import autocast and GradScaler")
print("2. Create GradScaler instance")
print("3. Wrap forward pass with autocast()")
print("4. Use scaler.scale(loss).backward()")
print("5. Use scaler.step() and scaler.update()")

print("\nBenefits:")
print("✓ 2-3x faster training")
print("✓ 50% less GPU memory")
print("✓ Same accuracy (if done correctly)")
print("✓ Minimal code changes")

print("\nWhen to use:")
print("✓ Training large models")
print("✓ Limited GPU memory")
print("✓ Want faster training")
print("✗ Don't use for very small models (overhead not worth it)")
```

---

### Model Profiling and Optimization (1.5 minutes)

**"Find and fix performance bottlenecks"**

```python
import torch
import torch.nn as nn
import time

print("Model Profiling:\n")

# Simple profiling
def profile_model(model, input_size=(1, 3, 224, 224), device='cpu'):
    """Profile model inference time"""

    model = model.to(device).eval()
    x = torch.randn(*input_size).to(device)

    # Warmup
    for _ in range(10):
        _ = model(x)

    # Profile
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    num_iterations = 100

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)

    if device == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time = elapsed / num_iterations

    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {1/avg_time:.2f} images/sec")

    return avg_time

# Count FLOPs and parameters
def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameter Count:")
    print(f"- Total: {total:,}")
    print(f"- Trainable: {trainable:,}")
    print(f"- Model size: {total * 4 / 1e6:.2f} MB (FP32)")
    print(f"- Model size: {total * 2 / 1e6:.2f} MB (FP16)")

def profile_layer_by_layer(model, x):
    """Profile each layer's computation time"""

    print("\nLayer-by-Layer Profiling:")
    print("=" * 60)

    model.eval()
    times = {}

    # Hook to measure time
    def make_hook(name):
        def hook(module, input, output):
            start = time.time()
            _ = output  # Ensure computation is done
            times[name] = time.time() - start
        return hook

    # Register hooks
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module.register_forward_hook(make_hook(name))

    # Forward pass
    with torch.no_grad():
        _ = model(x)

    # Print results
    print(f"{'Layer':<40s} {'Time (ms)':<15s}")
    print("-" * 60)
    for name, t in sorted(times.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{name:<40s} {t*1000:<15.4f}")

# PyTorch Profiler (advanced)
print("=" * 60)
print("Using PyTorch Profiler:")
print("=" * 60)

from torch.profiler import profile, ProfilerActivity

model = SimpleCNN()
x = torch.randn(4, 3, 32, 32)

with profile(activities=[ProfilerActivity.CPU],
             record_shapes=True) as prof:
    model(x)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Optimization tips
print("\n" + "=" * 60)
print("Optimization Tips:")
print("=" * 60)

print("\n1. Batch Operations:")
print("   ❌ for x in batch: output = model(x)")
print("   ✓  output = model(batch)")

print("\n2. Use In-Place Operations:")
print("   ❌ x = x + 1")
print("   ✓  x += 1  (or x.add_(1))")

print("\n3. Avoid .item() in Training Loop:")
print("   ❌ loss = criterion(...).item()  # Synchronizes CPU-GPU")
print("   ✓  loss = criterion(...)  # Keep on GPU")

print("\n4. Use DataLoader Efficiently:")
print("   ✓  num_workers=4  # Parallel data loading")
print("   ✓  pin_memory=True  # Faster GPU transfer")
print("   ✓  persistent_workers=True  # Reuse workers")

print("\n5. Gradient Accumulation (for large models):")
print("   # Effective batch size = batch_size * accumulation_steps")
print("   for i, (x, y) in enumerate(loader):")
print("       loss = model(x, y) / accumulation_steps")
print("       loss.backward()")
print("       if (i + 1) % accumulation_steps == 0:")
print("           optimizer.step()")
print("           optimizer.zero_grad()")

print("\n6. Model Optimization:")
print("   ✓  Use torch.compile() (PyTorch 2.0+)")
print("   ✓  Quantization (8-bit weights)")
print("   ✓  Pruning (remove unnecessary connections)")
print("   ✓  Knowledge distillation (train smaller model)")

# torch.compile (PyTorch 2.0+)
print("\n7. Torch Compile (PyTorch 2.0+):")
print("   model = torch.compile(model)  # Just-in-time compilation")
print("   # 30-50% speedup with single line!")

# Model size reduction
print("\n8. Reduce Model Size:")
print("   # Quantization")
print("   quantized_model = torch.quantization.quantize_dynamic(")
print("       model, {nn.Linear}, dtype=torch.qint8)")
print("   # 4x smaller model, minimal accuracy loss")
```

---

### Debugging Checklist (1 minute)

**"Step-by-step debugging guide"**

```python
print("DEBUGGING CHECKLIST")
print("=" * 60)

checklist = """
□ 1. DATA LOADING
   □ Check data shapes match model input
   □ Verify labels are correct type (long for CrossEntropy)
   □ Check normalization (mean=0, std=1)
   □ Visualize a few samples
   □ Ensure no NaN/Inf in data

□ 2. MODEL ARCHITECTURE
   □ Print model architecture
   □ Count parameters (reasonable size?)
   □ Test forward pass with dummy data
   □ Check output shape matches target shape
   □ Verify no dimension mismatches

□ 3. LOSS FUNCTION
   □ Using correct loss (CrossEntropy for classification)
   □ Check loss value (should be around -log(1/num_classes) initially)
   □ Verify loss decreases on single batch (overfit test)

□ 4. OPTIMIZER
   □ Learning rate in right range (1e-5 to 1e-1)
   □ Use LR finder to find optimal LR
   □ Try different optimizers (Adam is good default)
   □ Check optimizer is updating correct parameters

□ 5. TRAINING LOOP
   □ model.train() before training
   □ model.eval() before validation
   □ optimizer.zero_grad() before backward
   □ loss.backward() before optimizer.step()
   □ Move data to correct device (CPU/GPU)

□ 6. GRADIENTS
   □ Check gradients are not None
   □ Verify gradients are not too large (>100) or small (<1e-7)
   □ Use gradient clipping if exploding
   □ Check for dead ReLUs (all zeros)

□ 7. OVERFITTING/UNDERFITTING
   □ Try overfitting single batch (sanity check)
   □ Add regularization if overfitting
   □ Increase capacity if underfitting
   □ Check learning curves

□ 8. DEVICE ISSUES
   □ All tensors on same device
   □ Model on correct device
   □ Check CUDA availability
   □ Monitor GPU memory usage

□ 9. NUMERICAL STABILITY
   □ Check for NaN/Inf in loss
   □ Use mixed precision carefully
   □ Add epsilon to divisions
   □ Check input ranges

□ 10. REPRODUCIBILITY
    □ Set random seeds (torch, numpy, random)
    □ Disable cudnn.benchmark for reproducibility
    □ Document hyperparameters
"""

print(checklist)

# Quick sanity check
def sanity_check(model, train_loader, criterion, optimizer, device):
    """
    Overfit test: Can model overfit a single batch?
    If not, there's a bug somewhere!
    """

    print("\nSanity Check: Overfitting Single Batch")
    print("=" * 60)

    model.train()

    # Get one batch
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    # Try to overfit
    for i in range(100):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            acc = (outputs.argmax(1) == labels).float().mean()
            print(f"Iter {i+1:3d}: Loss = {loss.item():.4f}, Acc = {acc.item():.4f}")

    final_acc = (outputs.argmax(1) == labels).float().mean()

    if final_acc > 0.95:
        print("\n✓ Sanity check PASSED: Model can learn")
    else:
        print("\n✗ Sanity check FAILED: Model cannot overfit single batch")
        print("  → Check model architecture, loss function, optimizer")

print("\nUse this checklist when things go wrong!")
print("Work through it systematically to find the issue.")
```

---

## Key Takeaways

1. **Common Problems**:
   - Loss not decreasing → Check LR, initialization, gradients
   - Overfitting → Add regularization, data augmentation
   - Underfitting → Increase model capacity
   - Exploding gradients → Lower LR, gradient clipping
   - Vanishing gradients → Skip connections, BatchNorm

2. **Debugging Tools**:
   - Gradient flow visualization
   - Learning rate finder
   - Sanity check (overfit single batch)
   - Layer-by-layer profiling

3. **Optimization**:
   - Mixed precision training (2-3x speedup)
   - Batch operations
   - Efficient data loading
   - torch.compile (PyTorch 2.0+)

4. **Profiling**:
   - Measure inference time
   - Count parameters
   - Find bottleneck layers
   - Use PyTorch Profiler

5. **Best Practices**:
   - Always run sanity check first
   - Use LR finder
   - Monitor gradient flow
   - Profile before optimizing

---

## Today's Practice Exercise

**Debug and optimize a broken model**

```python
# YOUR TASK:
# 1. Intentionally create bugs (wrong loss, bad LR, etc.)
# 2. Use debugging tools to find and fix them
# 3. Run LR finder to optimize learning rate
# 4. Profile model and find bottlenecks
# 5. Implement mixed precision training
# 6. Compare before/after performance

# Bonus: Implement gradient accumulation
# Bonus: Try torch.compile() if using PyTorch 2.0+
```

---

## Tomorrow's Preview

**Day 13: Introduction to Recurrent Neural Networks (RNNs)**

- Understanding sequential data
- Vanilla RNNs and their limitations
- LSTM and GRU architectures
- Bidirectional RNNs
- Applications: Text generation, time series

---

**"You now know how to debug and optimize models! Tomorrow we'll dive into sequence modeling with RNNs. 🚀"**
