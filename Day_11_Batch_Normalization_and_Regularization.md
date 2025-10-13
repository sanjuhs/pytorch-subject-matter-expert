# Day 11: Batch Normalization and Regularization Techniques

**Duration**: 10 minutes
**Platform**: Google Colab

---

## Video Script

### Introduction (30 seconds)

**"Welcome to Day 11 - mastering the techniques that make deep learning work!"**

Training deep networks is tricky. Today we'll learn the tricks that make it reliable:
- Batch Normalization and why it's revolutionary
- Other normalization techniques (Layer, Group, Instance)
- Dropout and its variants
- Weight initialization strategies
- Data augmentation best practices
- Preventing overfitting

By the end, you'll train stable, high-performing networks.

---

### The Problem: Internal Covariate Shift (1 minute)

**"Why deep networks are hard to train"**

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Simulate activation distributions through layers
def simulate_forward_pass(num_layers=5, samples=1000):
    x = torch.randn(samples, 100)  # Initial input
    activations = [x]

    for i in range(num_layers):
        # Simple linear transformation + ReLU
        W = torch.randn(100, 100) * 0.5  # Random weights
        x = torch.relu(x @ W)
        activations.append(x)

    return activations

# Without normalization
print("Without Normalization:")
activations = simulate_forward_pass(num_layers=5)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, act in enumerate(activations):
    axes[i].hist(act.flatten().numpy(), bins=50, alpha=0.7)
    axes[i].set_title(f'Layer {i}')
    axes[i].set_xlabel('Activation value')
    axes[i].set_ylabel('Frequency')

    mean = act.mean().item()
    std = act.std().item()
    axes[i].text(0.7, 0.9, f'μ={mean:.2f}\nσ={std:.2f}',
                transform=axes[i].transAxes)

plt.suptitle('Activation Distributions Without Normalization', fontsize=16)
plt.tight_layout()
plt.show()

print("\nProblems observed:")
print("❌ Distribution shifts between layers (covariate shift)")
print("❌ Activations may vanish (all near 0)")
print("❌ Or explode (very large values)")
print("❌ Gradients become unstable")
print("❌ Training is slow and sensitive to learning rate")

print("\nInternal Covariate Shift:")
print("- Each layer's input distribution changes during training")
print("- As previous layers update, current layer sees different inputs")
print("- Network must constantly adapt → slow training")
```

---

### Batch Normalization: The Solution (2 minutes)

**"Normalize activations to stabilize training"**

```python
import torch
import torch.nn as nn

# What BatchNorm does
print("Batch Normalization Algorithm:\n")
print("For each mini-batch:")
print("1. Compute mean: μ = (1/m) Σ x_i")
print("2. Compute variance: σ² = (1/m) Σ (x_i - μ)²")
print("3. Normalize: x̂ = (x - μ) / √(σ² + ε)")
print("4. Scale and shift: y = γ·x̂ + β")
print("   (γ and β are learnable parameters)")

# Manual BatchNorm implementation
class ManualBatchNorm1d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = torch.ones(num_features)   # Scale
        self.beta = torch.zeros(num_features)   # Shift

        # Running statistics (for inference)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x, training=True):
        if training:
            # Training mode: use batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Normalize
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * batch_var
        else:
            # Inference mode: use running statistics
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # Scale and shift
        out = self.gamma * x_norm + self.beta
        return out

# Test manual implementation
batch_size = 32
features = 10
x = torch.randn(batch_size, features)

manual_bn = ManualBatchNorm1d(features)
output = manual_bn.forward(x, training=True)

print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output mean: {output.mean(dim=0)}")  # Should be close to 0
print(f"Output std: {output.std(dim=0)}")    # Should be close to 1

# PyTorch's BatchNorm
print("\n" + "="*50)
print("Using PyTorch BatchNorm:")
print("="*50)

bn = nn.BatchNorm1d(num_features=10)
print(bn)

# For 2D images: BatchNorm2d
bn2d = nn.BatchNorm2d(num_features=64)  # 64 channels
print(f"\nBatchNorm2d: {bn2d}")

# Test with image data
batch_images = torch.randn(16, 64, 32, 32)  # (B, C, H, W)
normalized = bn2d(batch_images)

print(f"\nInput shape: {batch_images.shape}")
print(f"Output shape: {normalized.shape}")

# Per-channel statistics
for c in range(min(3, 64)):
    channel_mean = normalized[:, c].mean()
    channel_std = normalized[:, c].std()
    print(f"Channel {c}: mean={channel_mean:.4f}, std={channel_std:.4f}")

# Benefits of BatchNorm
print("\n" + "="*50)
print("BatchNorm Benefits:")
print("="*50)
print("✓ Allows higher learning rates → faster training")
print("✓ Reduces sensitivity to initialization")
print("✓ Acts as regularization (slight noise from batch statistics)")
print("✓ Enables very deep networks")
print("✓ Reduces internal covariate shift")
```

---

### Normalization Variants (1.5 minutes)

**"Different normalizations for different scenarios"**

```python
import torch
import torch.nn as nn

# Different normalization techniques
batch_size, channels, height, width = 4, 3, 8, 8
x = torch.randn(batch_size, channels, height, width)

print(f"Input shape: {x.shape} (B, C, H, W)\n")

# 1. Batch Normalization
bn = nn.BatchNorm2d(channels)
bn_out = bn(x)
print("1. Batch Normalization (BatchNorm2d):")
print("   - Normalizes across batch dimension")
print("   - Computes mean/std per channel across all samples")
print("   - Good for: Large batch sizes (>16)")
print("   - Bad for: Small batches, RNNs, online learning")
print(f"   Output shape: {bn_out.shape}\n")

# 2. Layer Normalization
ln = nn.LayerNorm([channels, height, width])
ln_out = ln(x)
print("2. Layer Normalization (LayerNorm):")
print("   - Normalizes across channel, height, width dimensions")
print("   - Independent of batch size")
print("   - Good for: RNNs, transformers, small batches")
print("   - Used in: BERT, GPT, and most LLMs")
print(f"   Output shape: {ln_out.shape}\n")

# 3. Instance Normalization
in_norm = nn.InstanceNorm2d(channels)
in_out = in_norm(x)
print("3. Instance Normalization (InstanceNorm2d):")
print("   - Normalizes each sample independently")
print("   - Per-channel, per-sample statistics")
print("   - Good for: Style transfer, GANs")
print("   - Removes instance-specific contrast information")
print(f"   Output shape: {in_out.shape}\n")

# 4. Group Normalization
gn = nn.GroupNorm(num_groups=1, num_channels=channels)  # 1 group = LayerNorm
gn_out = gn(x)
print("4. Group Normalization (GroupNorm):")
print("   - Divides channels into groups, normalizes within groups")
print("   - Independent of batch size")
print("   - Good for: Small batches, computer vision")
print("   - Middle ground between LayerNorm and InstanceNorm")
print(f"   Output shape: {gn_out.shape}\n")

# Visual comparison
print("="*60)
print("Normalization Comparison:")
print("="*60)

comparison = """
Dimension normalized:
                    Batch   Channels   Height   Width
BatchNorm            ✓        ✗         ✓        ✓
LayerNorm            ✗        ✓         ✓        ✓
InstanceNorm         ✗        ✗         ✓        ✓
GroupNorm            ✗       (groups)   ✓        ✓

When to use:
- BatchNorm:     Standard CNNs, large batches
- LayerNorm:     Transformers, RNNs, small batches
- InstanceNorm:  Style transfer, GANs
- GroupNorm:     Computer vision with small batches
"""

print(comparison)

# Practical example in a network
class ModernCNN(nn.Module):
    def __init__(self, num_classes=10, norm_type='batch'):
        super().__init__()

        if norm_type == 'batch':
            norm_layer = lambda c: nn.BatchNorm2d(c)
        elif norm_type == 'layer':
            norm_layer = lambda c: nn.LayerNorm([c, 32, 32])  # Fixed size
        elif norm_type == 'group':
            norm_layer = lambda c: nn.GroupNorm(8, c)  # 8 groups
        elif norm_type == 'instance':
            norm_layer = lambda c: nn.InstanceNorm2d(c)
        else:
            norm_layer = lambda c: nn.Identity()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            norm_layer(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            norm_layer(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Test different normalizations
for norm_type in ['batch', 'layer', 'group', 'instance', 'none']:
    model = ModernCNN(norm_type=norm_type)
    print(f"\n{norm_type.title()} Normalization model created")
```

---

### Dropout and Regularization (2 minutes)

**"Preventing overfitting"**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Standard Dropout
print("1. Standard Dropout:")
print("   - Randomly zero neurons with probability p")
print("   - Forces network to not rely on specific neurons")
print("   - Acts as ensemble of subnetworks\n")

dropout = nn.Dropout(p=0.5)  # Drop 50% of neurons

x = torch.randn(4, 10)
print(f"Input:\n{x[0]}\n")

# Training mode
dropout.train()
x_train = dropout(x)
print(f"After dropout (training):\n{x_train[0]}")
print(f"~50% of values are 0, others scaled by 1/(1-p)\n")

# Inference mode
dropout.eval()
x_eval = dropout(x)
print(f"After dropout (inference):\n{x_eval[0]}")
print("No dropout applied during inference\n")

# 2. Dropout2d (spatial dropout for CNNs)
print("="*60)
print("2. Dropout2d (Spatial Dropout):")
print("="*60)
print("   - Drops entire channels instead of individual values")
print("   - Better for convolutional layers")
print("   - Removes entire feature maps\n")

dropout2d = nn.Dropout2d(p=0.3)
images = torch.randn(2, 16, 8, 8)  # (B, C, H, W)

dropout2d.train()
dropped = dropout2d(images)

print(f"Input shape: {images.shape}")
print(f"Channels dropped: {(dropped.sum(dim=(2,3)) == 0).sum().item()} / {16}")

# 3. AlphaDropout (for SELU activation)
print("\n" + "="*60)
print("3. AlphaDropout (for SELU networks):")
print("="*60)
print("   - Self-normalizing dropout")
print("   - Maintains mean and variance")
print("   - Use with SELU activation\n")

# 4. Dropout variants comparison
class DropoutComparison(nn.Module):
    def __init__(self, dropout_type='standard'):
        super().__init__()

        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)

        if dropout_type == 'standard':
            self.dropout = nn.Dropout(0.5)
        elif dropout_type == 'spatial':
            self.dropout = nn.Dropout2d(0.5)
        elif dropout_type == 'alpha':
            self.dropout = nn.AlphaDropout(0.5)
        else:
            self.dropout = nn.Identity()

        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.dropout(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Other regularization techniques
print("="*60)
print("Other Regularization Techniques:")
print("="*60)

print("\n1. Weight Decay (L2 Regularization):")
print("   - Add penalty: loss = loss + λ·||W||²")
print("   - Prevents weights from growing too large")
print("   - Use in optimizer:\n")
print("   optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)")

print("\n2. Data Augmentation:")
print("   - Transform training data (rotate, flip, crop, etc.)")
print("   - Increases effective dataset size")
print("   - Best regularization for images")

print("\n3. Early Stopping:")
print("   - Stop training when validation loss stops improving")
print("   - Prevents overfitting to training data")

print("\n4. Label Smoothing:")
print("   - Soften hard labels: [0, 1, 0] → [0.05, 0.9, 0.05]")
print("   - Prevents overconfident predictions")

# Label smoothing example
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)

        # Convert target to one-hot
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)

        # Apply smoothing
        smooth_target = one_hot * (1 - self.smoothing) + \
                       self.smoothing / n_classes

        loss = (-smooth_target * log_probs).sum(dim=1).mean()
        return loss

print("\n5. Mixup:")
print("   - Mix two samples: x = λ·x1 + (1-λ)·x2")
print("   - Interpolate labels too")
print("   - Creates synthetic training examples")

# Mixup implementation
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

print("\n6. Cutout/CutMix:")
print("   - Randomly mask out regions of images")
print("   - Forces model to use entire image")
print("   - Very effective for computer vision")
```

---

### Weight Initialization (1.5 minutes)

**"Start training on the right foot"**

```python
import torch
import torch.nn as nn
import math

print("Weight Initialization Strategies:\n")

# Problem: Bad initialization
print("Why initialization matters:")
print("❌ Too small: Activations/gradients vanish")
print("❌ Too large: Activations/gradients explode")
print("✓ Just right: Stable training from the start\n")

# 1. Xavier/Glorot Initialization
print("1. Xavier Initialization (for tanh/sigmoid):")
print("   - Uniform: U(-√(6/(nin+nout)), √(6/(nin+nout)))")
print("   - Normal: N(0, √(2/(nin+nout)))")
print("   - Maintains variance across layers\n")

layer = nn.Linear(100, 50)
nn.init.xavier_uniform_(layer.weight)
print(f"Xavier uniform: mean={layer.weight.mean():.4f}, "
      f"std={layer.weight.std():.4f}")

nn.init.xavier_normal_(layer.weight)
print(f"Xavier normal: mean={layer.weight.mean():.4f}, "
      f"std={layer.weight.std():.4f}\n")

# 2. Kaiming/He Initialization
print("2. Kaiming Initialization (for ReLU):")
print("   - Accounts for ReLU killing half the neurons")
print("   - Uniform: U(-√(6/nin), √(6/nin))")
print("   - Normal: N(0, √(2/nin))")
print("   - Default in PyTorch for Conv and Linear\n")

layer = nn.Linear(100, 50)
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
print(f"Kaiming uniform: mean={layer.weight.mean():.4f}, "
      f"std={layer.weight.std():.4f}")

nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
print(f"Kaiming normal: mean={layer.weight.mean():.4f}, "
      f"std={layer.weight.std():.4f}\n")

# 3. Orthogonal Initialization
print("3. Orthogonal Initialization (for RNNs):")
print("   - Weights form orthogonal matrix")
print("   - Preserves gradient norms")
print("   - Good for recurrent networks\n")

layer = nn.Linear(100, 100)
nn.init.orthogonal_(layer.weight)
print(f"Orthogonal: mean={layer.weight.mean():.4f}, "
      f"std={layer.weight.std():.4f}\n")

# Complete initialization function
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                   nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Apply initialization
model = SimpleCNN()
initialize_weights(model)

print("Model initialized with:")
print("- Kaiming initialization for Conv2d and Linear")
print("- Constant initialization for BatchNorm")
print("\nInitialization Guidelines:")
print("- ReLU/LeakyReLU → Kaiming/He init")
print("- tanh/sigmoid → Xavier/Glorot init")
print("- SELU → lecun_normal init")
print("- RNNs → Orthogonal init")
```

---

### Putting It All Together (1.5 minutes)

**"A fully regularized, modern architecture"**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernCNN(nn.Module):
    """
    Modern CNN with all best practices:
    - Kaiming initialization
    - Batch normalization
    - Dropout regularization
    - Residual connections
    """

    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(ModernCNN, self).__init__()

        # Initial conv block
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 128)
        self.res_block2 = self._make_residual_block(128, 256)
        self.res_block3 = self._make_residual_block(256, 512)

        # Classifier with dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        x = self.res_block1(x)
        x = F.relu(x)

        x = self.res_block2(x)
        x = F.relu(x)

        x = self.res_block3(x)
        x = F.relu(x)

        # Global pooling and classifier
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

# Create model
model = ModernCNN(num_classes=10, dropout_rate=0.5)
print(model)

print("\n" + "="*60)
print("Best Practices Applied:")
print("="*60)
print("✓ Kaiming initialization for Conv layers")
print("✓ Batch normalization after each conv")
print("✓ ReLU activations")
print("✓ Residual connections")
print("✓ Dropout before classifier")
print("✓ Global average pooling")

print("\nTraining recommendations:")
print("- Optimizer: AdamW with weight_decay=0.01")
print("- Learning rate: 0.001 with cosine decay")
print("- Data augmentation: RandomCrop, HorizontalFlip, ColorJitter")
print("- Batch size: 64-128")
print("- Early stopping: patience=10")
print("- Label smoothing: 0.1")
print("- Gradient clipping: max_norm=1.0")

# Test model
test_input = torch.randn(4, 3, 32, 32)
output = model(test_input)
print(f"\nInput shape: {test_input.shape}")
print(f"Output shape: {output.shape}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

---

## Key Takeaways

1. **Batch Normalization**:
   - Normalizes layer inputs to stabilize training
   - Allows higher learning rates
   - Acts as regularization
   - Use after Conv/Linear, before activation

2. **Normalization Variants**:
   - BatchNorm: CNNs with large batches
   - LayerNorm: Transformers, RNNs
   - GroupNorm: Small batches
   - InstanceNorm: Style transfer

3. **Dropout**:
   - Standard Dropout: After FC layers (p=0.3-0.5)
   - Dropout2d: After Conv layers
   - Not needed if using BatchNorm in conv layers

4. **Weight Initialization**:
   - Kaiming/He: For ReLU networks (default)
   - Xavier/Glorot: For tanh/sigmoid
   - Orthogonal: For RNNs

5. **Regularization Stack**:
   - Data augmentation (most important!)
   - Batch normalization
   - Weight decay (L2)
   - Dropout (in classifier)
   - Early stopping
   - Label smoothing

---

## Today's Practice Exercise

**Build a fully regularized network**

```python
# YOUR TASK:
# 1. Build a CNN with all regularization techniques
# 2. Train with and without each technique
# 3. Compare overfitting (train vs val accuracy)
# 4. Visualize the effect of different normalizations
# 5. Implement mixup or cutout
# 6. Try different initialization strategies

# Bonus: Implement your own BatchNorm from scratch
# Bonus: Add label smoothing and compare results
```

---

## Tomorrow's Preview

**Day 12: Model Debugging and Optimization**

- Diagnosing training problems
- Gradient flow visualization
- Learning rate finding
- Mixed precision training
- Model profiling and optimization
- Common pitfalls and solutions

---

**"You now master the techniques that make deep learning reliable! Tomorrow we'll learn to debug and optimize models. 🚀"**
