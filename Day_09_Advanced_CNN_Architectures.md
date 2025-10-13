# Day 9: Advanced CNN Architectures and Techniques

**Duration**: 10 minutes
**Platform**: Google Colab

---

## Video Script

### Introduction (30 seconds)

**"Welcome to Day 9 - where we learn the architectures that power modern computer vision!"**

Yesterday we built a simple CNN. Today we'll explore groundbreaking architectures:
- VGG: Going deeper with small filters
- ResNet: Residual connections that enable 100+ layers
- Inception: Multiple filter sizes in parallel
- MobileNet: Efficient networks for mobile devices
- Best practices for CNN design

By the end, you'll understand the innovations that revolutionized deep learning.

---

### VGG: The Power of Depth and Small Filters (1.5 minutes)

**"VGG showed that deeper is better"**

```python
import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    """
    VGG block: Multiple Conv layers followed by MaxPool
    Key insight: Stack 3x3 convs instead of one large filter
    Two 3x3 convs = same receptive field as 5x5
    Three 3x3 convs = same receptive field as 7x7
    But fewer parameters and more non-linearities!
    """

    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()

        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels  # After first conv

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class SimplifiedVGG(nn.Module):
    """
    Simplified VGG-like architecture for CIFAR-10
    Based on VGG16 but scaled down
    """

    def __init__(self, num_classes=10):
        super(SimplifiedVGG, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            VGGBlock(3, 64, num_convs=2),

            # Block 2: 16x16 -> 8x8
            VGGBlock(64, 128, num_convs=2),

            # Block 3: 8x8 -> 4x4
            VGGBlock(128, 256, num_convs=3),

            # Block 4: 4x4 -> 2x2
            VGGBlock(256, 512, num_convs=3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Create model
vgg_model = SimplifiedVGG(num_classes=10)
print(vgg_model)

# Count parameters
total_params = sum(p.numel() for p in vgg_model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Test
test_input = torch.randn(2, 3, 32, 32)
output = vgg_model(test_input)
print(f"\nInput shape:  {test_input.shape}")
print(f"Output shape: {output.shape}")

# Why VGG?
print("\nVGG Key Innovations:")
print("✓ Use only 3x3 convolutions")
print("✓ Stack multiple conv layers before pooling")
print("✓ Consistently double channels after pooling")
print("✓ Deep architecture (16-19 layers)")
print("✓ Simple and uniform design")
```

---

### ResNet: Skip Connections Enable Very Deep Networks (2.5 minutes)

**"The breakthrough that enabled 100+ layer networks"**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection
    Key idea: Learn residual F(x) instead of H(x)
    Output = F(x) + x  (skip connection!)

    This solves vanishing gradient problem:
    - Gradients can flow directly through skip connections
    - Network can learn identity mapping (F(x) = 0)
    - Makes training deep networks possible
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity or projection)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Projection shortcut to match dimensions
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Skip connection!
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    Simplified ResNet for CIFAR-10
    """

    def __init__(self, num_blocks_per_stage=[2, 2, 2, 2], num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = 64

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual stages
        self.stage1 = self._make_stage(64, num_blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(128, num_blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(256, num_blocks_per_stage[2], stride=2)
        self.stage4 = self._make_stage(512, num_blocks_per_stage[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_stage(self, out_channels, num_blocks, stride):
        layers = []

        # First block may downsample
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Create model
resnet = ResNet(num_blocks_per_stage=[2, 2, 2, 2], num_classes=10)
print(resnet)

total_params = sum(p.numel() for p in resnet.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Test
test_input = torch.randn(2, 3, 32, 32)
output = resnet(test_input)
print(f"\nInput shape:  {test_input.shape}")
print(f"Output shape: {output.shape}")

# Visualize skip connection
print("\nSkip Connection Visualization:")
print("Input x -> [Conv -> BN -> ReLU -> Conv -> BN] -> F(x)")
print("                    ↓")
print("Input x ----------> (+) -> ReLU -> Output")
print("                    ↑")
print("              Skip connection")
print("\nOutput = F(x) + x")

# Compare: With vs without skip connections
class NoSkipBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x  # No skip!

class WithSkipBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x + identity  # With skip!

print("\nResNet Key Innovations:")
print("✓ Skip connections (residual learning)")
print("✓ Batch normalization throughout")
print("✓ No dropout needed")
print("✓ Global average pooling instead of FC layers")
print("✓ Can train 50, 101, 152+ layer networks")
```

---

### Inception: Multi-Scale Feature Extraction (2 minutes)

**"Why choose one filter size when you can use all?"**

```python
import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    """
    Inception module: Apply multiple filter sizes in parallel
    Key insight: Objects can appear at different scales
    Solution: Use 1x1, 3x3, 5x5 convolutions simultaneously

    Also uses 1x1 convolutions for dimensionality reduction
    (reduces computational cost)
    """

    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3,
                 ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()

        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 3x3 convolution branch (with 1x1 reduction)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 5x5 convolution branch (with 1x1 reduction)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        # MaxPool branch (with 1x1 projection)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Run all branches in parallel
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # Concatenate along channel dimension
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs

class SimplifiedInception(nn.Module):
    """
    Simplified Inception-style network for CIFAR-10
    """

    def __init__(self, num_classes=10):
        super(SimplifiedInception, self).__init__()

        # Initial layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Inception modules
        # InceptionModule(in, 1x1, 3x3red, 3x3, 5x5red, 5x5, pool_proj)
        self.inception1 = InceptionModule(64, 32, 48, 64, 8, 16, 16)
        # Output channels: 32 + 64 + 16 + 16 = 128

        self.inception2 = InceptionModule(128, 64, 96, 128, 16, 32, 32)
        # Output channels: 64 + 128 + 32 + 32 = 256

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inception3 = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        # Output channels: 128 + 192 + 96 + 64 = 480

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(480, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.maxpool(x)
        x = self.inception3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Create model
inception = SimplifiedInception(num_classes=10)
print(inception)

total_params = sum(p.numel() for p in inception.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Test
test_input = torch.randn(2, 3, 32, 32)
output = inception(test_input)
print(f"\nInput shape:  {test_input.shape}")
print(f"Output shape: {output.shape}")

print("\nInception Key Innovations:")
print("✓ Multi-scale feature extraction (1x1, 3x3, 5x5 in parallel)")
print("✓ 1x1 convolutions for dimensionality reduction")
print("✓ Efficient use of parameters")
print("✓ Rich feature representations")
```

---

### MobileNet: Efficient Networks for Mobile (2 minutes)

**"Deep learning on your phone"**

```python
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution = Depthwise + Pointwise

    Standard conv: in_ch × out_ch × k × k multiplications
    Depthwise separable: (in_ch × k × k) + (in_ch × out_ch × 1 × 1)

    Example: 3→32 channels, 3x3 kernel
    Standard: 3 × 32 × 3 × 3 = 864 parameters
    Depthwise separable: (3 × 3 × 3) + (3 × 32 × 1 × 1) = 27 + 96 = 123 parameters
    7x fewer parameters!
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise convolution (each input channel separately)
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                     stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Pointwise convolution (1x1 to combine channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNetV1(nn.Module):
    """
    Simplified MobileNetV1 for CIFAR-10
    """

    def __init__(self, num_classes=10):
        super(MobileNetV1, self).__init__()

        # Standard convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Depthwise separable convolutions
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
        )

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Create model
mobilenet = MobileNetV1(num_classes=10)
print(mobilenet)

total_params = sum(p.numel() for p in mobilenet.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Compare with standard convolution
standard_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
depthwise_sep = DepthwiseSeparableConv(32, 64)

standard_params = sum(p.numel() for p in standard_conv.parameters())
depthwise_params = sum(p.numel() for p in depthwise_sep.parameters())

print(f"\nParameter Comparison (32→64 channels):")
print(f"Standard Conv:           {standard_params:,} parameters")
print(f"Depthwise Separable:     {depthwise_params:,} parameters")
print(f"Reduction:               {standard_params / depthwise_params:.1f}x fewer!")

# Test
test_input = torch.randn(2, 3, 32, 32)
output = mobilenet(test_input)
print(f"\nInput shape:  {test_input.shape}")
print(f"Output shape: {output.shape}")

print("\nMobileNet Key Innovations:")
print("✓ Depthwise separable convolutions (much fewer parameters)")
print("✓ Designed for mobile and embedded devices")
print("✓ Trade-off: slightly lower accuracy for much faster inference")
print("✓ Width and resolution multipliers for scaling")
```

---

### Comparing Architectures (1.5 minutes)

**"Which architecture should you use?"**

```python
import torch
import torch.nn as nn
import time

# Create all models
models = {
    'SimpleCNN': SimplifiedVGG(num_classes=10),
    'ResNet': ResNet(num_blocks_per_stage=[2, 2, 2, 2], num_classes=10),
    'Inception': SimplifiedInception(num_classes=10),
    'MobileNet': MobileNetV1(num_classes=10)
}

# Compare parameters and inference time
test_input = torch.randn(1, 3, 32, 32)

print("Architecture Comparison:\n")
print(f"{'Model':<15} {'Parameters':<15} {'Inference Time (ms)':<20}")
print("=" * 50)

for name, model in models.items():
    model.eval()

    # Count parameters
    params = sum(p.numel() for p in model.parameters())

    # Measure inference time
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            _ = model(test_input)
        elapsed = (time.time() - start) * 10  # ms per inference

    print(f"{name:<15} {params:<15,} {elapsed:<20.2f}")

print("\nWhen to use each:")
print("\n1. VGG:")
print("   ✓ Baseline/educational purposes")
print("   ✓ Transfer learning (pre-trained weights)")
print("   ✗ Many parameters, slow")

print("\n2. ResNet:")
print("   ✓ Need very deep networks")
print("   ✓ General-purpose computer vision")
print("   ✓ Best accuracy for size")
print("   ✓ Most popular choice")

print("\n3. Inception:")
print("   ✓ Multi-scale feature extraction")
print("   ✓ When objects vary in size")
print("   ✓ Good accuracy-to-parameter ratio")

print("\n4. MobileNet:")
print("   ✓ Mobile/embedded deployment")
print("   ✓ Need fast inference")
print("   ✓ Limited compute/memory")
print("   ✗ Slightly lower accuracy")

# Architecture evolution timeline
print("\n" + "="*50)
print("Evolution Timeline:")
print("="*50)
print("2012: AlexNet      - First deep CNN to win ImageNet")
print("2014: VGG          - Deeper with uniform 3x3 filters")
print("2015: ResNet       - Skip connections enable 100+ layers")
print("2015: Inception    - Multi-scale features in parallel")
print("2017: MobileNet    - Efficient networks for mobile")
print("2019: EfficientNet - Neural architecture search")
print("2020: Vision Transformer (ViT) - Attention-based")
```

---

### Best Practices for CNN Design (1 minute)

**"Lessons from the masters"**

```python
"""
CNN Design Best Practices
"""

# 1. Start small, go deep
print("1. Layer Design:")
print("   ✓ Use 3x3 convolutions (occasionally 1x1)")
print("   ✗ Avoid large filters (5x5, 7x7) except first layer")
print("   ✓ Stack multiple small convs instead")

# 2. Channel progression
print("\n2. Channel Progression:")
print("   ✓ Double channels when halving spatial dimensions")
print("   Example: 32x32x64 → 16x16x128 → 8x8x256")

# 3. Normalization
print("\n3. Normalization:")
print("   ✓ BatchNorm after Conv, before activation")
print("   ✓ Or use GroupNorm/LayerNorm for small batches")

# 4. Activation functions
print("\n4. Activations:")
print("   ✓ ReLU is default")
print("   ✓ Try LeakyReLU, ELU, or GELU for better gradients")
print("   ✗ Avoid sigmoid/tanh in hidden layers")

# 5. Pooling strategy
print("\n5. Pooling:")
print("   ✓ MaxPool for aggressive downsampling")
print("   ✓ Or use stride=2 in conv layers")
print("   ✓ Global average pooling before classifier")

# 6. Regularization
print("\n6. Regularization:")
print("   ✓ Dropout in classifier (0.3-0.5)")
print("   ✓ Data augmentation")
print("   ✓ Weight decay (L2 regularization)")
print("   ✗ Usually don't need dropout in conv layers with BatchNorm")

# 7. Skip connections
print("\n7. Skip Connections:")
print("   ✓ Essential for deep networks (>20 layers)")
print("   ✓ Residual or dense connections")

# Example template
class ModernCNN(nn.Module):
    """Modern CNN template incorporating best practices"""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Residual blocks with increasing channels
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),  # Downsample
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),  # Downsample
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, stride=2),  # Downsample
            ResidualBlock(512, 512),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

print("\n8. General Tips:")
print("   ✓ Start with proven architectures (ResNet)")
print("   ✓ Use transfer learning when possible")
print("   ✓ Profile before optimizing")
print("   ✓ Validate design choices with ablation studies")
```

---

## Key Takeaways

1. **VGG**: Deep networks with uniform 3x3 convolutions
2. **ResNet**: Skip connections solve vanishing gradients
3. **Inception**: Multi-scale features via parallel branches
4. **MobileNet**: Depthwise separable convs for efficiency

5. **Design Principles**:
   - Use 3x3 convolutions
   - Add skip connections for depth
   - BatchNorm for stability
   - Global average pooling
   - Double channels when downsampling

6. **Choose Based on**:
   - Accuracy needed → ResNet
   - Speed/size → MobileNet
   - Multi-scale objects → Inception
   - Simplicity → VGG

---

## Today's Practice Exercise

**Implement and compare architectures**

```python
# YOUR TASK:
# 1. Implement a small ResNet-18 for CIFAR-10
# 2. Train VGG, ResNet, and MobileNet side-by-side
# 3. Compare accuracy, parameters, and training time
# 4. Visualize learned features from each
# 5. Try adding your own innovations!

# Bonus: Implement EfficientNet-style compound scaling
```

---

## Tomorrow's Preview

**Day 10: Transfer Learning and Fine-Tuning**

- Using pre-trained models (torchvision.models)
- Feature extraction vs fine-tuning
- Training strategies for transfer learning
- Domain adaptation techniques
- Building on ImageNet weights

---

**"You now understand the architectures powering modern AI! Tomorrow we'll learn to leverage pre-trained models. 🚀"**
