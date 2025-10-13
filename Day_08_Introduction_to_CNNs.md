# Day 8: Introduction to Convolutional Neural Networks (CNNs)

**Duration**: 10 minutes
**Platform**: Google Colab

---

## Video Script

### Introduction (30 seconds)

**"Welcome to Day 8 - where we learn how computers see!"**

Today we enter the world of Convolutional Neural Networks (CNNs) - the backbone of computer vision. We'll learn:
- What convolutions are and why they work for images
- Understanding `nn.Conv2d` in PyTorch
- Pooling layers (MaxPool, AvgPool)
- Building your first CNN from scratch
- Training on image classification tasks

By the end, you'll understand why CNNs revolutionized computer vision.

---

### Why CNNs for Images? (1 minute)

**"Regular neural networks don't work well for images"**

```python
import torch
import torch.nn as nn

# Problem with fully connected networks for images
image_size = 224 * 224 * 3  # 224x224 RGB image
hidden_size = 1000

fc_layer = nn.Linear(image_size, hidden_size)
print(f"Image size: {image_size:,} pixels")
print(f"Parameters in one FC layer: {image_size * hidden_size:,}")
print(f"That's {(image_size * hidden_size) / 1e6:.1f} million parameters!")

# Issues with fully connected networks:
# 1. Too many parameters → slow, memory intensive
# 2. Lose spatial structure (image becomes 1D vector)
# 3. Not translation invariant (cat on left ≠ cat on right)
# 4. Don't capture local patterns (edges, textures)

# CNNs solve these problems!
print("\nCNN advantages:")
print("✓ Fewer parameters (weight sharing)")
print("✓ Preserve spatial structure")
print("✓ Translation invariant")
print("✓ Learn hierarchical features (edges → shapes → objects)")
```

---

### Understanding Convolution Operation (2 minutes)

**"Convolution: sliding a filter over an image"**

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Create a simple 5x5 image
image = torch.tensor([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype=torch.float32)

print("Original image:")
print(image)

# Define a 3x3 edge detection filter (vertical edges)
vertical_filter = torch.tensor([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=torch.float32)

horizontal_filter = torch.tensor([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=torch.float32)

# Reshape for conv2d: (batch, channels, height, width)
image_batch = image.unsqueeze(0).unsqueeze(0)  # (1, 1, 5, 5)
vertical_filter_batch = vertical_filter.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
horizontal_filter_batch = horizontal_filter.unsqueeze(0).unsqueeze(0)

# Apply convolution
vertical_edges = F.conv2d(image_batch, vertical_filter_batch, padding=1)
horizontal_edges = F.conv2d(image_batch, horizontal_filter_batch, padding=1)

print("\nVertical edge detection:")
print(vertical_edges.squeeze())

print("\nHorizontal edge detection:")
print(horizontal_edges.squeeze())

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(vertical_filter, cmap='gray')
axes[1].set_title('Vertical Filter')
axes[1].axis('off')

axes[2].imshow(vertical_edges.squeeze(), cmap='gray')
axes[2].set_title('Vertical Edges')
axes[2].axis('off')

axes[3].imshow(horizontal_edges.squeeze(), cmap='gray')
axes[3].set_title('Horizontal Edges')
axes[3].axis('off')

plt.tight_layout()
plt.show()

# Manual convolution (for understanding)
def manual_conv2d(image, kernel):
    """
    Simple convolution implementation (no padding)
    """
    h, w = image.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1
    output = torch.zeros(output_h, output_w)

    for i in range(output_h):
        for j in range(output_w):
            # Extract patch
            patch = image[i:i+kh, j:j+kw]
            # Element-wise multiply and sum
            output[i, j] = (patch * kernel).sum()

    return output

manual_result = manual_conv2d(image, vertical_filter)
print("\nManual convolution result:")
print(manual_result)
```

---

### nn.Conv2d in PyTorch (2 minutes)

**"Understanding PyTorch's convolutional layer"**

```python
import torch
import torch.nn as nn

# nn.Conv2d parameters
conv = nn.Conv2d(
    in_channels=3,      # Input channels (RGB = 3)
    out_channels=16,    # Number of filters (output channels)
    kernel_size=3,      # Filter size (3x3)
    stride=1,           # Step size when sliding filter
    padding=1,          # Add padding to preserve size
    bias=True           # Include bias term
)

print("Conv2d layer:")
print(conv)
print(f"\nWeight shape: {conv.weight.shape}")  # (out_channels, in_channels, kh, kw)
print(f"Bias shape: {conv.bias.shape}")        # (out_channels,)
print(f"Total parameters: {conv.weight.numel() + conv.bias.numel()}")

# Test with random input
batch_size = 4
input_tensor = torch.randn(batch_size, 3, 32, 32)  # (B, C, H, W)
output = conv(input_tensor)

print(f"\nInput shape:  {input_tensor.shape}")   # (4, 3, 32, 32)
print(f"Output shape: {output.shape}")           # (4, 16, 32, 32)

# Understanding stride
conv_stride1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
conv_stride2 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)

out_stride1 = conv_stride1(input_tensor)
out_stride2 = conv_stride2(input_tensor)

print(f"\nStride 1 output: {out_stride1.shape}")  # (4, 16, 32, 32) - same size
print(f"Stride 2 output: {out_stride2.shape}")    # (4, 16, 16, 16) - halved

# Understanding padding
conv_no_pad = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
conv_pad = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

out_no_pad = conv_no_pad(input_tensor)
out_pad = conv_pad(input_tensor)

print(f"\nNo padding output: {out_no_pad.shape}")  # (4, 16, 30, 30) - shrunk
print(f"Padding=1 output:  {out_pad.shape}")      # (4, 16, 32, 32) - preserved

# Formula for output size:
# output_size = (input_size + 2*padding - kernel_size) / stride + 1
input_size = 32
kernel_size = 3
stride = 1
padding = 1

output_size = (input_size + 2*padding - kernel_size) // stride + 1
print(f"\nCalculated output size: {output_size}")
```

---

### Pooling Layers (1.5 minutes)

**"Downsampling to reduce spatial dimensions"**

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create sample feature map
feature_map = torch.randn(1, 1, 8, 8)

# MaxPool2d: Take maximum value in window
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
maxpool_output = maxpool(feature_map)

# AvgPool2d: Take average value in window
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
avgpool_output = avgpool(feature_map)

print(f"Input shape:    {feature_map.shape}")      # (1, 1, 8, 8)
print(f"MaxPool output: {maxpool_output.shape}")   # (1, 1, 4, 4)
print(f"AvgPool output: {avgpool_output.shape}")   # (1, 1, 4, 4)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(feature_map.squeeze(), cmap='viridis')
axes[0].set_title('Original (8x8)')
axes[0].axis('off')

axes[1].imshow(maxpool_output.squeeze(), cmap='viridis')
axes[1].set_title('MaxPool (4x4)')
axes[1].axis('off')

axes[2].imshow(avgpool_output.squeeze(), cmap='viridis')
axes[2].set_title('AvgPool (4x4)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Manual MaxPool example
x = torch.tensor([
    [1, 3, 2, 4],
    [5, 6, 7, 8],
    [2, 1, 9, 3],
    [4, 7, 2, 6]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

print("\nOriginal 4x4:")
print(x.squeeze())

maxpool_2x2 = nn.MaxPool2d(2)
result = maxpool_2x2(x)

print("\nAfter MaxPool (2x2 with stride 2):")
print(result.squeeze())
print("\nEach value is the max of a 2x2 region:")
print("Top-left: max(1,3,5,6) = 6")
print("Top-right: max(2,4,7,8) = 8")
print("Bottom-left: max(2,1,4,7) = 7")
print("Bottom-right: max(9,3,2,6) = 9")

# Why pooling?
print("\nBenefits of pooling:")
print("✓ Reduces spatial dimensions → fewer parameters")
print("✓ Provides translation invariance")
print("✓ Captures most important features (MaxPool)")
print("✓ Controls overfitting")
```

---

### Building Your First CNN (2.5 minutes)

**"A complete CNN for image classification"**

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Simple CNN for CIFAR-10 classification
    Input: 3x32x32 RGB images
    Output: 10 class logits
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input: (batch, 3, 32, 32)

        x = self.conv1(x)    # (batch, 32, 32, 32)
        x = self.relu1(x)
        x = self.pool1(x)    # (batch, 32, 16, 16)

        x = self.conv2(x)    # (batch, 64, 16, 16)
        x = self.relu2(x)
        x = self.pool2(x)    # (batch, 64, 8, 8)

        x = self.conv3(x)    # (batch, 128, 8, 8)
        x = self.relu3(x)
        x = self.pool3(x)    # (batch, 128, 4, 4)

        x = self.flatten(x)  # (batch, 128*4*4)
        x = self.fc1(x)      # (batch, 256)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)      # (batch, 10)

        return x

# Create model
model = SimpleCNN(num_classes=10)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Test forward pass
batch_size = 4
test_input = torch.randn(batch_size, 3, 32, 32)
output = model(test_input)

print(f"\nInput shape:  {test_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Output logits:\n{output}")

# More compact version using nn.Sequential
class CompactCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CompactCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

compact_model = CompactCNN()
print("\n" + "="*50)
print("Compact version:")
print(compact_model)
```

---

### Training a CNN on CIFAR-10 (2 minutes)

**"Complete training pipeline"**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Data preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                  transform=transform_train)
test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                                 transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                          num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                         num_workers=2, pin_memory=True)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# Validation function
def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# Training loop
epochs = 20
train_losses, train_accs = [], []
test_losses, test_accs = [], []

print("\nStarting training...")

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                         optimizer, device)
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")

print("\nTraining complete!")

# Plot results
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, label='Train Loss', marker='o')
ax1.plot(test_losses, label='Test Loss', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(train_accs, label='Train Acc', marker='o')
ax2.plot(test_accs, label='Test Acc', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

---

### Visualizing What CNNs Learn (1 minute)

**"Peek inside the black box"**

```python
import torch
import matplotlib.pyplot as plt

# Get first convolutional layer filters
model.eval()
first_layer = model.conv1
filters = first_layer.weight.data.cpu()

print(f"First layer filters shape: {filters.shape}")  # (32, 3, 3, 3)
print("32 filters, each 3x3 for RGB input")

# Visualize first 16 filters
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
axes = axes.flatten()

for idx in range(32):
    # Get filter and normalize for visualization
    filter_img = filters[idx]

    # If RGB, transpose to (H, W, C)
    if filter_img.shape[0] == 3:
        filter_img = filter_img.permute(1, 2, 0)

    # Normalize to [0, 1]
    filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())

    axes[idx].imshow(filter_img)
    axes[idx].axis('off')
    axes[idx].set_title(f'Filter {idx}')

plt.suptitle('First Layer Filters (Edge Detectors)', fontsize=16)
plt.tight_layout()
plt.show()

# Visualize activation maps
test_image, test_label = test_dataset[0]
test_image_batch = test_image.unsqueeze(0).to(device)

# Get activations from first conv layer
activation = first_layer(test_image_batch)
activation = activation.squeeze().detach().cpu()

print(f"\nActivation shape: {activation.shape}")  # (32, 32, 32)

# Visualize first 16 activation maps
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()

for idx in range(16):
    axes[idx].imshow(activation[idx], cmap='viridis')
    axes[idx].axis('off')
    axes[idx].set_title(f'Feature Map {idx}')

plt.suptitle('Activation Maps from First Conv Layer', fontsize=16)
plt.tight_layout()
plt.show()
```

---

## Key Takeaways

1. **CNNs vs Fully Connected**:
   - Fewer parameters (weight sharing)
   - Preserve spatial structure
   - Translation invariant
   - Learn hierarchical features

2. **Convolution Operation**:
   - Slide filter over image
   - Element-wise multiply and sum
   - Detects local patterns (edges, textures, shapes)

3. **nn.Conv2d Parameters**:
   - `in_channels`: Input depth
   - `out_channels`: Number of filters
   - `kernel_size`: Filter size (3x3, 5x5, etc.)
   - `stride`: Step size
   - `padding`: Border padding

4. **Pooling**:
   - **MaxPool**: Takes maximum (preserves strong features)
   - **AvgPool**: Takes average (smooths features)
   - Reduces spatial dimensions
   - Provides translation invariance

5. **CNN Architecture Pattern**:
   ```
   Input → [Conv → ReLU → Pool] × N → Flatten → FC → Output
   ```

6. **Feature Hierarchy**:
   - Early layers: Edges, colors, textures
   - Middle layers: Shapes, patterns
   - Later layers: High-level objects

---

## Today's Practice Exercise

**Build and train your own CNN**

```python
# YOUR TASK:
# 1. Build a CNN with 4 convolutional blocks
# 2. Add batch normalization after each conv layer
# 3. Train on FashionMNIST or CIFAR-10
# 4. Achieve >85% test accuracy
# 5. Visualize filters and activation maps
# 6. Compare with fully connected network

import torch
import torch.nn as nn

class YourCNN(nn.Module):
    def __init__(self):
        super(YourCNN, self).__init__()
        # YOUR CODE HERE
        pass

    def forward(self, x):
        # YOUR CODE HERE
        pass

# Train and compare!
```

---

## Tomorrow's Preview

**Day 9: Advanced CNN Architectures and Techniques**

- VGG blocks and deep networks
- Residual connections (ResNets)
- Inception modules
- Depthwise separable convolutions
- Best practices for CNN design

---

**"You now understand how CNNs work! Tomorrow we'll explore advanced architectures used in production. 🚀"**
