# Day 10: Transfer Learning and Fine-Tuning

**Duration**: 10 minutes
**Platform**: Google Colab

---

## Video Script

### Introduction (30 seconds)

**"Welcome to Day 10 - where we learn to stand on the shoulders of giants!"**

Training deep networks from scratch is slow and requires huge datasets. Transfer learning lets you leverage pre-trained models:
- Using pre-trained models from torchvision
- Feature extraction vs fine-tuning strategies
- When and how to freeze layers
- Fine-tuning best practices
- Adapting models to new domains

By the end, you'll train state-of-the-art models in minutes, not days.

---

### What is Transfer Learning? (1 minute)

**"Reuse knowledge learned from large datasets"**

```python
import torch
import torch.nn as nn

# The problem: Training from scratch
print("Training from Scratch:")
print("❌ Requires millions of labeled images")
print("❌ Takes days/weeks on expensive GPUs")
print("❌ Needs lots of data to prevent overfitting")
print("❌ High computational cost")

# The solution: Transfer learning
print("\nTransfer Learning:")
print("✓ Start with model pre-trained on ImageNet (1.2M images, 1000 classes)")
print("✓ Adapt to your task (maybe only 1000 images!)")
print("✓ Train in minutes/hours, not days")
print("✓ Better performance with less data")
print("✓ Lower computational cost")

# Why does it work?
print("\nWhy Transfer Learning Works:")
print("Early layers learn general features:")
print("  - Edges, textures, colors (universal to all images)")
print("  - These transfer well to new tasks!")
print("Later layers learn task-specific features:")
print("  - Object parts, specific shapes")
print("  - Need to adapt these to your task")

# Two main strategies
print("\nTwo Strategies:")
print("\n1. Feature Extraction:")
print("   - Freeze pre-trained layers (no training)")
print("   - Only train new classifier head")
print("   - Fast, prevents overfitting")
print("   - Use when: Small dataset, similar domain")

print("\n2. Fine-Tuning:")
print("   - Unfreeze some/all pre-trained layers")
print("   - Train entire network (lower learning rate)")
print("   - Better performance, requires more data")
print("   - Use when: Larger dataset, different domain")
```

---

### Loading Pre-trained Models (1.5 minutes)

**"torchvision provides models trained on ImageNet"**

```python
import torch
import torchvision.models as models

# Available pre-trained models
print("Popular Pre-trained Models:")
print("- ResNet: resnet18, resnet34, resnet50, resnet101, resnet152")
print("- VGG: vgg11, vgg13, vgg16, vgg19")
print("- MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large")
print("- EfficientNet: efficientnet_b0 to efficientnet_b7")
print("- Vision Transformer: vit_b_16, vit_b_32")
print("- And many more!")

# Load a pre-trained ResNet-18
print("\nLoading pre-trained ResNet-18...")
model = models.resnet18(pretrained=True)  # Or weights='DEFAULT' in newer versions
print(model)

# Model architecture
print("\nResNet-18 architecture:")
print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"- Input: 224x224 RGB images")
print(f"- Output: 1000 class logits (ImageNet classes)")

# Inspect the model structure
print("\nKey components:")
print(f"- conv1: {model.conv1}")
print(f"- fc (classifier): {model.fc}")

# The classifier layer
print(f"\nClassifier details:")
print(f"- Input features: {model.fc.in_features}")
print(f"- Output classes: {model.fc.out_features}")

# Test with random input
test_input = torch.randn(1, 3, 224, 224)  # ImageNet size
output = model(test_input)
print(f"\nInput shape: {test_input.shape}")
print(f"Output shape: {output.shape}")  # (1, 1000)

# Get predictions
probs = torch.nn.functional.softmax(output, dim=1)
top5_prob, top5_idx = torch.topk(probs, 5)
print(f"\nTop 5 predictions:")
for i in range(5):
    print(f"  Class {top5_idx[0][i].item()}: {top5_prob[0][i].item():.4f}")

# Load without pre-trained weights
print("\nFor comparison, untrained model:")
untrained_model = models.resnet18(pretrained=False)
print("Same architecture, random weights")
print("Would need to train from scratch on ImageNet (weeks of GPU time!)")
```

---

### Strategy 1: Feature Extraction (2 minutes)

**"Freeze the backbone, train only the classifier"**

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Load pre-trained ResNet-18
model = models.resnet18(pretrained=True)

# Freeze all layers
print("Freezing all pre-trained layers...")
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier for our task (e.g., 10 classes)
num_classes = 10
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

print(f"\nReplaced classifier:")
print(f"- Old: Linear({num_features}, 1000)")
print(f"- New: Linear({num_features}, {num_classes})")

# Check which parameters are trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"\nParameter count:")
print(f"- Total: {total_params:,}")
print(f"- Trainable: {trainable_params:,}")
print(f"- Frozen: {total_params - trainable_params:,}")
print(f"- Trainable %: {100 * trainable_params / total_params:.2f}%")

# Verify
print("\nTrainable layers:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")

# Setup for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Optimizer: Only optimize trainable parameters
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
# Note: model.parameters() would also work, but explicit is better

criterion = nn.CrossEntropyLoss()

print("\nReady for training!")
print("Only the new classifier will be updated")
print("Pre-trained features stay fixed → Fast training!")

# Training loop (pseudo-code)
print("\nTypical training:")
print("""
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()  # Gradients only for unfrozen layers
    optimizer.step()
""")

# When to use feature extraction
print("\nUse Feature Extraction When:")
print("✓ Small dataset (< 10k images)")
print("✓ Similar domain to ImageNet (natural images)")
print("✓ Need fast training")
print("✓ Limited compute resources")
print("✓ Want to prevent overfitting")
```

---

### Strategy 2: Fine-Tuning (2 minutes)

**"Update pre-trained weights with lower learning rate"**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Load pre-trained model
model = models.resnet18(pretrained=True)

# Replace classifier
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Strategy 2a: Fine-tune ALL layers
print("Strategy 2a: Fine-tune entire network")
print("All parameters are trainable (no freezing)")

# Use different learning rates for different parts
# Lower LR for pre-trained layers, higher for new classifier
optimizer = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-5},
    {'params': model.bn1.parameters(), 'lr': 1e-5},
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}  # Highest LR for new layer
])

print("\nLearning rates:")
print("- Early layers (conv1, layer1-2): 1e-5 (very small)")
print("- Middle layers (layer3-4): 1e-4")
print("- Classifier (fc): 1e-3 (larger)")
print("\nRationale: Early features are more general, change them slowly")

# Strategy 2b: Gradual unfreezing
print("\n" + "="*50)
print("Strategy 2b: Gradual unfreezing")
print("="*50)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Phase 1: Train only classifier (like feature extraction)
print("\nPhase 1: Train classifier only (5 epochs)")
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)  # Unfreeze classifier

optimizer_phase1 = optim.Adam(model.fc.parameters(), lr=1e-3)

# After 5 epochs...

# Phase 2: Unfreeze layer4
print("Phase 2: Unfreeze layer4 (5 epochs)")
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer_phase2 = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

# After 5 more epochs...

# Phase 3: Unfreeze all
print("Phase 3: Fine-tune all layers (10 epochs)")
for param in model.parameters():
    param.requires_grad = True

optimizer_phase3 = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-5},
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

print("\nGradual unfreezing benefits:")
print("✓ Prevents catastrophic forgetting")
print("✓ More stable training")
print("✓ Often better final performance")

# When to use fine-tuning
print("\nUse Fine-Tuning When:")
print("✓ Medium to large dataset (> 10k images)")
print("✓ Different domain from ImageNet")
print("✓ Need best possible performance")
print("✓ Have sufficient compute resources")
```

---

### Complete Transfer Learning Example: CIFAR-10 (2.5 minutes)

**"End-to-end transfer learning pipeline"**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# 1. Data preparation
print("Step 1: Prepare data")

# IMPORTANT: Use ImageNet normalization for pre-trained models!
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),  # ImageNet input size
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                  transform=train_transform)
test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                                 transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                          num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,
                         num_workers=2, pin_memory=True)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# 2. Load pre-trained model
print("\nStep 2: Load pre-trained ResNet-18")
model = models.resnet18(pretrained=True)

# 3. Modify classifier
print("Step 3: Modify classifier for CIFAR-10")
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 4. Freeze backbone (feature extraction mode)
print("Step 4: Freeze backbone")
for name, param in model.named_parameters():
    if 'fc' not in name:  # Freeze all except classifier
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")

# 5. Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nStep 5: Setup training on {device}")

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 6. Training loop
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):
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

    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), 100 * correct / total

# Train
print("\nStep 6: Train model")
epochs = 10

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")

    train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                         optimizer, device)
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    scheduler.step()

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")

print("\nFeature extraction training complete!")

# 7. Fine-tuning phase
print("\n" + "="*50)
print("Step 7: Fine-tuning (unfreeze all layers)")
print("="*50)

# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Lower learning rates for fine-tuning
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Fine-tune for additional epochs
for epoch in range(5):
    print(f"\nFine-tuning Epoch {epoch+1}/5")

    train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                         optimizer, device)
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    scheduler.step()

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")

print("\nFine-tuning complete!")
print("Transfer learning achieved high accuracy with minimal training!")
```

---

### Choosing the Right Model (1 minute)

**"Different models for different needs"**

```python
import torchvision.models as models

print("Model Selection Guide:\n")

# Accuracy vs Speed vs Size
models_comparison = [
    ("ResNet-18", "Fast", "Small", "Good", "General purpose, great starting point"),
    ("ResNet-50", "Medium", "Medium", "Better", "More capacity, still efficient"),
    ("ResNet-152", "Slow", "Large", "Best", "Maximum accuracy, very deep"),
    ("MobileNet-V2", "Very Fast", "Very Small", "Good", "Mobile deployment, edge devices"),
    ("EfficientNet-B0", "Fast", "Small", "Very Good", "Best accuracy/size trade-off"),
    ("EfficientNet-B7", "Slow", "Large", "Best", "State-of-the-art accuracy"),
    ("VGG-16", "Slow", "Very Large", "Good", "Large memory footprint, avoid"),
    ("ViT-B/16", "Medium", "Medium", "Excellent", "Transformer-based, needs more data"),
]

print(f"{'Model':<18} {'Speed':<12} {'Size':<14} {'Accuracy':<10} {'Use Case'}")
print("="*90)

for model_name, speed, size, accuracy, use_case in models_comparison:
    print(f"{model_name:<18} {speed:<12} {size:<14} {accuracy:<10} {use_case}")

print("\n" + "="*90)
print("Recommendations:")
print("="*90)

print("\n🎯 General Classification:")
print("   → Start with ResNet-18 or ResNet-50")
print("   → Try EfficientNet-B0 for better efficiency")

print("\n📱 Mobile/Edge Deployment:")
print("   → MobileNet-V2 or MobileNet-V3")
print("   → EfficientNet-Lite variants")

print("\n🎨 Fine-grained Classification (birds, flowers):")
print("   → Larger models: ResNet-101, EfficientNet-B3+")
print("   → More capacity for subtle differences")

print("\n⚡ Real-time Applications:")
print("   → MobileNet or small EfficientNet")
print("   → Consider quantization/pruning")

print("\n🏆 Maximum Accuracy (competitions):")
print("   → EfficientNet-B7 or ViT-Large")
print("   → Ensemble multiple models")

# Example: Loading different models
print("\n" + "="*90)
print("Loading Different Models:")
print("="*90)

# ResNet
resnet18 = models.resnet18(pretrained=True)
resnet50 = models.resnet50(pretrained=True)

# MobileNet
mobilenet = models.mobilenet_v2(pretrained=True)

# EfficientNet (requires torchvision >= 0.11)
try:
    efficientnet = models.efficientnet_b0(pretrained=True)
except:
    print("EfficientNet requires torchvision >= 0.11")

# Vision Transformer
vit = models.vit_b_16(pretrained=True)

print("All models loaded successfully!")
```

---

### Advanced: Custom Pre-training and Domain Adaptation (1 minute)

**"When ImageNet isn't enough"**

```python
import torch
import torch.nn as nn

print("When ImageNet Pre-training Isn't Enough:\n")

# Scenario 1: Very different domain
print("1. Medical Images (X-rays, CT scans):")
print("   → ImageNet (natural images) may not help much")
print("   → Consider pre-training on medical datasets")
print("   → Or train from scratch if you have enough data")

# Scenario 2: Different image characteristics
print("\n2. Satellite/Aerial Images:")
print("   → Different perspective than ImageNet")
print("   → Pre-train on satellite datasets (e.g., ImageNet-S)")
print("   → Or use self-supervised learning")

# Scenario 3: Non-RGB images
print("\n3. Multi-spectral/Hyperspectral Images:")
print("   → More than 3 channels")
print("   → Modify first conv layer:")

model = models.resnet18(pretrained=True)

# Original: 3 input channels
print(f"\nOriginal conv1: {model.conv1}")

# Adapt for 5-channel input (e.g., RGB + IR + depth)
new_conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Initialize: Copy RGB weights, random for extra channels
with torch.no_grad():
    new_conv1.weight[:, :3, :, :] = model.conv1.weight
    new_conv1.weight[:, 3:, :, :] = model.conv1.weight.mean(dim=1, keepdim=True)

model.conv1 = new_conv1
print(f"Modified conv1: {model.conv1}")

# Self-supervised pre-training
print("\n4. Self-supervised Learning:")
print("   → Pre-train on unlabeled data from your domain")
print("   → Methods: SimCLR, MoCo, BYOL, SwAV")
print("   → Then fine-tune on labeled data")

# Domain adaptation techniques
print("\n5. Domain Adaptation:")
print("   → Discriminative learning rates (lower for early layers)")
print("   → Gradual unfreezing")
print("   → Data augmentation specific to target domain")
print("   → Mix source and target data during training")

print("\nKey Principle:")
print("The more different your domain, the more you need to adapt!")
```

---

## Key Takeaways

1. **Transfer Learning Benefits**:
   - Train faster with less data
   - Better performance than training from scratch
   - Leverage knowledge from large datasets

2. **Two Strategies**:
   - **Feature Extraction**: Freeze backbone, train classifier only
   - **Fine-Tuning**: Update all layers with lower learning rates

3. **Best Practices**:
   - Use ImageNet normalization for pre-trained models
   - Lower learning rates for pre-trained layers
   - Gradual unfreezing for stability
   - Different LRs for different layers

4. **Model Selection**:
   - ResNet: Best general-purpose choice
   - MobileNet: Fast inference for deployment
   - EfficientNet: Best accuracy/size trade-off

5. **When to Use Each**:
   - Small dataset + similar domain → Feature extraction
   - Large dataset + different domain → Fine-tuning
   - Very different domain → Consider custom pre-training

---

## Today's Practice Exercise

**Transfer learning on a custom dataset**

```python
# YOUR TASK:
# 1. Download a dataset (e.g., Caltech-101, Food-101)
# 2. Load pre-trained ResNet-50
# 3. Try both feature extraction and fine-tuning
# 4. Compare performance and training time
# 5. Visualize what the pre-trained model sees
# 6. Try different architectures (MobileNet, EfficientNet)

# Bonus: Implement gradual unfreezing
# Bonus: Try different learning rate schedules
```

---

## Tomorrow's Preview

**Day 11: Batch Normalization and Regularization Techniques**

- Deep dive into Batch Normalization
- Layer Norm, Group Norm, Instance Norm
- Dropout variants
- Data augmentation strategies
- Weight initialization
- Preventing overfitting

---

**"You now know how to leverage pre-trained models! Tomorrow we'll master normalization and regularization. 🚀"**
