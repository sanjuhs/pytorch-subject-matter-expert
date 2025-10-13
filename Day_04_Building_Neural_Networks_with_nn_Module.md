# Day 4: Building Neural Networks with nn.Module

**Duration**: 10 minutes
**Platform**: Google Colab

---

## Video Script

### Introduction (30 seconds)

**"Welcome to Day 4 - where we learn to build neural networks the PyTorch way!"**

Yesterday we built neural networks manually with raw tensors. Today we'll learn PyTorch's **`nn.Module`** - the foundation for building any neural network architecture.

We'll cover:
- What is `nn.Module` and why use it
- Building custom neural network classes
- Using built-in layers (`nn.Linear`, `nn.ReLU`, etc.)
- The `forward()` method
- Sequential models for quick prototyping

By the end, you'll be writing clean, professional PyTorch code.

---

### What is nn.Module? (1 minute)

**"nn.Module is the base class for all neural network components"**

```python
import torch
import torch.nn as nn

# The old way (Day 3) - manual parameter management
W1 = torch.randn(10, 20, requires_grad=True)
b1 = torch.randn(20, requires_grad=True)
W2 = torch.randn(20, 5, requires_grad=True)
b2 = torch.randn(5, requires_grad=True)

print("Manual approach:")
print(f"- Need to track 4 separate variables")
print(f"- Need to manually zero gradients for each")
print(f"- Need to manually update each parameter")

# The PyTorch way - using nn.Module
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = SimpleNet()
print("\nnn.Module approach:")
print(f"- All parameters tracked automatically")
print(f"- Can call model.parameters() to get all params")
print(f"- Can call model.zero_grad() to zero all grads")
print(f"- Clean, organized, reusable!")

# Access all parameters
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
```

**Benefits of nn.Module:**
- Automatic parameter tracking
- Built-in methods (`.parameters()`, `.zero_grad()`, `.train()`, `.eval()`)
- Easy to save and load models
- Composable (modules can contain modules)
- Industry standard

---

### Your First nn.Module Class (2 minutes)

**"Let's build a neural network class step by step"**

```python
import torch
import torch.nn as nn

class MyFirstNetwork(nn.Module):
    """
    A simple 2-layer neural network
    Input: 784 features (28x28 image flattened)
    Hidden: 128 neurons
    Output: 10 classes (digits 0-9)
    """

    def __init__(self):
        # Always call parent constructor first
        super(MyFirstNetwork, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(784, 128)  # First fully connected layer
        self.relu = nn.ReLU()            # Activation function
        self.fc2 = nn.Linear(128, 10)   # Output layer

    def forward(self, x):
        """
        Defines the forward pass
        x: input tensor of shape (batch_size, 784)
        """
        x = self.fc1(x)      # Linear transformation
        x = self.relu(x)     # Apply ReLU activation
        x = self.fc2(x)      # Output layer
        return x             # Return logits (no activation)

# Create instance
model = MyFirstNetwork()
print(model)

# Test with random input
batch_size = 4
x = torch.randn(batch_size, 784)
output = model(x)  # Calls forward() automatically

print(f"\nInput shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output:\n{output}")

# Inspect parameters
print(f"\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

**Key concepts:**
1. `__init__()`: Define your layers/modules here
2. `forward()`: Define how data flows through your network
3. Calling `model(x)` automatically calls `forward(x)`
4. Don't define `backward()` - autograd handles it!

---

### Understanding nn.Linear (1.5 minutes)

**"Let's understand what nn.Linear actually does"**

```python
# Manual linear layer (Day 3 style)
x = torch.randn(3, 4)  # 3 samples, 4 features
W = torch.randn(4, 2, requires_grad=True)
b = torch.randn(2, requires_grad=True)

manual_output = x @ W + b
print("Manual linear layer output:")
print(manual_output)

# Using nn.Linear
linear = nn.Linear(in_features=4, out_features=2)
nn_output = linear(x)
print("\nnn.Linear output:")
print(nn_output)

# Inspect nn.Linear parameters
print(f"\nnn.Linear weight shape: {linear.weight.shape}")  # (out, in)
print(f"nn.Linear bias shape: {linear.bias.shape}")

# IMPORTANT: nn.Linear stores weight as (out_features, in_features)
# It computes: output = x @ W.T + b (note the transpose!)

# Verify this
with torch.no_grad():
    # Set our manual W to match nn.Linear's transposed weight
    W = linear.weight.T.clone()
    b = linear.bias.clone()

manual_output = x @ W + b
print(f"\nOutputs match: {torch.allclose(manual_output, nn_output)}")
```

**nn.Linear parameters:**
- `in_features`: number of input features
- `out_features`: number of output features
- `bias=True`: whether to include bias term (default True)

**Behind the scenes:**
```
y = x @ W.T + b
where W.shape = (out_features, in_features)
```

---

### Building a Complete Model for MNIST (2 minutes)

**"Let's build a real model to classify handwritten digits"**

```python
import torch
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10):
        super(MNISTClassifier, self).__init__()

        # Build layers dynamically
        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer (no activation - we'll use CrossEntropyLoss later)
        layers.append(nn.Linear(prev_size, num_classes))

        # Combine into sequential container
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten image if needed: (batch, 28, 28) -> (batch, 784)
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        return self.network(x)

# Create model
model = MNISTClassifier(hidden_sizes=[256, 128])
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test with sample data
batch_images = torch.randn(32, 28, 28)  # 32 images of 28x28
logits = model(batch_images)

print(f"\nInput shape: {batch_images.shape}")
print(f"Output shape: {logits.shape}")  # (32, 10)
print(f"\nSample logits for first image: {logits[0]}")

# Get predicted class
predictions = logits.argmax(dim=1)
print(f"Predicted classes for batch: {predictions}")
```

---

### nn.Sequential: Quick Model Building (1.5 minutes)

**"For simple architectures, use nn.Sequential"**

```python
import torch.nn as nn

# Method 1: Sequential with ordered arguments
model1 = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

print("Method 1: Basic Sequential")
print(model1)

# Method 2: Sequential with OrderedDict (named layers)
from collections import OrderedDict

model2 = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 128)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(128, 10))
]))

print("\nMethod 2: Named Sequential")
print(model2)

# Access layers by index or name
print(f"\nFirst layer (by index): {model1[0]}")
print(f"First layer (by name): {model2.fc1}")

# Forward pass
x = torch.randn(16, 784)
output = model1(x)
print(f"\nOutput shape: {output.shape}")

# When to use Sequential vs custom nn.Module?
# Sequential: Simple feed-forward architectures
# Custom nn.Module: Complex architectures with skip connections, multiple inputs/outputs
```

---

### Adding Dropout and Batch Normalization (1.5 minutes)

**"Make your networks more robust with regularization"**

```python
import torch
import torch.nn as nn

class ImprovedNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10, dropout_prob=0.5):
        super(ImprovedNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Normalize layer outputs
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)  # Randomly zero neurons
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)      # Normalize
        x = self.relu(x)
        x = self.dropout(x)  # Dropout (only active during training)
        x = self.fc2(x)
        return x

model = ImprovedNet()
print(model)

# Training vs Evaluation mode
model.train()  # Enable dropout and batch norm updates
x_train = torch.randn(32, 784)
out_train = model(x_train)
print(f"\nTraining mode output (with dropout): {out_train[0]}")

model.eval()   # Disable dropout, use running stats for batch norm
with torch.no_grad():
    out_eval = model(x_train)
    print(f"Evaluation mode output (no dropout): {out_eval[0]}")

# IMPORTANT: Always call model.eval() before inference!
```

**What these layers do:**
- **BatchNorm1d**: Normalizes activations (mean=0, std=1) → faster training, less sensitive to initialization
- **Dropout**: Randomly sets neurons to 0 during training → prevents overfitting
- **`.train()` vs `.eval()`**: Changes behavior of dropout and batch norm

---

### Complete Training Example (2 minutes)

**"Let's put everything together: model + data + training"**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data (we'll use real MNIST tomorrow)
torch.manual_seed(42)
X_train = torch.randn(1000, 784)  # 1000 samples
y_train = torch.randint(0, 10, (1000,))  # Random labels 0-9

# Define model
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)

# Initialize model
model = SimpleClassifier()
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

print("Starting training...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
epochs = 50
batch_size = 32

for epoch in range(epochs):
    model.train()  # Set to training mode
    epoch_loss = 0
    correct = 0
    total = 0

    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        # Get batch
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters

        # Track metrics
        epoch_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    # Print progress
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / (len(X_train) // batch_size)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

print("\nTraining complete!")

# Evaluation
model.eval()  # Set to evaluation mode
with torch.no_grad():
    test_outputs = model(X_train[:100])
    test_predictions = test_outputs.argmax(dim=1)
    test_accuracy = (test_predictions == y_train[:100]).float().mean()
    print(f"Test accuracy on 100 samples: {test_accuracy.item():.2%}")
```

---

### Model Inspection and Debugging (1 minute)

**"Useful methods for understanding your model"**

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 1. Print model architecture
print("Model architecture:")
print(model)

# 2. Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 3. Inspect each layer
print("\nLayer details:")
for idx, (name, param) in enumerate(model.named_parameters()):
    print(f"{idx}: {name:15s} {str(param.shape):20s} {param.numel():>8,} params")

# 4. Check model state
print(f"\nModel in training mode: {model.training}")
model.eval()
print(f"Model in training mode: {model.training}")

# 5. Get specific layer
first_layer = model[0]
print(f"\nFirst layer weights shape: {first_layer.weight.shape}")
print(f"First layer bias shape: {first_layer.bias.shape}")

# 6. Hook to inspect intermediate outputs (advanced)
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model[0].register_forward_hook(get_activation('layer1'))
model[2].register_forward_hook(get_activation('layer2'))

# Forward pass
x = torch.randn(4, 784)
output = model(x)

print(f"\nIntermediate activations:")
print(f"Layer 1 output shape: {activations['layer1'].shape}")
print(f"Layer 2 output shape: {activations['layer2'].shape}")
```

---

## Key Takeaways

1. **nn.Module** - Base class for all PyTorch models
2. **`__init__()`** - Define layers and submodules
3. **`forward()`** - Define the forward pass (backward is automatic!)
4. **nn.Linear** - Fully connected layer: `y = x @ W.T + b`
5. **nn.Sequential** - Quick way to build simple models
6. **model.train()** vs **model.eval()** - Control dropout and batch norm behavior
7. **model.parameters()** - Access all learnable parameters
8. **Dropout & BatchNorm** - Regularization techniques to prevent overfitting

---

## Today's Practice Exercise

**Build a classifier for the FashionMNIST dataset**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Download FashionMNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                       download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define your model
class FashionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

# Initialize
model = FashionClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 200 == 199:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/200:.3f}')
            running_loss = 0.0

print('Training complete!')

# Test your model
model.eval()
correct = 0
total = 0

test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                      download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')
```

---

## Tomorrow's Preview

**Day 5: Loss Functions and Optimizers**

- Understanding different loss functions (MSE, CrossEntropy, etc.)
- How optimizers work (SGD, Adam, RMSprop)
- Learning rate scheduling
- Gradient clipping
- Best practices for training

---

**"You now know how to build professional neural networks with PyTorch! Tomorrow we'll learn how to train them effectively. 🚀"**
