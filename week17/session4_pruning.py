import os
import time

import torch
import torch.nn as nn
import torch_pruning as tp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_size_kb(model, tmp_path):
    torch.save(model.state_dict(), tmp_path)
    size_kb = os.path.getsize(tmp_path) / 1024
    os.remove(tmp_path)
    return size_kb


def evaluate(model, test_loader):
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = torch.argmax(model(images), dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def benchmark_throughput(model, batch_size=64, n_warmup_batches=5, n_bench_batches=50):
    bench_batch = torch.randn(batch_size, 3, 32, 32)

    with torch.no_grad():
        for _ in range(n_warmup_batches):
            model(bench_batch)

        start = time.perf_counter()
        for _ in range(n_bench_batches):
            model(bench_batch)
        elapsed = time.perf_counter() - start

    total_images = batch_size * n_bench_batches
    return total_images / elapsed


def report(label, model, test_loader, tmp_path):
    acc = evaluate(model, test_loader)
    throughput = benchmark_throughput(model)
    size_kb = get_model_size_kb(model, tmp_path)
    print(f"{label}: accuracy={acc:.4f}  throughput={throughput:.1f} img/s  size={size_kb:.1f} KB")
    return acc, throughput, size_kb


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root='data/', train=True,  download=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.CIFAR10(root='data/', train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10),
)
model.load_state_dict(torch.load('models/best_model_week16_session3.pth', map_location='cpu'))
model.eval()

# ---------------------------------------------------------------------------
# Checkpoint 1: original FP32
# ---------------------------------------------------------------------------

report("Original FP32", model, test_loader, "temp_fp32.pth")

# ---------------------------------------------------------------------------
# Prune 10% of channels
# ---------------------------------------------------------------------------

example_inputs = torch.randn(1, 3, 32, 32)
DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
importance = tp.importance.MagnitudeImportance(p=2)

pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=importance,
    pruning_ratio=0.1,
    ignored_layers=[model[-1]],  # keep final Linear(512, 10) intact
)
pruner.step()

# ---------------------------------------------------------------------------
# Checkpoint 2: pruned, no fine-tuning
# ---------------------------------------------------------------------------

report("Pruned (no fine-tuning)", model, test_loader, "temp_pruned.pth")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs    = 10
best_test_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        logits = model(images)
        loss   = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss   = running_loss / len(train_loader)
    train_acc  = evaluate(model, train_loader)
    test_acc   = evaluate(model, test_loader)

    print(f"Epoch {epoch+1:2d} | loss: {avg_loss:.4f} | train: {train_acc:.4f} | test: {test_acc:.4f}")

report("Pruned (fine-tuning)", model, test_loader, "temp_pruned_fine_tune.pth")