import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time

# --- Data (identical to Week 15 Session 5 — no changes here) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),   # CIFAR-10 mean per channel
                         (0.2470, 0.2435, 0.2616))   # CIFAR-10 std per channel
])

train_dataset = datasets.CIFAR10(root='data/', train=True,  download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root='data/', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape)   # → torch.Size([64, 3, 32, 32])
print(labels.shape)   # → torch.Size([64])

# --- Model ---
# Same architecture as week15/session5_cifar.py, ONE change: BatchNorm2d
# inserted after each Conv2d, before ReLU. This is the only variable we're
# changing this session — everything else (filters, Linear sizes, Dropout,
# lr, epochs) stays identical so the before/after comparison is clean.
#
# Placement rule: Conv2d → BatchNorm2d → ReLU → (MaxPool2d)
# BatchNorm normalizes the conv output's activations (per-channel mean/var
# across the batch) BEFORE the nonlinearity clips negative values. Doing it
# after ReLU would normalize an already-truncated distribution, which
# defeats the point.
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),                            # num_features = out_channels of the conv above
    nn.ReLU(),
    nn.MaxPool2d(2),                               # 32×32 → 16×16

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),                               # 16×16 → 8×8

    nn.Flatten(),                                  # 64 × 8 × 8 = 4096, same as Week 15

    nn.Linear(4096, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            preds   = torch.argmax(model(images), dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs    = 15

best_test_acc = 0.0

for epoch in range(epochs):
    # model.train() now matters for TWO layer types, not one:
    #   - Dropout: randomly zeroes activations
    #   - BatchNorm2d: uses CURRENT batch statistics (mean/var), and
    #     simultaneously updates its running_mean/running_var buffers,
    #     which is what eval() mode will use instead.
    # Skip this call and BatchNorm silently keeps using stale/default
    # running stats — same "silent failure" shape as the Dropout bug
    # from Week 15, different mechanism.
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
    train_acc  = evaluate(model, train_loader)   # evaluate() already calls model.eval() internally
    test_acc   = evaluate(model, test_loader)

    print(f"Epoch {epoch+1:2d} | loss: {avg_loss:.4f} | train: {train_acc:.4f} | test: {test_acc:.4f}")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'models/best_model_week16_session1.pth')

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Best test accuracy: {best_test_acc:.4f}")