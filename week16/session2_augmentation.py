import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time

# --- Data ---
# TRAIN transform gets augmentation added: RandomCrop + RandomHorizontalFlip,
# applied BEFORE ToTensor/Normalize (torchvision's PIL-based transforms expect
# a PIL image, not a tensor, so order matters here).
#
# TEST transform is UNCHANGED from Session 1 — no augmentation. We need to
# measure generalization to real, unaltered images, not to augmented ones.
# Augmenting the test set would change what we're measuring.
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # pad 4px, crop back to 32x32 -> random shift
    transforms.RandomHorizontalFlip(),             # 50% chance mirror left-right
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
test_dataset  = datasets.CIFAR10(root='data/', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)

# --- Model ---
# Identical to week16/session1_batchnorm.py. BatchNorm stays in — it's now
# an established improvement, not the thing being tested this session.
# Augmentation is the only new variable.
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

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

# NOTE: with augmentation on, train_acc is measured against RANDOMLY
# AUGMENTED images each epoch (since train_loader re-applies the transform
# every time it's iterated), not a fixed set. This makes train_acc a noisier,
# slightly pessimistic signal than before — expect it to run lower than
# Session 1 even at equal "true" model quality. test_acc remains the clean,
# comparable number across all sessions.
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

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'models/best_model_week16_session2.pth')

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Best test accuracy: {best_test_acc:.4f}")