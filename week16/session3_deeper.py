import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time

# --- Data ---
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
test_dataset  = datasets.CIFAR10(root='data/', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)

# --- Model ---
# Spatial dimension trace:
#   Input:        32x32, 3 channels
#   Conv1+Pool:   32x32 -> 16x16, 32 channels
#   Conv2+Pool:   16x16 ->  8x8,  64 channels
#   Conv3+Pool:    8x8  ->  4x4,  128 channels
#   Flatten:      128 * 4 * 4 = 2048
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
    nn.Linear(512, 10)
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
        torch.save(model.state_dict(), 'models/best_model_week16_session3.pth')

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Best test accuracy: {best_test_acc:.4f}")