import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='data/', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='data/', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# --- Model ---
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),   # → (batch, 16, 28, 28)
    nn.ReLU(),
    nn.MaxPool2d(2),                               # → (batch, 16, 14, 14)

    nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → (batch, 32, 14, 14)
    nn.ReLU(),
    nn.MaxPool2d(2),                               # → (batch, 32,  7,  7)

    nn.Flatten(),                                  # → (batch, 1568)

    nn.Linear(1568, 128),
    nn.ReLU(),
    nn.Linear(128, 10)                             # 10 classes, no activation
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
epochs    = 5

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

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")