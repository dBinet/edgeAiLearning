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
    nn.Conv2d(1, 32, kernel_size=3, padding=1),    # doubled filters
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),   # doubled filters
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),                                   # → (batch, 64×7×7) = (batch, 3136)

    nn.Linear(3136, 256),                           # bigger linear layer
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
epochs    = 10

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

    avg_train_loss   = running_loss / len(train_loader)

    # --- measure test loss ---
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            test_loss += criterion(model(images), labels).item()
    avg_test_loss = test_loss / len(test_loader)
    model.train()   # ← don't forget this

    train_acc  = evaluate(model, train_loader)
    test_acc   = evaluate(model, test_loader)

    print(f"Epoch {epoch+1:2d} | train loss: {avg_train_loss:.4f} | test loss: {avg_test_loss:.4f} | train acc: {train_acc:.4f} | test acc: {test_acc:.4f}")

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")