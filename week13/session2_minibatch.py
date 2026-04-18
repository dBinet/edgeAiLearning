import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import time

# ── Data ──────────────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.FloatTensor(X_train)
X_test_t  = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)

# TensorDataset pairs X and y so they stay aligned when shuffled
# DataLoader handles the splitting into batches + shuffling each epoch
dataset = TensorDataset(X_train_t, y_train_t)

# ── Model factory ─────────────────────────────────────────────────────────────
def make_model():
    return nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )

# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate(model):
    with torch.no_grad():
        pred = model(X_test_t)
        predicted = (pred >= 0.5).float()
        return (predicted == y_test_t).float().mean().item()

# ── Training function ─────────────────────────────────────────────────────────
def train(model, loader, lr, epochs, label):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'─'*50}")
    print(f"{label}")
    print(f"{'─'*50}")

    start = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0

        # This loop is new — in full-batch there was no inner loop
        # Each iteration: one batch, one forward pass, one backward pass, one update
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average loss across all batches in this epoch
        avg_loss = epoch_loss / len(loader)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

    elapsed = time.time() - start
    acc = evaluate(model)
    print(f"\nAccuracy: {acc:.4f} | Time: {elapsed:.2f}s")
    return acc

# ── Experiment ────────────────────────────────────────────────────────────────
EPOCHS = 50  # 50 epochs mini-batch ≈ many more updates than 1000 epochs full-batch

# Batch size 32 — the most common default in practice
loader_32  = DataLoader(dataset, batch_size=32,  shuffle=True)
# Batch size 128 — fewer, larger steps
loader_128 = DataLoader(dataset, batch_size=128, shuffle=True)
# Batch size 800 — this IS full-batch (all training data at once)
loader_full = DataLoader(dataset, batch_size=800, shuffle=False)

print(f"Training samples: 800")
print(f"Batch size 32  → {len(loader_32)} updates per epoch")
print(f"Batch size 128 → {len(loader_128)} updates per epoch")
print(f"Batch size 800 → {len(loader_full)} update per epoch (full-batch)")

train(make_model(), loader_32,   lr=0.001, epochs=EPOCHS, label="Mini-batch 32  | lr=0.001 (Adam default)")
train(make_model(), loader_128,  lr=0.001, epochs=EPOCHS, label="Mini-batch 128 | lr=0.001")
train(make_model(), loader_full, lr=0.001, epochs=EPOCHS, label="Full-batch 800 | lr=0.001 (Week 12 finding: too slow)")
train(make_model(), loader_full, lr=0.01,  epochs=EPOCHS, label="Full-batch 800 | lr=0.01  (Week 12 fix)")