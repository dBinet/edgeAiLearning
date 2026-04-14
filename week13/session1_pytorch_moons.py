import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# Identical setup to Week 12 Session 1
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch works with tensors, not numpy arrays
# This is the equivalent of just having your numpy arrays — same data, different container
X_train_t = torch.FloatTensor(X_train)
X_test_t  = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)  # shape (800,1) — same as your (N,1) columns
y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)

# ── Model ─────────────────────────────────────────────────────────────────────
# nn.Sequential replaces your forward() function
# nn.Linear(in, out) replaces: Z = X @ W + b
# nn.ReLU() replaces: A = np.maximum(0, Z)
# nn.Sigmoid() replaces: output = 1 / (1 + np.exp(-Z))
model = nn.Sequential(
    nn.Linear(2, 16),   # W1: (2,16), b1: (16,)  ← He init is PyTorch's default for Linear
    nn.ReLU(),
    nn.Linear(16, 1),   # W2: (16,1), b2: (1,)
    nn.Sigmoid()
)

# ── Loss & Optimizer ──────────────────────────────────────────────────────────
# BCELoss = your binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
# Adam replaces your from-scratch Adam loop
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ── Training loop ─────────────────────────────────────────────────────────────
# This replaces your entire manual forward + backward + update block
for epoch in range(1000):
    # Forward pass — replaces: A1 = relu(X @ W1 + b1); output = sigmoid(A1 @ W2 + b2)
    y_pred = model(X_train_t)

    # Loss — replaces: loss = -np.mean(y*log(p) + (1-y)*log(1-p))
    loss = criterion(y_pred, y_train_t)

    # Backward pass — replaces your entire manual backprop block
    # optimizer.zero_grad()   # clear old gradients (they accumulate by default)
    loss.backward()         # computes ALL gradients via autograd — your chain rule, automated
    optimizer.step()        # W -= lr * update  (Adam formula applied internally)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

# ── Evaluation ────────────────────────────────────────────────────────────────
# torch.no_grad() = inference mode: don't track gradients (saves memory, same as your predict())
with torch.no_grad():
    test_pred = model(X_test_t)
    predicted = (test_pred >= 0.5).float()
    accuracy = (predicted == y_test_t).float().mean().item()
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Expected:      ~0.9800")