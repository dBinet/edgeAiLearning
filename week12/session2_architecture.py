# week12/session2_architecture.py
# Goal: Understand how width and depth affect a neural network's capacity

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────
# 1. DATASET (same as Session 1)
# ─────────────────────────────────────────
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
X_all   = scaler.transform(X)

# ─────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ─────────────────────────────────────────
# 3. 2-LAYER NETWORK (width experiment)
# ─────────────────────────────────────────
def train_2layer(X, y, n_hidden, lr=0.1, epochs=3000):
    np.random.seed(42)
    W1 = np.random.randn(X.shape[1], n_hidden) * np.sqrt(2.0 / X.shape[1])
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros((1, 1))

    for _ in range(epochs):
        # Forward
        Z1 = X @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = sigmoid(Z2)
        # Backward
        m  = X.shape[0]
        dZ2 = A2 - y
        dW2 = (1/m) * A1.T @ dZ2
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = (dZ2 @ W2.T) * relu_deriv(Z1)
        dW1 = (1/m) * X.T @ dZ1
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        # Update
        W1 -= lr * dW1;  b1 -= lr * db1
        W2 -= lr * dW2;  b2 -= lr * db2

    return W1, b1, W2, b2

def predict_2layer(X, W1, b1, W2, b2):
    A1 = relu(X @ W1 + b1)
    A2 = sigmoid(A1 @ W2 + b2)
    return (A2 >= 0.5).astype(int)

# ─────────────────────────────────────────
# 4. 3-LAYER NETWORK (depth experiment)
# ─────────────────────────────────────────
def train_3layer(X, y, n_hidden=16, lr=0.1, epochs=3000):
    np.random.seed(42)
    W1 = np.random.randn(X.shape[1], n_hidden) * np.sqrt(2.0 / X.shape[1])
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, n_hidden) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros((1, n_hidden))
    W3 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
    b3 = np.zeros((1, 1))

    for _ in range(epochs):
        # Forward
        Z1 = X @ W1 + b1;   A1 = relu(Z1)
        Z2 = A1 @ W2 + b2;  A2 = relu(Z2)
        Z3 = A2 @ W3 + b3;  A3 = sigmoid(Z3)
        # Backward
        m   = X.shape[0]
        dZ3 = A3 - y
        dW3 = (1/m) * A2.T @ dZ3
        db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)
        dZ2 = (dZ3 @ W3.T) * relu_deriv(Z2)
        dW2 = (1/m) * A1.T @ dZ2
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = (dZ2 @ W2.T) * relu_deriv(Z1)
        dW1 = (1/m) * X.T @ dZ1
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        # Update
        W1 -= lr * dW1;  b1 -= lr * db1
        W2 -= lr * dW2;  b2 -= lr * db2
        W3 -= lr * dW3;  b3 -= lr * db3

    return W1, b1, W2, b2, W3, b3

def predict_3layer(X, W1, b1, W2, b2, W3, b3):
    A1 = relu(X @ W1 + b1)
    A2 = relu(A1 @ W2 + b2)
    A3 = sigmoid(A2 @ W3 + b3)
    return (A3 >= 0.5).astype(int)

# ─────────────────────────────────────────
# 5. RUN ALL EXPERIMENTS
# ─────────────────────────────────────────
experiments = []

# Width sweep
for n in [2, 8, 16, 32]:
    label = f"2-layer, {n} hidden"
    t0 = time.time()
    params = train_2layer(X_train, y_train, n_hidden=n)
    elapsed = time.time() - t0
    preds = predict_2layer(X_test, *params)
    acc = accuracy_score(y_test, preds)
    n_params = (2 * n + n) + (n * 1 + 1)  # W1+b1 + W2+b2
    experiments.append({
        "label": label,
        "acc": acc,
        "time": elapsed,
        "params": n_params,
        "predict_fn": lambda grid, p=params: predict_2layer(grid, *p)
    })
    print(f"{label:25s}  acc={acc:.4f}  params={n_params:4d}  time={elapsed:.2f}s")

# Depth experiment
label = "3-layer, 16+16 hidden"
t0 = time.time()
params_3 = train_3layer(X_train, y_train, n_hidden=16)
elapsed = time.time() - t0
preds_3 = predict_3layer(X_test, *params_3)
acc_3 = accuracy_score(y_test, preds_3)
n_params_3 = (2*16+16) + (16*16+16) + (16*1+1)
experiments.append({
    "label": label,
    "acc": acc_3,
    "time": elapsed,
    "params": n_params_3,
    "predict_fn": lambda grid, p=params_3: predict_3layer(grid, *p)
})
print(f"{label:25s}  acc={acc_3:.4f}  params={n_params_3:4d}  time={elapsed:.2f}s")

# ─────────────────────────────────────────
# 6. PLOT — 5 decision boundaries
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle("Architecture Comparison — make_moons", fontsize=14, fontweight='bold')

x0_min, x0_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
x1_min, x1_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x0_min, x0_max, 300),
    np.linspace(x1_min, x1_max, 300)
)
grid = np.c_[xx.ravel(), yy.ravel()]

for ax, exp in zip(axes, experiments):
    Z = exp["predict_fn"](grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X_all[:, 0], X_all[:, 1], c=y.ravel(),
               cmap='RdBu', edgecolors='k', linewidths=0.2, s=12)
    ax.set_title(
        f"{exp['label']}\nacc={exp['acc']:.4f} | {exp['params']} params",
        fontsize=9
    )
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig("week12/session2_architecture.png", dpi=150)
print("\nPlot saved → week12/session2_architecture.png")