# week12/session1_nonlinear.py
# Goal: Prove neural nets beat logistic regression on non-linear data

import numpy as np
import matplotlib
matplotlib.use('Agg')  # No display needed — SSH safe
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Dataset: {X.shape[0]} samples, 2 features")
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}\n")

# ─────────────────────────────────────────
# 2. SHARED HELPERS
# ─────────────────────────────────────────
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# ─────────────────────────────────────────
# 3. LOGISTIC REGRESSION (from scratch)
# ─────────────────────────────────────────
def train_logistic(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    W = np.zeros((n, 1))
    b = 0.0
    for epoch in range(epochs):
        y_pred = sigmoid(X @ W + b)
        dW = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)
        W -= lr * dW
        b -= lr * db
    return W, b

def predict_logistic(X, W, b, threshold=0.5):
    return (sigmoid(X @ W + b) >= threshold).astype(int)

# ─────────────────────────────────────────
# 4. NEURAL NETWORK (from scratch)
# ─────────────────────────────────────────
def init_weights(n_input, n_hidden, n_output):
    np.random.seed(42)
    W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros((1, n_output))
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def backward(X, y, Z1, A1, A2, W1, W2):
    m = X.shape[0]
    dZ2 = A2 - y
    dW2 = (1/m) * A1.T @ dZ2
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * X.T @ dZ1
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

def train_nn(X, y, n_hidden=8, lr=0.1, epochs=1000):
    W1, b1, W2, b2 = init_weights(X.shape[1], n_hidden, 1)
    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(X, y, Z1, A1, A2, W1, W2)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        if epoch % 1000 == 0:
            loss = binary_cross_entropy(y, A2)
            print(f"  [NN]  Epoch {epoch:4d}  Loss: {loss:.4f}")
    return W1, b1, W2, b2

def predict_nn(X, W1, b1, W2, b2, threshold=0.5):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return (A2 >= threshold).astype(int)

# ─────────────────────────────────────────
# 5. TRAIN BOTH MODELS
# ─────────────────────────────────────────
print("── Logistic Regression ──")
W_lr, b_lr = train_logistic(X_train, y_train, lr=0.1, epochs=1000)
y_pred_lr  = predict_logistic(X_test, W_lr, b_lr)
acc_lr     = accuracy_score(y_test, y_pred_lr)
print(f"  Test accuracy: {acc_lr:.4f}\n")

print("── Neural Network (16 hidden neurons) ──")
W1, b1, W2, b2 = train_nn(X_train, y_train, n_hidden=16, lr=0.1, epochs=5000)
y_pred_nn  = predict_nn(X_test, W1, b1, W2, b2)
acc_nn     = accuracy_score(y_test, y_pred_nn.flatten())
print(f"\n  Test accuracy: {acc_nn:.4f}\n")

print(f"Gap (NN - LR): {acc_nn - acc_lr:+.4f}")

# ─────────────────────────────────────────
# 6. DECISION BOUNDARY PLOT
# ─────────────────────────────────────────
def plot_decision_boundary(ax, X, y, predict_fn, title, accuracy):
    # Build a fine grid covering the data
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdBu',
               edgecolors='k', linewidths=0.3, s=20)
    ax.set_title(f"{title}\nAccuracy: {accuracy:.4f}", fontsize=13)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

# Use the full (scaled) dataset for plotting so the boundary is visible
X_all = scaler.transform(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("make_moons — Decision Boundaries", fontsize=15, fontweight='bold')

plot_decision_boundary(
    axes[0], X_all, y,
    lambda grid: predict_logistic(grid, W_lr, b_lr),
    "Logistic Regression (from scratch)", acc_lr
)

plot_decision_boundary(
    axes[1], X_all, y,
    lambda grid: predict_nn(grid, W1, b1, W2, b2),
    "Neural Network — 16 hidden (from scratch)", acc_nn
)

plt.tight_layout()
output_path = "week12/session1_decision_boundary.png"
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved → {output_path}")