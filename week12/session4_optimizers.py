# week12/session4_optimizers.py
# Compare vanilla GD, momentum, and Adam on make_moons

import numpy as np
import matplotlib
matplotlib.use('Agg')
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
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

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
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

def init_weights(n_in, n_hidden):
    np.random.seed(42)
    W1 = np.random.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1;  A1 = relu(Z1)
    Z2 = A1 @ W2 + b2; A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def backward(X, y, Z1, A1, A2, W2):
    m   = X.shape[0]
    dZ2 = A2 - y
    dW2 = (1/m) * A1.T @ dZ2
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
    dZ1 = (dZ2 @ W2.T) * relu_deriv(Z1)
    dW1 = (1/m) * X.T @ dZ1
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

# ─────────────────────────────────────────
# 3. VANILLA GRADIENT DESCENT
# ─────────────────────────────────────────
def train_vanilla(X, y, n_hidden=16, lr=0.1, epochs=500):
    W1, b1, W2, b2 = init_weights(X.shape[1], n_hidden)
    losses = []
    for _ in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        losses.append(bce(y, A2))
        dW1, db1, dW2, db2 = backward(X, y, Z1, A1, A2, W2)
        W1 -= lr * dW1;  b1 -= lr * db1
        W2 -= lr * dW2;  b2 -= lr * db2
    return W1, b1, W2, b2, losses

# ─────────────────────────────────────────
# 4. MOMENTUM
# ─────────────────────────────────────────
def train_momentum(X, y, n_hidden=16, lr=0.1, beta=0.9, epochs=500):
    W1, b1, W2, b2 = init_weights(X.shape[1], n_hidden)
    # Velocity terms — same shape as each weight matrix
    vW1 = np.zeros_like(W1);  vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2);  vb2 = np.zeros_like(b2)
    losses = []
    for _ in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        losses.append(bce(y, A2))
        dW1, db1, dW2, db2 = backward(X, y, Z1, A1, A2, W2)
        # Update velocities
        vW1 = beta * vW1 + dW1;  vb1 = beta * vb1 + db1
        vW2 = beta * vW2 + dW2;  vb2 = beta * vb2 + db2
        # Update weights using velocity
        W1 -= lr * vW1;  b1 -= lr * vb1
        W2 -= lr * vW2;  b2 -= lr * vb2
    return W1, b1, W2, b2, losses

# ─────────────────────────────────────────
# 5. ADAM
# ─────────────────────────────────────────
def train_adam(X, y, n_hidden=16, lr=0.001, beta1=0.9, beta2=0.999,
               epsilon=1e-8, epochs=500):
    W1, b1, W2, b2 = init_weights(X.shape[1], n_hidden)
    # First moment (momentum) — one per weight matrix
    mW1 = np.zeros_like(W1);  mb1 = np.zeros_like(b1)
    mW2 = np.zeros_like(W2);  mb2 = np.zeros_like(b2)
    # Second moment (variance) — one per weight matrix
    vW1 = np.zeros_like(W1);  vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2);  vb2 = np.zeros_like(b2)
    losses = []
    for t in range(1, epochs + 1):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        losses.append(bce(y, A2))
        dW1, db1, dW2, db2 = backward(X, y, Z1, A1, A2, W2)

        # Update first moments
        mW1 = beta1*mW1 + (1-beta1)*dW1;  mb1 = beta1*mb1 + (1-beta1)*db1
        mW2 = beta1*mW2 + (1-beta1)*dW2;  mb2 = beta1*mb2 + (1-beta1)*db2
        # Update second moments
        vW1 = beta2*vW1 + (1-beta2)*dW1**2;  vb1 = beta2*vb1 + (1-beta2)*db1**2
        vW2 = beta2*vW2 + (1-beta2)*dW2**2;  vb2 = beta2*vb2 + (1-beta2)*db2**2
        # Bias correction
        mW1_c = mW1/(1-beta1**t);  mb1_c = mb1/(1-beta1**t)
        mW2_c = mW2/(1-beta1**t);  mb2_c = mb2/(1-beta1**t)
        vW1_c = vW1/(1-beta2**t);  vb1_c = vb1/(1-beta2**t)
        vW2_c = vW2/(1-beta2**t);  vb2_c = vb2/(1-beta2**t)
        # Update weights
        W1 -= lr * mW1_c / (np.sqrt(vW1_c) + epsilon)
        b1 -= lr * mb1_c / (np.sqrt(vb1_c) + epsilon)
        W2 -= lr * mW2_c / (np.sqrt(vW2_c) + epsilon)
        b2 -= lr * mb2_c / (np.sqrt(vb2_c) + epsilon)
    return W1, b1, W2, b2, losses

# ─────────────────────────────────────────
# 6. TRAIN & COMPARE
# ─────────────────────────────────────────
EPOCHS = 500

print("Training vanilla GD...")
W1v, b1v, W2v, b2v, losses_vanilla = train_vanilla(
    X_train, y_train, epochs=EPOCHS)

print("Training momentum...")
W1m, b1m, W2m, b2m, losses_momentum = train_momentum(
    X_train, y_train, epochs=EPOCHS)

print("Training Adam...")
W1a, b1a, W2a, b2a, losses_adam = train_adam(
    X_train, y_train, epochs=EPOCHS, lr=0.01)

def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return (A2 >= 0.5).astype(int)

acc_v = accuracy_score(y_test, predict(X_test, W1v, b1v, W2v, b2v))
acc_m = accuracy_score(y_test, predict(X_test, W1m, b1m, W2m, b2m))
acc_a = accuracy_score(y_test, predict(X_test, W1a, b1a, W2a, b2a))

print(f"\n{'Optimizer':<12} {'Final Loss':>12} {'Test Acc':>10}")
print("-" * 36)
print(f"{'Vanilla':<12} {losses_vanilla[-1]:>12.4f} {acc_v:>10.4f}")
print(f"{'Momentum':<12} {losses_momentum[-1]:>12.4f} {acc_m:>10.4f}")
print(f"{'Adam':<12} {losses_adam[-1]:>12.4f} {acc_a:>10.4f}")

# ─────────────────────────────────────────
# 7. PLOT LOSS CURVES
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses_vanilla,  label=f"Vanilla GD  (acc={acc_v:.4f})", linewidth=2)
ax.plot(losses_momentum, label=f"Momentum    (acc={acc_m:.4f})", linewidth=2)
ax.plot(losses_adam,     label=f"Adam        (acc={acc_a:.4f})", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Optimizer Comparison — make_moons (500 epochs)", fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("week12/session4_optimizers.png", dpi=150)
print("\nPlot saved → week12/session4_optimizers.png")