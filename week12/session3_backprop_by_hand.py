# week12/session3_backprop_by_hand.py
# Verify manual backprop calculations against NumPy

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

# ── Inputs & weights ──
X  = np.array([[1.0, 0.5]])
y  = np.array([[1.0]])
W1 = np.array([[0.5, -0.3],
               [0.2,  0.8]])
b1 = np.zeros((1, 2))
W2 = np.array([[0.7],
               [-0.4]])
b2 = np.zeros((1, 1))
lr = 0.1

# ── Forward ──
Z1 = X @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)

print("── Forward Pass ──")
print(f"Z1 = {Z1}")
print(f"A1 = {A1}")
print(f"Z2 = {Z2}")
print(f"A2 = {A2}")

# ── Loss ──
loss = -np.mean(y * np.log(A2) + (1 - y) * np.log(1 - A2))
print(f"\nLoss = {loss:.4f}")

# ── Backward ──
m   = X.shape[0]
dZ2 = A2 - y
dW2 = (1/m) * A1.T @ dZ2
db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
dA1 = dZ2 @ W2.T
dZ1 = dA1 * relu_deriv(Z1)
dW1 = (1/m) * X.T @ dZ1
db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

print("\n── Backward Pass ──")
print(f"dZ2 = {dZ2}")
print(f"dW2 = {dW2}")
print(f"db2 = {db2}")
print(f"dA1 = {dA1}")
print(f"dZ1 = {dZ1}")
print(f"dW1 =\n{dW1}")
print(f"db1 = {db1}")

# ── Weight update ──
W2_new = W2 - lr * dW2
W1_new = W1 - lr * dW1

print("\n── Updated Weights ──")
print(f"W2_new =\n{W2_new}")
print(f"W1_new =\n{W1_new}")