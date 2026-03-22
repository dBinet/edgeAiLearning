# week11/session5_neural_net.py
# Goal: Build a 2-layer neural network from scratch
# Architecture: Input → Hidden (ReLU) → Output (Sigmoid)

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- Data ---
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# --- Activations ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return (z > 0).astype(float)

# --- Initialize weights ---
def init_weights(n_input, n_hidden):
    np.random.seed(42)
    W1 = np.random.randn(n_input, n_hidden) * 0.01
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1) * 0.01
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

# --- Forward pass ---
def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    return Z1, A1, A2

# --- Loss ---
def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# --- Backward pass ---
def backward(X, y, A1, A2, W2, Z1):
    n_samples = X.shape[0]
    # Output layer gradients
    dZ2 = A2 - y
    dW2 = (1 / n_samples) * A1.T @ dZ2
    db2 = (1 / n_samples) * np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer gradients — chain rule through ReLU
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1 / n_samples) * X.T @ dZ1
    db1 = (1 / n_samples) * np.sum(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

# --- Train ---
def train(X, y, n_hidden=8, learning_rate=0.01, epochs=5000):
    n_input = X.shape[1]
    W1, b1, W2, b2 = init_weights(n_input, n_hidden)

    for epoch in range(epochs):
        # 1. forward pass — get predictions
        Z1, A1, A2 = forward(X, W1, b1, W2, b2)
        # compute loss
        loss = compute_loss(y, A2)
        # 3. backward pass — compute gradients
        dW1, db1, dW2, db2 = backward(X, y, A1, A2, W2, Z1)
        # 4. update weights
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        # print loss every 100 epochs
        if (epoch % 100 == 0):
            print(f"Loss: {loss}")

    return W1, b1, W2, b2

# --- Predict ---
def predict(X, W1, b1, W2, b2, threshold=0.5):
    _, _, A2 = forward(X, W1, b1, W2, b2)
    return (A2 >= threshold).astype(int)

# --- Run ---
W1, b1, W2, b2 = train(X_train, y_train, n_hidden=8, learning_rate=0.01, epochs=1000)
y_pred = predict(X_test, W1, b1, W2, b2)
print(f"Neural net accuracy: {accuracy_score(y_test.flatten(), y_pred.flatten()):.4f}")

# Compare to your logistic regression from scratch
# Expected: similar or slightly better accuracy