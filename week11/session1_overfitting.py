# week11/session1_overfitting.py
# Goal: See overfitting happen, then fix it with L2 regularization

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- Data ---
X, y = load_breast_cancer(return_X_y=True)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test = scaler.transform(X_test)

# --- Tiny subset to force overfitting ---
X_train_tiny = X_train_full[:5]
y_train_tiny = y_train_full[:5]

# --- Noisy to force overfitting ---
np.random.seed(42)
X_train_noisy = X_train_tiny + np.random.normal(0, 3.0, X_train_tiny.shape)
y_train_noisy = y_train_full[:5]

# --- Functions ---

# --- sigmoid function ---
def sigmoid_function(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# --- binary cross-entropy loss function ---
def compute_loss(y_true, y_pred, n_samples, lam, w):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # add this line
    original_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    L2_penalty = (lam / (2 * n_samples)) * np.sum(w ** 2)
    return original_loss + L2_penalty

# --- train function ---
def train(X, y, learning_rate=0.1, epochs=10000, lam = 0.15):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for epoch in range(epochs):
        # 1. forward pass — compute predictions
        y_pred = sigmoid_function(X @ w + b)
        # 2. compute loss
        loss = compute_loss(y, y_pred, n_samples, lam, w)
        # 3. compute gradients dw and db
        original_w = (1 / n_samples) * X.T @ (y_pred - y)
        dw = original_w + (lam / n_samples) * w

        db = np.mean(y_pred - y)
        # 4. update w and b
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b

def predict(X, w, b):
    y_pred = sigmoid_function(X @ w + b)
    return (y_pred >= 0.5).astype(int)  # convert probabilities to 0 or 1

print(f"Tiny data:")
for lam in [0, 0.01, 0.1, 1.0, 10.0, 100.0]:
    w, b = train(X_train_tiny, y_train_tiny, learning_rate=0.1, epochs=10000, lam=lam)
    train_acc = np.mean(predict(X_train_tiny, w, b) == y_train_tiny)
    test_acc = np.mean(predict(X_test, w, b) == y_test)
    print(f"lam={lam:6} | train: {train_acc:.4f} | test: {test_acc:.4f}")

print(f"Noisy data:")
for lam in [0, 0.01, 0.1, 1.0, 10.0, 100.0]:
    w, b = train(X_train_noisy, y_train_noisy, learning_rate=0.1, epochs=10000, lam=lam)
    train_acc = np.mean(predict(X_train_noisy, w, b) == y_train_noisy)
    test_acc = np.mean(predict(X_test, w, b) == y_test)
    print(f"lam={lam:6} | train: {train_acc:.4f} | test: {test_acc:.4f}")