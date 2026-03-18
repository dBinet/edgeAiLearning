# week11/session2_threshold.py
# Goal: Tune classification threshold, observe precision/recall tradeoff

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# --- Data (full dataset this time) ---
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- sigmoid function ---
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

# --- binary cross-entropy loss function ---
def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# --- train function ---
def train(X, y, learning_rate=0.1, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    
    for epoch in range(epochs):
        # 1. forward pass — compute predictions
        y_pred = sigmoid_function(X @ w + b)
        # 2. compute loss
        loss = compute_loss(y, y_pred)
        # 3. compute gradients dw and db
        dw = (1 / n_samples) * X.T @ (y_pred - y)
        db = np.mean(y_pred - y)
        # 4. update w and b
        w = w - learning_rate * dw
        b = b - learning_rate * db
    return w, b

def predict(X, w, b):
    return sigmoid_function(X @ w + b)

# Train on full training set

w, b = train(X_train, y_train, 0.1, 1000)
probs = predict(X_test, w, b)
# --- Part 1: Threshold sweep ---
for thres in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    y_pred_test = (probs >= thres).astype(int)
    accuracy = np.mean(y_pred_test == y_test)
    print(f"Threshold: {thres:.1f}")
    print(confusion_matrix(y_test, y_pred_test))
    print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_test):.4f}")
    print(f"F1: {f1_score(y_test, y_pred_test):.4f}")

# --- Part 2: Find the threshold that maximizes recall ---
# 0.4 is the threshold that maximizes the recall value. It's what we should aim for since false negatives
# are more dangerous than false positive

# --- Part 3: Find the threshold that maximizes F1 ---
# 0.4 is the threshold that maximizes the F1 value. But this comes at the downside of having false negatives
# when running the model. This could cost the live of a patient that should have been flagged as postive