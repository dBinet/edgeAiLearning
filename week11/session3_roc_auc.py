# week11/session3_roc_auc.py
# Goal: Build ROC curve from scratch, compute AUC, verify against sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

# --- Data ---
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Functions ---
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train(X, y, learning_rate=0.1, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    for epoch in range(epochs):
        y_pred = sigmoid_function(X @ w + b)
        loss = compute_loss(y, y_pred)
        dw = (1 / n_samples) * X.T @ (y_pred - y)
        db = np.mean(y_pred - y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    return w, b

def predict_proba(X, w, b):
    return sigmoid_function(X @ w + b)

# --- Train ---
w, b = train(X_train, y_train)
probs = predict_proba(X_test, w, b)

# --- Part 1: ROC curve from scratch ---
# Sweep thresholds, compute TPR and FPR at each point
FPR = []
TPR = []
for thres in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    y_pred = (probs >= thres).astype(int)
    tn = fp = fn = tp = 0
    for pred, real in np.nditer([y_pred, y_test]):
        if pred == 0:
            tn += (pred == real)
            fn += (pred != real)
        else:
            tp += (pred == real)
            fp += (pred != real)
    FPR.append(fp / (fp + tn))
    TPR.append(tp / (tp + fn))

# Add corner points and sort by FPR for correct integration
sorted_pairs = sorted(zip(FPR, TPR))
FPR_full = [0.0] + [p[0] for p in sorted_pairs] + [1.0]
TPR_full = [0.0] + [p[1] for p in sorted_pairs] + [1.0]

# --- Part 2: AUC from scratch ---
auc_custom = np.trapezoid(TPR_full, x=FPR_full)
print(f"From-scratch AUC: {auc_custom:.4f}")

# --- Part 3: Compare to sklearn ---
fpr_sklearn, tpr_sklearn, _ = roc_curve(y_test, probs)
auc_sklearn = roc_auc_score(y_test, probs)
print(f"Sklearn AUC:       {auc_sklearn:.4f}")

# --- Plot ---
plt.plot(FPR_full, TPR_full, label=f"From scratch (AUC={auc_custom:.4f})")
plt.plot(fpr_sklearn, tpr_sklearn, label=f"Sklearn (AUC={auc_sklearn:.4f})", linestyle="--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.savefig("week11/roc_curve.png")