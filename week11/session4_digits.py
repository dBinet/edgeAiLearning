# Goal: Build a model from scratch on a new dataset 

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# --- Functions ---
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train(X, y, learning_rate=0.1, epochs=20000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    prev_loss = float('inf')
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
        
        # 5. print loss every 500 epochs
        if (epoch % 500 == 0):
            print(f"Loss: {loss}")
        if epoch > 0 and abs(prev_loss - loss) < 1e-6:
            print(f"Converged at epoch {epoch}")
            break
        prev_loss = loss

    return w, b

def predict_proba(X, w, b):
    return sigmoid_function(X @ w + b)

# --- Data ---
# 142 Samples and 13 features
X, y = load_wine(return_X_y=True)
# Keep only class 1 and 2, drop class 0
mask = y != 0
X, y = X[mask], y[mask]
y = (y == 1).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df = pd.DataFrame(X)

# Calculate the correlation matrix
correlation_matrix = df.corr()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# --- Train ---
w, b = train(X_train, y_train)
probs = predict_proba(X_test, w, b)



# Sweep thresholds
for thres in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    y_pred = (probs >= thres).astype(int)
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")

# Evaluate at chosen threshold
threshold = 0.7
y_pred_final = (probs >= threshold).astype(int)
print(confusion_matrix(y_test, y_pred_final))
print(f"Precision: {precision_score(y_test, y_pred_final):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_final):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_final):.4f}")
print(f"AUC:       {roc_auc_score(y_test, probs):.4f}")

# --- Findings ---
# Classes 1 vs 2 are linearly separable after scaling
# AUC = 1.0, accuracy = 1.0 across thresholds 0.1-0.6
# Feature engineering skipped — model already perfect, dropping features would only hurt
# Multicollinearity noted: features 2-3 (0.72), features 5-6 (0.77)