import numpy as np
import time
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Load real data ---
data = fetch_california_housing()
X, y = data.data, data.target

print(f"Dataset shape: {X.shape}")
print(f"Features: {data.feature_names}")
print(f"Target: median house value (hundreds of thousands)")
print(f"Price range: ${y.min():.2f} - ${y.max():.2f} (x100k)")

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Normalize using sklearn ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# --- Normalize target ---
y_mean, y_std = y_train.mean(), y_train.std()
y_train_n = (y_train - y_mean) / y_std
y_test_n  = (y_test  - y_mean) / y_std

# --- Training ---
w = np.zeros(X_train.shape[1])
b = 0.0
lr = 0.01
epochs = 1000
n_train = len(X_train)

start = time.time()
for _ in range(epochs):
    preds = X_train @ w + b
    err   = preds - y_train_n
    w    -= lr * (X_train.T @ err) / n_train
    b    -= lr * err.mean()
elapsed = time.time() - start

# --- Evaluation ---
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - (ss_res / ss_tot)

train_preds = X_train @ w + b
test_preds  = X_test  @ w + b

train_r2 = r_squared(y_train_n, train_preds)
test_r2  = r_squared(y_test_n,  test_preds)

# --- Results ---
print(f"\nTraining time: {elapsed:.4f}s")
print(f"\nFeature weights:")
for name, weight in zip(data.feature_names, w):
    print(f"  {name:10s}: {weight:+.4f}")
print(f"\n--- Evaluation ---")
print(f"Train R²: {train_r2:.4f}  |  Test R²: {test_r2:.4f}")