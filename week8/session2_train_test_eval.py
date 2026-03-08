import numpy as np
import time

# --- Data generation ---
np.random.seed(42)
n = 50000

size     = np.random.uniform(500, 3500, n)
bedrooms = np.random.randint(1, 6, n).astype(float)
age      = np.random.uniform(1, 50, n)

price = (150 * size) + (10000 * bedrooms) - (500 * age) + np.random.normal(0, 10000, n)

# --- Normalize ---
def normalize(arr):
    return (arr - arr.mean()) / arr.std(), arr.mean(), arr.std()

size_n,     size_mean,     size_std     = normalize(size)
bedrooms_n, bedrooms_mean, bedrooms_std = normalize(bedrooms)
age_n,      age_mean,      age_std      = normalize(age)
price_n,    price_mean,    price_std    = normalize(price)

X = np.column_stack([size_n, bedrooms_n, age_n])
y = price_n

# --- Train/test split (80/20) ---
split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# --- Training ---
w = np.zeros(3)
b = 0.0
lr = 0.01
epochs = 1000
n_train = len(X_train)

start = time.time()
for _ in range(epochs):
    preds = X_train @ w + b
    err   = preds - y_train
    w    -= lr * (X_train.T @ err) / n_train
    b    -= lr * err.mean()
elapsed = time.time() - start

# --- Evaluation ---
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - (ss_res / ss_tot)

train_preds = X_train @ w + b
test_preds  = X_test  @ w + b

train_mse = mse(y_train, train_preds)
test_mse  = mse(y_test,  test_preds)
train_r2  = r_squared(y_train, train_preds)
test_r2   = r_squared(y_test,  test_preds)

# --- Results ---
print(f"\nTraining time : {elapsed:.4f}s")
print(f"Weights — size: {w[0]:.4f}, bedrooms: {w[1]:.4f}, age: {w[2]:.4f}")
print(f"\n--- Evaluation ---")
print(f"Train MSE: {train_mse:.4f}  |  Test MSE: {test_mse:.4f}")
print(f"Train R²:  {train_r2:.4f}  |  Test R²:  {test_r2:.4f}")