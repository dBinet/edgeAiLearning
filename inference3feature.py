import numpy as np
import time

# --- Synthetic dataset: size, bedrooms, house age ---
np.random.seed(42)
n = 50000

size     = np.random.uniform(500, 3500, n)
bedrooms = np.random.randint(1, 6, n).astype(float)
age      = np.random.uniform(1, 50, n)  # years old

# Price formula: size matters most, bedrooms add value, age reduces it
price = (150 * size) + (10000 * bedrooms) - (500 * age) + np.random.normal(0, 10000, n)

# --- Normalize features ---
def normalize(arr):
    return (arr - arr.mean()) / arr.std()

X = np.column_stack([normalize(size), normalize(bedrooms), normalize(age)])
y = normalize(price)

# --- Initialize weights ---
w = np.zeros(3)
b = 0.0
lr = 0.01
epochs = 1000

# --- Training ---
start = time.time()
for _ in range(epochs):
    preds = X @ w + b
    err = preds - y
    w -= lr * (X.T @ err) / n
    b -= lr * err.mean()
elapsed = time.time() - start

# --- Results ---
print(f"Training time: {elapsed:.4f}s")
print(f"Samples: {n}, Epochs: {epochs}")
print(f"Weights — size: {w[0]:.4f}, bedrooms: {w[1]:.4f}, age: {w[2]:.4f}")
print(f"Bias: {b:.4f}")

# --- Prediction function ---
def predict_house(size_sqft, num_bedrooms, house_age):
    # Normalize using training data stats
    size_n     = (size_sqft    - size.mean())     / size.std()
    bed_n      = (num_bedrooms - bedrooms.mean())  / bedrooms.std()
    age_n      = (house_age    - age.mean())        / age.std()

    x = np.array([size_n, bed_n, age_n])
    pred_normalized = x @ w + b

    # Denormalize back to actual price
    pred_price = (pred_normalized * price.std()) + price.mean()
    return pred_price

# --- Test some houses ---
houses = [
    (800,  1, 40),   # small, 1 bed, old
    (2000, 3, 10),   # medium, 3 bed, newer
    (3500, 5, 2),    # large, 5 bed, brand new
]

print("\n--- House Predictions ---")
for s, b_n, a in houses:
    p = predict_house(s, b_n, a)
    print(f"  {s} sqft | {b_n} bed | {a} yrs old  →  ${p:,.0f}")