import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- Setup ---
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# --- Benchmark training ---
start = time.time()
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
train_time = time.time() - start

# --- Benchmark single inference ---
single_sample = X_test[0:1]
start = time.time()
for _ in range(10000):
    model.predict(single_sample)
inference_time = (time.time() - start) / 10000

# --- Benchmark batch inference ---
start = time.time()
model.predict(X_test)
batch_time = time.time() - start

print(f"Training time       : {train_time:.4f}s")
print(f"Single inference    : {inference_time*1000:.4f}ms")
print(f"Batch inference     : {batch_time:.4f}s ({len(X_test)} samples)")