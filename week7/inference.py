import numpy as np
import time
from sklearn.preprocessing import StandardScaler

# Dataset
np.random.seed(42)
size = np.random.uniform(500, 3000, 10000)
price = size * 0.15 + np.random.normal(0, 20, 10000)

# Model functions
def linear_model(x, w, b):
    return x * w + b

def compute_gradient(x, y, w, b):
    predictions = w * x + b
    errors = predictions - y
    dj_dw = np.dot(errors, x) / len(x)
    dj_db = np.sum(errors) / len(x)
    return dj_dw, dj_db

def gradient_descent(x, y, w, b, learning_rate, num_iters):
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db
    return w, b

# Scale features
scaler_x = StandardScaler()
scaler_y = StandardScaler()
size_scaled = scaler_x.fit_transform(size.reshape(-1, 1)).flatten()
price_scaled = scaler_y.fit_transform(price.reshape(-1, 1)).flatten()

# Train and time it
start = time.time()
w, b = gradient_descent(size_scaled, price_scaled, 0, 0, 0.1, 200)
train_time = time.time() - start

# Inference: predict price for a 1500 sq ft house
test_size = np.array([[1500]])
test_scaled = scaler_x.transform(test_size)

infer_start = time.time()
prediction_scaled = linear_model(test_scaled[0][0], w, b)
prediction = scaler_y.inverse_transform([[prediction_scaled]])[0][0]
infer_time = time.time() - infer_start

print(f"Training time:   {train_time:.4f}s")
print(f"Inference time:  {infer_time:.6f}s")
print(f"Predicted price for 1500 sq ft: ${prediction:.1f}k")