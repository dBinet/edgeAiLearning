import numpy as np
import time
from sklearn.preprocessing import StandardScaler

# Dataset - now with 2 features: size and bedrooms
size     = np.array([650, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2500])
bedrooms = np.array([1,   1,   2,    2,    3,    3,    4,    4,    4,    5   ])
price    = np.array([150, 180, 220,  250,  280,  310,  340,  370,  400,  450 ])

# Stack features into a matrix (10 rows, 2 columns)
X = np.column_stack([size, bedrooms])
y = price

# Scale features
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Vectorized gradient descent - now w is a vector, not a scalar
def compute_gradient(X, y, w, b):
    predictions = X.dot(w) + b
    errors = predictions - y
    dj_dw = X.T.dot(errors) / len(y)
    dj_db = np.sum(errors) / len(y)
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, learning_rate, num_iters):
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db
    return w, b

# Train
w = np.zeros(X_scaled.shape[1])  # one weight per feature
b = 0.0

start = time.time()
w, b = gradient_descent(X_scaled, y_scaled, w, b, 0.1, 200)
train_time = time.time() - start

test = np.array([[2500, 1]])  # big house, 1 bedroom
test_scaled = scaler_x.transform(test)

infer_start = time.time()
prediction_scaled = test_scaled.dot(w) + b
prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
infer_time = time.time() - infer_start

print(f"Training time:   {train_time:.4f}s")
print(f"Inference time:  {infer_time:.6f}s")
print(f"Weights: size={w[0]:.4f}, bedrooms={w[1]:.4f}")
print(f"Predicted price for 2500 sq ft, 1 bed: ${prediction:.1f}k")