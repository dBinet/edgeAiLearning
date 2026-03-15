import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
        
        # 5. print loss every 100 epochs
        if (epoch % 100 == 0):
            print(f"Loss: {loss}")

    return w, b

def predict(X, w, b):
    y_pred = sigmoid_function(X @ w + b)
    return (y_pred >= 0.5).astype(int)  # convert probabilities to 0 or 1



# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

start = time.time()
w, b = train(X_train, y_train, 0.1, 1000)
scratch_train_time = time.time() - start

start = time.time()
y_pred_test = predict(X_test, w, b)
scratch_batch_time = time.time() - start

start = time.time()
y_pred_single_test = predict(X_test[0:1], w, b)
scratch_single_time = time.time() - start

accuracy = np.mean(y_pred_test == y_test)


# From scracth algorithm before a bit better but it's because there is no safety that would avoid overfitting the data.
# This could become a problem on noisier data where overfitting is more likely to happen.
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

start = time.time()
clf.fit(X_train, y_train)
sklearn_train_time = time.time() - start

start = time.time()
y_pred_test = clf.predict(X_test)
sklearn_batch_time = time.time() - start

start = time.time()
y_pred_single_test = clf.predict(X_test[0:1])
sklearn_single_time = time.time() - start


# From-scratch inference is 3.5x faster than sklearn on Pi. Training is 20x slower.
# For edge deployment: train offline on powerful hardware, 
# deploy only the weights and the forward pass.

print(f"From-scratch Training time   : {scratch_train_time:.4f}s")
print(f"sklearn Training time        : {sklearn_train_time:.4f}s")

print(f"From-scratch Single time     : {scratch_single_time*1000:.4f}ms")
print(f"sklearn Single time          : {sklearn_single_time*1000:.4f}ms")

print(f"From-scratch Batch time      : {scratch_batch_time*1000:.4f}ms ({len(X_test)} samples)")
print(f"sklearn Batch time           : {sklearn_batch_time*1000:.4f}ms ({len(X_test)} samples)")

print(f"From-scratch accuracy        : {accuracy:.4f}")
print(f"sklearn accuracy             : {clf.score(X_test, y_test):.4f}")


