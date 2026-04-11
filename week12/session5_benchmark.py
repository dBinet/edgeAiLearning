# week12/session5_benchmark.py
# Measure real training and inference cost on Pi hardware

import numpy as np
import time
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Single sample for per-sample inference benchmark
X_single = X_test[[0]]

# ─────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

def init_weights(n_in, n_hidden):
    np.random.seed(42)
    W1 = np.random.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

# ─────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────
def train_logistic(X, y, lr=0.1, epochs=500):
    m, n = X.shape
    W = np.zeros((n, 1))
    b = 0.0
    for _ in range(epochs):
        y_pred = sigmoid(X @ W + b)
        dW = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)
        W -= lr * dW
        b -= lr * db
    return W, b

def infer_logistic(X, W, b):
    return (sigmoid(X @ W + b) >= 0.5).astype(int)

def forward_2layer(X, W1, b1, W2, b2):
    A1 = relu(X @ W1 + b1)
    A2 = sigmoid(A1 @ W2 + b2)
    return (A2 >= 0.5).astype(int)

def forward_3layer(X, W1, b1, W2, b2, W3, b3):
    A1 = relu(X @ W1 + b1)
    A2 = relu(A1 @ W2 + b2)
    A3 = sigmoid(A2 @ W3 + b3)
    return (A3 >= 0.5).astype(int)

def train_2layer_vanilla(X, y, n_hidden=16, lr=0.1, epochs=500):
    W1, b1, W2, b2 = init_weights(X.shape[1], n_hidden)
    for _ in range(epochs):
        Z1 = X @ W1 + b1;  A1 = relu(Z1)
        Z2 = A1 @ W2 + b2; A2 = sigmoid(Z2)
        m   = X.shape[0]
        dZ2 = A2 - y
        dW2 = (1/m) * A1.T @ dZ2
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = (dZ2 @ W2.T) * relu_deriv(Z1)
        dW1 = (1/m) * X.T @ dZ1
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        W1 -= lr * dW1;  b1 -= lr * db1
        W2 -= lr * dW2;  b2 -= lr * db2
    return W1, b1, W2, b2

def train_2layer_momentum(X, y, n_hidden=16, lr=0.1, beta=0.9, epochs=500):
    W1, b1, W2, b2 = init_weights(X.shape[1], n_hidden)
    vW1=np.zeros_like(W1); vb1=np.zeros_like(b1)
    vW2=np.zeros_like(W2); vb2=np.zeros_like(b2)
    for _ in range(epochs):
        Z1 = X @ W1 + b1;  A1 = relu(Z1)
        Z2 = A1 @ W2 + b2; A2 = sigmoid(Z2)
        m   = X.shape[0]
        dZ2 = A2 - y
        dW2 = (1/m) * A1.T @ dZ2
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = (dZ2 @ W2.T) * relu_deriv(Z1)
        dW1 = (1/m) * X.T @ dZ1
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        vW1=beta*vW1+dW1; vb1=beta*vb1+db1
        vW2=beta*vW2+dW2; vb2=beta*vb2+db2
        W1-=lr*vW1; b1-=lr*vb1
        W2-=lr*vW2; b2-=lr*vb2
    return W1, b1, W2, b2

def train_2layer_adam(X, y, n_hidden=16, lr=0.01, beta1=0.9,
                      beta2=0.999, epsilon=1e-8, epochs=500):
    W1, b1, W2, b2 = init_weights(X.shape[1], n_hidden)
    mW1=np.zeros_like(W1); mb1=np.zeros_like(b1)
    mW2=np.zeros_like(W2); mb2=np.zeros_like(b2)
    vW1=np.zeros_like(W1); vb1=np.zeros_like(b1)
    vW2=np.zeros_like(W2); vb2=np.zeros_like(b2)
    for t in range(1, epochs+1):
        Z1 = X @ W1 + b1;  A1 = relu(Z1)
        Z2 = A1 @ W2 + b2; A2 = sigmoid(Z2)
        m   = X.shape[0]
        dZ2 = A2 - y
        dW2 = (1/m) * A1.T @ dZ2
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = (dZ2 @ W2.T) * relu_deriv(Z1)
        dW1 = (1/m) * X.T @ dZ1
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        mW1=beta1*mW1+(1-beta1)*dW1; mb1=beta1*mb1+(1-beta1)*db1
        mW2=beta1*mW2+(1-beta1)*dW2; mb2=beta1*mb2+(1-beta1)*db2
        vW1=beta2*vW1+(1-beta2)*dW1**2; vb1=beta2*vb1+(1-beta2)*db1**2
        vW2=beta2*vW2+(1-beta2)*dW2**2; vb2=beta2*vb2+(1-beta2)*db2**2
        mW1_c=mW1/(1-beta1**t); mb1_c=mb1/(1-beta1**t)
        mW2_c=mW2/(1-beta1**t); mb2_c=mb2/(1-beta1**t)
        vW1_c=vW1/(1-beta2**t); vb1_c=vb1/(1-beta2**t)
        vW2_c=vW2/(1-beta2**t); vb2_c=vb2/(1-beta2**t)
        W1-=lr*mW1_c/(np.sqrt(vW1_c)+epsilon)
        b1-=lr*mb1_c/(np.sqrt(vb1_c)+epsilon)
        W2-=lr*mW2_c/(np.sqrt(vW2_c)+epsilon)
        b2-=lr*mb2_c/(np.sqrt(vb2_c)+epsilon)
    return W1, b1, W2, b2

def train_3layer_vanilla(X, y, n_hidden=16, lr=0.1, epochs=500):
    np.random.seed(42)
    W1=np.random.randn(X.shape[1],n_hidden)*np.sqrt(2.0/X.shape[1])
    b1=np.zeros((1,n_hidden))
    W2=np.random.randn(n_hidden,n_hidden)*np.sqrt(2.0/n_hidden)
    b2=np.zeros((1,n_hidden))
    W3=np.random.randn(n_hidden,1)*np.sqrt(2.0/n_hidden)
    b3=np.zeros((1,1))
    for _ in range(epochs):
        Z1=X@W1+b1;   A1=relu(Z1)
        Z2=A1@W2+b2;  A2=relu(Z2)
        Z3=A2@W3+b3;  A3=sigmoid(Z3)
        m=X.shape[0]
        dZ3=A3-y
        dW3=(1/m)*A2.T@dZ3
        db3=(1/m)*np.sum(dZ3,axis=0,keepdims=True)
        dZ2=(dZ3@W3.T)*relu_deriv(Z2)
        dW2=(1/m)*A1.T@dZ2
        db2=(1/m)*np.sum(dZ2,axis=0,keepdims=True)
        dZ1=(dZ2@W2.T)*relu_deriv(Z1)
        dW1=(1/m)*X.T@dZ1
        db1=(1/m)*np.sum(dZ1,axis=0,keepdims=True)
        W1-=lr*dW1; b1-=lr*db1
        W2-=lr*dW2; b2-=lr*db2
        W3-=lr*dW3; b3-=lr*db3
    return W1, b1, W2, b2, W3, b3

# ─────────────────────────────────────────
# 4. BENCHMARK HELPER
# ─────────────────────────────────────────
INFERENCE_REPS = 10000  # repeat inference to get stable timing

def bench_inference_single(infer_fn, reps=INFERENCE_REPS):
    """Time single-sample inference — what edge deployment actually does."""
    t0 = time.perf_counter()
    for _ in range(reps):
        infer_fn(X_single)
    elapsed = time.perf_counter() - t0
    return (elapsed / reps) * 1e6  # microseconds per call

def bench_inference_batch(infer_fn):
    """Time full test-set batch inference."""
    t0 = time.perf_counter()
    result = infer_fn(X_test)
    elapsed = time.perf_counter() - t0
    return elapsed * 1000  # milliseconds

# ─────────────────────────────────────────
# 5. RUN BENCHMARKS
# ─────────────────────────────────────────
results = []

configs = [
    ("Logistic Regression", "logistic"),
    ("NN 2-layer vanilla",  "2v"),
    ("NN 2-layer momentum", "2m"),
    ("NN 2-layer Adam",     "2a"),
    ("NN 3-layer vanilla",  "3v"),
]

for name, key in configs:
    print(f"Training {name}...")

    t0 = time.perf_counter()
    if key == "logistic":
        W, b = train_logistic(X_train, y_train)
        infer = lambda X, W=W, b=b: infer_logistic(X, W, b)
        acc = accuracy_score(y_test, infer(X_test))
    elif key == "2v":
        W1,b1,W2,b2 = train_2layer_vanilla(X_train, y_train)
        infer = lambda X, W1=W1,b1=b1,W2=W2,b2=b2: forward_2layer(X,W1,b1,W2,b2)
        acc = accuracy_score(y_test, infer(X_test))
    elif key == "2m":
        W1,b1,W2,b2 = train_2layer_momentum(X_train, y_train)
        infer = lambda X, W1=W1,b1=b1,W2=W2,b2=b2: forward_2layer(X,W1,b1,W2,b2)
        acc = accuracy_score(y_test, infer(X_test))
    elif key == "2a":
        W1,b1,W2,b2 = train_2layer_adam(X_train, y_train)
        infer = lambda X, W1=W1,b1=b1,W2=W2,b2=b2: forward_2layer(X,W1,b1,W2,b2)
        acc = accuracy_score(y_test, infer(X_test))
    elif key == "3v":
        W1,b1,W2,b2,W3,b3 = train_3layer_vanilla(X_train, y_train)
        infer = lambda X, W1=W1,b1=b1,W2=W2,b2=b2,W3=W3,b3=b3: forward_3layer(X,W1,b1,W2,b2,W3,b3)
        acc = accuracy_score(y_test, infer(X_test))
    train_time = (time.perf_counter() - t0) * 1000  # ms

    us_single = bench_inference_single(infer)
    ms_batch  = bench_inference_batch(infer)

    results.append((name, train_time, us_single, ms_batch, acc))

# ─────────────────────────────────────────
# 6. PRINT RESULTS TABLE
# ─────────────────────────────────────────
print(f"\n{'Model':<24} {'Train(ms)':>10} {'Infer/sample(µs)':>18} {'Infer batch(ms)':>16} {'Acc':>7}")
print("-" * 78)
for name, tr, us, ms, acc in results:
    print(f"{name:<24} {tr:>10.1f} {us:>18.2f} {ms:>16.3f} {acc:>7.4f}")

print(f"\nInference reps per model: {INFERENCE_REPS:,}")
print(f"Batch size: {X_test.shape[0]} samples")