import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import time

# ── Data ──────────────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.FloatTensor(X_train)
X_test_t  = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)

# Single sample for per-sample inference benchmark
x_single_t = X_test_t[0:1]   # shape (1, 2)
x_single_np = X_test[0:1]    # same sample, numpy

# ── From-scratch 2-layer model (Week 12 reference) ────────────────────────────
# Reproducing momentum model — best from Week 12
np.random.seed(42)
def he(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

W1_np = he(2, 16); b1_np = np.zeros((1, 16))
W2_np = he(16, 1); b2_np = np.zeros((1, 1))

def forward_np(X):
    A1 = np.maximum(0, X @ W1_np + b1_np)
    return 1 / (1 + np.exp(-(A1 @ W2_np + b2_np)))

def train_from_scratch():
    W1, b1 = he(2, 16), np.zeros((1, 16))
    W2, b2 = he(16, 1), np.zeros((1, 1))
    v_W1 = v_b1 = v_W2 = v_b2 = 0
    lr, beta = 0.1, 0.9
    X_tr, y_tr = X_train, y_train.reshape(-1, 1)

    for _ in range(1000):
        A1 = np.maximum(0, X_tr @ W1 + b1)
        out = 1 / (1 + np.exp(-(A1 @ W2 + b2)))
        dout = out - y_tr
        dW2 = A1.T @ dout / len(X_tr)
        db2 = dout.mean(axis=0, keepdims=True)
        dA1 = dout @ W2.T
        dZ1 = dA1 * (X_tr @ W1 + b1 > 0)
        dW1 = X_tr.T @ dZ1 / len(X_tr)
        db1 = dZ1.mean(axis=0, keepdims=True)
        v_W1 = beta*v_W1 + dW1; W1 -= lr*v_W1
        v_b1 = beta*v_b1 + db1; b1 -= lr*v_b1
        v_W2 = beta*v_W2 + dW2; W2 -= lr*v_W2
        v_b2 = beta*v_b2 + db2; b2 -= lr*v_b2

    return W1, b1, W2, b2

# ── PyTorch models ─────────────────────────────────────────────────────────────
def make_2layer():
    return nn.Sequential(nn.Linear(2,16), nn.ReLU(), nn.Linear(16,1), nn.Sigmoid())

def make_3layer():
    return nn.Sequential(nn.Linear(2,16), nn.ReLU(),
                         nn.Linear(16,16), nn.ReLU(),
                         nn.Linear(16,1), nn.Sigmoid())

def train_pytorch(model, epochs=1000, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = criterion(model(X_train_t), y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ── Benchmark helpers ─────────────────────────────────────────────────────────
N_INFER = 10_000  # repeat inference this many times for stable µs measurement

def bench_infer_single_pt(model):
    with torch.no_grad():
        # warmup
        for _ in range(100):
            model(x_single_t)
        start = time.perf_counter()
        for _ in range(N_INFER):
            model(x_single_t)
        return (time.perf_counter() - start) * 1e6 / N_INFER  # µs per sample

def bench_infer_batch_pt(model):
    with torch.no_grad():
        for _ in range(100):
            model(X_test_t)
        start = time.perf_counter()
        for _ in range(N_INFER):
            model(X_test_t)
        return (time.perf_counter() - start) * 1e3 / N_INFER  # ms per batch

def bench_infer_single_np(fwd_fn):
    for _ in range(100):
        fwd_fn(x_single_np)
    start = time.perf_counter()
    for _ in range(N_INFER):
        fwd_fn(x_single_np)
    return (time.perf_counter() - start) * 1e6 / N_INFER

def bench_infer_batch_np(fwd_fn):
    for _ in range(100):
        fwd_fn(X_test)
    start = time.perf_counter()
    for _ in range(N_INFER):
        fwd_fn(X_test)
    return (time.perf_counter() - start) * 1e3 / N_INFER

def accuracy_pt(model):
    with torch.no_grad():
        pred = (model(X_test_t) >= 0.5).float()
        return (pred == y_test_t).float().mean().item()

def accuracy_np(fwd_fn):
    pred = (fwd_fn(X_test) >= 0.5).astype(int).flatten()
    return (pred == y_test).mean()

# ── Run benchmarks ────────────────────────────────────────────────────────────
print("Training models...")

# From-scratch
t0 = time.perf_counter()
W1_np, b1_np, W2_np, b2_np = train_from_scratch()
t_scratch = (time.perf_counter() - t0) * 1e3

def fwd_scratch(X):
    A1 = np.maximum(0, X @ W1_np + b1_np)
    return 1 / (1 + np.exp(-(A1 @ W2_np + b2_np)))

# PyTorch 2-layer
model_2l = make_2layer()
t0 = time.perf_counter()
train_pytorch(model_2l)
t_pt2 = (time.perf_counter() - t0) * 1e3

# PyTorch 3-layer
model_3l = make_3layer()
t0 = time.perf_counter()
train_pytorch(model_3l)
t_pt3 = (time.perf_counter() - t0) * 1e3

print("Benchmarking inference...")

results = [
    ("From-scratch 2L (numpy)",
        t_scratch,
        bench_infer_single_np(fwd_scratch),
        bench_infer_batch_np(fwd_scratch),
        accuracy_np(fwd_scratch)),
    ("PyTorch 2-layer",
        t_pt2,
        bench_infer_single_pt(model_2l),
        bench_infer_batch_pt(model_2l),
        accuracy_pt(model_2l)),
    ("PyTorch 3-layer",
        t_pt3,
        bench_infer_single_pt(model_3l),
        bench_infer_batch_pt(model_3l),
        accuracy_pt(model_3l)),
]

print(f"\n{'Model':<26} {'Train(ms)':>10} {'µs/sample':>10} {'Batch(ms)':>10} {'Accuracy':>10}")
print("─" * 70)
for label, tr, us, ms, acc in results:
    print(f"{label:<26} {tr:>10.1f} {us:>10.2f} {ms:>10.3f} {acc:>10.4f}")

print(f"\n── Week 12 reference ──────────────────────────────────────────────────")
wk12 = [
    ("NN 2L momentum (Wk12)",   119.1, 19.27, 0.048, 0.9850),
    ("NN 3L vanilla  (Wk12)",   223.6, 23.61, 0.086, 0.9200),
]
for label, tr, us, ms, acc in wk12:
    print(f"{label:<26} {tr:>10.1f} {us:>10.2f} {ms:>10.3f} {acc:>10.4f}")

print(f"\n── Edge deployment headroom ───────────────────────────────────────────")
print(f"At 100Hz (10,000µs budget per sample):")
for label, _, us, _, _ in results:
    headroom = 10000 - us
    pct = us / 10000 * 100
    print(f"  {label:<26} uses {us:5.2f}µs ({pct:.2f}% of budget) — {headroom:.0f}µs remaining")