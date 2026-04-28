import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import time

# ── Data ──────────────────────────────────────────────────────────────────────
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.FloatTensor(X_train)
X_test_t  = torch.FloatTensor(X_test)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
y_test_t  = torch.FloatTensor(y_test).unsqueeze(1)

# ── Model factory ─────────────────────────────────────────────────────────────
# In Week 12, changing architecture meant rewriting forward() and all of backprop
# Here it's a list of layers — that's the entire change
def make_2layer(hidden):
    return nn.Sequential(
        nn.Linear(2, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
        nn.Sigmoid()
    )

def make_3layer(h1, h2):
    return nn.Sequential(
        nn.Linear(2, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, 1),
        nn.Sigmoid()
    )

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# ── Training ──────────────────────────────────────────────────────────────────
def train_and_evaluate(model, epochs=1000, lr=0.01):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    for epoch in range(epochs):
        y_pred = model(X_train_t)
        loss = criterion(y_pred, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start

    with torch.no_grad():
        pred = model(X_test_t)
        predicted = (pred >= 0.5).float()
        acc = (predicted == y_test_t).float().mean().item()

    return elapsed, acc, count_params(model)

# ── Sweep ─────────────────────────────────────────────────────────────────────
experiments = [
    ("2-layer,  2 hidden", make_2layer(2)),
    ("2-layer,  8 hidden", make_2layer(8)),
    ("2-layer, 16 hidden", make_2layer(16)),
    ("2-layer, 32 hidden", make_2layer(32)),
    ("3-layer, 16+16    ", make_3layer(16, 16)),
]

print(f"\n{'Architecture':<25} {'Params':>7} {'Time':>8} {'Accuracy':>10}")
print("─" * 55)

wk12_times = [0.37, 0.50, 0.68, 1.05, 1.28]
wk12_accs  = [0.8750, 0.9800, 0.9800, 0.9800, 0.9900]

results = []
for label, model in experiments:
    elapsed, acc, params = train_and_evaluate(model)
    results.append((label, params, elapsed, acc))
    print(f"{label:<25} {params:>7} {elapsed:>7.2f}s {acc:>10.4f}")

# ── Side-by-side comparison with Week 12 ─────────────────────────────────────
print(f"\n{'─'*75}")
print(f"{'Architecture':<25} {'Wk12 t':>8} {'Wk13 t':>8} {'Wk12 acc':>10} {'Wk13 acc':>10}")
print(f"{'─'*75}")
for i, (label, params, elapsed, acc) in enumerate(results):
    print(f"{label:<25} {wk12_times[i]:>7.2f}s {elapsed:>7.2f}s "
          f"{wk12_accs[i]:>10.4f} {acc:>10.4f}")

print(f"\nKey question: how much does the framework cost on this problem size?")