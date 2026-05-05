import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# ── Data ──────────────────────────────────────────────────────────────────────
iris = load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_t = torch.tensor(y_train, dtype=torch.long)

scaler    = StandardScaler()
X_train_n = scaler.fit_transform(X_train)   # fit on train only
X_test_n  = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train_n)
X_test_t  = torch.FloatTensor(X_test_n)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# ── PyTorch models ─────────────────────────────────────────────────────────────
def make_2layer():
    return nn.Sequential(nn.Linear(4,16), nn.ReLU(), nn.Linear(16,3))

def train(model, epochs=1000, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = criterion(model(X_train_t), y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ── Run benchmarks ────────────────────────────────────────────────────────────
print("Training models...")

model_2l = make_2layer()
t0 = time.perf_counter()
train(model_2l)
elapsed = (time.perf_counter() - t0) * 1e3

with torch.no_grad():
    logits = model_2l(X_test_t)
    predicted = torch.argmax(logits, dim=1)
    accuracy = (predicted == y_test_t).float().mean().item()

print(f"Train time: {elapsed:.1f}ms | Accuracy: {accuracy:.4f}")

wrong = (predicted != y_test_t).nonzero(as_tuple=True)[0]
print(f"Misclassified: index {wrong.tolist()}, predicted {predicted[wrong].tolist()}, actual {y_test_t[wrong].tolist()}")