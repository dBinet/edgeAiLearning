import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/winequality-red.csv', sep=';')

print(df.shape)
print(df.isnull().sum())
print(df['quality'].value_counts().sort_index())  # class distribution

X = df.drop('quality', axis=1).values 
y = df['quality'].values  

y = y - y.min() # set quality from 3-8 to 0-5 for cross entropy loss

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler    = StandardScaler()
X_train_n = scaler.fit_transform(X_train)   # fit on train only
X_test_n  = scaler.transform(X_test)

X_train_t = torch.FloatTensor(X_train_n)
X_test_t  = torch.FloatTensor(X_test_n)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

# ── PyTorch models ─────────────────────────────────────────────────────────────
def make_3layer():
    return nn.Sequential(nn.Linear(11,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,6))

def make_3layer_dropout(p=0.4):
    return nn.Sequential(nn.Linear(11,64), nn.ReLU(), nn.Dropout(p), nn.Linear(64,64), nn.ReLU(), nn.Dropout(p), nn.Linear(64,6))

def train(model, epochs=1000, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        loss = criterion(model(X_train_t), y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test_t), y_test_t)
            print(f"  epoch {epoch:4d} | train loss: {loss.item():.4f} | test loss: {test_loss.item():.4f}")

def evaluate(model):
    model.eval()
    with torch.no_grad():
        train_acc = (torch.argmax(model(X_train_t), dim=1) == y_train_t).float().mean().item()
        test_acc  = (torch.argmax(model(X_test_t),  dim=1) == y_test_t).float().mean().item()
    print(f"Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f} | Gap: {train_acc - test_acc:.4f}")

# ── Run benchmarks ────────────────────────────────────────────────────────────
print("Training models...")

model_3l = make_3layer()
print("Training model 3L")
train(model_3l)
evaluate(model_3l)

model_3l_dropout = make_3layer_dropout()
print("Training model 3L with dropout")
train(model_3l_dropout)
evaluate(model_3l_dropout)
