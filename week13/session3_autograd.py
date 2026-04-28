import torch
import torch.nn as nn
import numpy as np

# ── Exact Week 12 Session 3 values ────────────────────────────────────────────
W1 = np.array([[0.5, -0.3], [0.2, 0.8]])   # shape (2,2)
W2 = np.array([[0.7], [-0.4]])              # shape (2,1)
b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))
x  = np.array([[1.0, 0.5]])                 # shape (1,2)
y  = np.array([[1.0]])                      # shape (1,1)

# ── From-scratch forward pass (your Week 12 code) ─────────────────────────────
Z1     = x @ W1 + b1                        # pre-activation hidden
A1     = np.maximum(0, Z1)                  # ReLU
Z2     = A1 @ W2 + b2                       # pre-activation output
output = 1 / (1 + np.exp(-Z2))             # Sigmoid
loss_np = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))

print("=" * 55)
print("FROM SCRATCH — Forward pass")
print("=" * 55)
print(f"Z1     = {Z1}")
print(f"A1     = {A1}")
print(f"Z2     = {Z2}")
print(f"output = {output}")
print(f"loss   = {loss_np:.6f}")

# ── From-scratch backward pass (your Week 12 gradients) ───────────────────────
dL_dout = output - y                        # d(BCE)/d(sigmoid output)
dL_dW2  = A1.T @ dL_dout
dL_db2  = np.sum(dL_dout, axis=0, keepdims=True)
dL_dA1  = dL_dout @ W2.T
dL_dZ1  = dL_dA1 * (Z1 > 0).astype(float)  # ReLU derivative
dL_dW1  = x.T @ dL_dZ1
dL_db1  = np.sum(dL_dZ1, axis=0, keepdims=True)

print("\n" + "=" * 55)
print("FROM SCRATCH — Gradients")
print("=" * 55)
print(f"dW2 = {dL_dW2}")
print(f"db2 = {dL_db2}")
print(f"dW1 = {dL_dW1}")
print(f"db1 = {dL_db1}")

# ── PyTorch — same weights, same input ────────────────────────────────────────
# requires_grad=True tells PyTorch: track every operation on this tensor
# so it can compute gradients when we call .backward()
W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
b1_t = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
b2_t = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
x_t  = torch.tensor(x,  dtype=torch.float64)   # input: no grad needed
y_t  = torch.tensor(y,  dtype=torch.float64)

# Forward pass — identical structure, PyTorch records every operation
Z1_t     = x_t @ W1_t + b1_t
A1_t     = torch.relu(Z1_t)
Z2_t     = A1_t @ W2_t + b2_t
out_t    = torch.sigmoid(Z2_t)
loss_t   = -(y_t * torch.log(out_t) + (1 - y_t) * torch.log(1 - out_t)).mean()

print("\n" + "=" * 55)
print("PYTORCH — Forward pass")
print("=" * 55)
print(f"Z1     = {Z1_t.detach().numpy()}")
print(f"A1     = {A1_t.detach().numpy()}")
print(f"Z2     = {Z2_t.detach().numpy()}")
print(f"output = {out_t.detach().numpy()}")
print(f"loss   = {loss_t.item():.6f}")

# This single line replaces your entire manual backward pass
loss_t.backward()

print("\n" + "=" * 55)
print("PYTORCH — Gradients (via .backward())")
print("=" * 55)
print(f"dW2 = {W2_t.grad.numpy()}")
print(f"db2 = {b2_t.grad.numpy()}")
print(f"dW1 = {W1_t.grad.numpy()}")
print(f"db1 = {b1_t.grad.numpy()}")

# ── Verification ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("VERIFICATION — Max absolute difference")
print("=" * 55)
print(f"dW2 diff: {np.max(np.abs(dL_dW2 - W2_t.grad.numpy())):.2e}")
print(f"db2 diff: {np.max(np.abs(dL_db2 - b2_t.grad.numpy())):.2e}")
print(f"dW1 diff: {np.max(np.abs(dL_dW1 - W1_t.grad.numpy())):.2e}")
print(f"db1 diff: {np.max(np.abs(dL_db1 - b1_t.grad.numpy())):.2e}")
print(f"loss diff:{np.abs(loss_np - loss_t.item()):.2e}")
print("\nAll differences should be < 1e-10 (floating point only)")