import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import time

# --- Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='data/', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root='data/', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# --- Model ---
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1568, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

def conv2d_numpy(x, W, b, padding=1):
    # x: (C_in, H, W)
    # W: (C_out, C_in, kH, kW)
    C_out, C_in, kH, kW = W.shape
    C_in_x, H, W_size = x.shape

    # pad input
    x_pad = np.pad(x, ((0,0), (padding,padding), (padding,padding)))
    H_out = H
    W_out = W_size

    out = np.zeros((C_out, H_out, W_out))
    for f in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                out[f, i, j] = np.sum(x_pad[:, i:i+kH, j:j+kW] * W[f]) + b[f]
    return out

def maxpool2d_numpy(x, size=2):
    # x: (C, H, W)
    C, H, W = x.shape
    out = np.zeros((C, H//size, W//size))
    for c in range(C):
        for i in range(H//size):
            for j in range(W//size):
                out[c, i, j] = np.max(x[c, i*size:i*size+size, j*size:j*size+size])
    return out

def relu(x):
    return np.maximum(0, x)

def numpy_cnn_forward(x_numpy, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2):
    # x_numpy: (1, 28, 28) — single image, no batch dimension
    z1   = conv2d_numpy(x_numpy, W_conv1, b_conv1)   # (16, 28, 28)
    a1   = relu(z1)
    p1   = maxpool2d_numpy(a1)                        # (16, 14, 14)

    z2   = conv2d_numpy(p1, W_conv2, b_conv2)         # (32, 14, 14)
    a2   = relu(z2)
    p2   = maxpool2d_numpy(a2)                        # (32, 7, 7)

    flat = p2.flatten()                               # (1568,)

    z3   = flat @ W_fc1 + b_fc1                       # (128,)
    a3   = relu(z3)
    z4   = a3 @ W_fc2 + b_fc2                         # (10,)
    return z4


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            preds   = torch.argmax(model(images), dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs    = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        logits = model(images)
        loss   = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss   = running_loss / len(train_loader)

    # --- save model ---
    torch.save(model.state_dict(), 'models/cnn_mnist.pth')
    # --- measure test loss ---
    model.eval()

    # Conv layers — 4D: (out_channels, in_channels, kH, kW)
    W_conv1 = model[0].weight.detach().numpy()   # (16, 1, 3, 3)
    b_conv1 = model[0].bias.detach().numpy()     # (16,)

    W_conv2 = model[3].weight.detach().numpy()   # (32, 16, 3, 3)
    b_conv2 = model[3].bias.detach().numpy()     # (32,)

    # Linear layers — same as Week 14 (note index jump past Flatten)
    W_fc1 = model[7].weight.detach().numpy().T.copy()   # (1568, 128)
    b_fc1 = model[7].bias.detach().numpy().copy()

    W_fc2 = model[9].weight.detach().numpy().T.copy()   # (128, 10)
    b_fc2 = model[9].bias.detach().numpy().copy()


    # Grab one test image
    images, labels = next(iter(test_loader))
    x_single = images[0].numpy()   # (1, 28, 28)

    # PyTorch
    model.eval()
    with torch.no_grad():
        pt_out = model(images[0].unsqueeze(0)).numpy()[0]

    # Numpy
    np_out = numpy_cnn_forward(x_single, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)

    print(f"Max output diff: {np.max(np.abs(pt_out - np_out)):.2e}")
    print(f"PyTorch pred:    {np.argmax(pt_out)}")
    print(f"Numpy pred:      {np.argmax(np_out)}")

    N = 100   # single-sample trials

    # PyTorch single-sample
    model.eval()
    x_tensor = images[0].unsqueeze(0)
    start = time.perf_counter()
    for _ in range(N):
        with torch.no_grad():
            _ = model(x_tensor)
    pt_single = (time.perf_counter() - start) / N * 1000

    # Numpy single-sample
    start = time.perf_counter()
    for _ in range(N):
        _ = numpy_cnn_forward(x_single, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)
    np_single = (time.perf_counter() - start) / N * 1000

    print(f"PyTorch single-sample: {pt_single:.2f}ms")
    print(f"Numpy single-sample:   {np_single:.2f}ms")
    print(f"Speedup: {pt_single/np_single:.4f}x")

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")