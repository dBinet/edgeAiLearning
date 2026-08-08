import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import onnxruntime as ort
import time

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

test_dataset = datasets.CIFAR10(root='data/', train=False, download=True, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)
model.load_state_dict(torch.load('models/best_model_week16_session3.pth', map_location='cpu'))
model.eval()

ort_session = ort.InferenceSession('models/cnn_week16_session3.onnx')

pytorch_correct = 0
onnx_correct    = 0
agree_count     = 0
total           = 0

with torch.no_grad():
    for images, labels in test_loader:
        pytorch_logits = model(images)
        pytorch_preds  = torch.argmax(pytorch_logits, dim=1)

        onnx_logits = ort_session.run(None, {'input': images.numpy()})[0]
        onnx_preds  = np.argmax(onnx_logits, axis=1)

        labels_np      = labels.numpy()
        pytorch_preds_np = pytorch_preds.numpy()

        pytorch_correct += (pytorch_preds_np == labels_np).sum()
        onnx_correct    += (onnx_preds == labels_np).sum()
        agree_count      += (pytorch_preds_np == onnx_preds).sum()
        total            += labels.size(0)

pytorch_acc = pytorch_correct / total
onnx_acc    = onnx_correct / total
agreement   = agree_count / total

print(f"Full test set: {total} images")
print(f"PyTorch accuracy:      {pytorch_acc:.4f}")
print(f"ONNX Runtime accuracy: {onnx_acc:.4f}")
print(f"Prediction agreement:  {agreement:.4f}  (fraction of images where both engines predict the same class)")
print(f"Accuracy difference:   {abs(pytorch_acc - onnx_acc):.6f}")

batch_size = 64
n_warmup_batches = 5
n_bench_batches  = 50

bench_batch_torch = torch.randn(batch_size, 3, 32, 32)
bench_batch_numpy = bench_batch_torch.numpy()

with torch.no_grad():
    for _ in range(n_warmup_batches):
        model(bench_batch_torch)

with torch.no_grad():
    start = time.perf_counter()
    for _ in range(n_bench_batches):
        model(bench_batch_torch)
    pytorch_batch_time = time.perf_counter() - start

for _ in range(n_warmup_batches):
    ort_session.run(None, {'input': bench_batch_numpy})

start = time.perf_counter()
for _ in range(n_bench_batches):
    ort_session.run(None, {'input': bench_batch_numpy})
onnx_batch_time = time.perf_counter() - start

total_images = batch_size * n_bench_batches
pytorch_throughput = total_images / pytorch_batch_time
onnx_throughput    = total_images / onnx_batch_time

print(f"\nBatch throughput (batch_size={batch_size}, {n_bench_batches} batches, {total_images} images total):")
print(f"  PyTorch:      {pytorch_throughput:.1f} images/sec")
print(f"  ONNX Runtime: {onnx_throughput:.1f} images/sec")
print(f"  Speedup:      {onnx_throughput / pytorch_throughput:.2f}x")