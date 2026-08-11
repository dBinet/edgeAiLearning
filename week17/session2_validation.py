"""
Week 17 Session 2 — Quantization Accuracy/Speed Tradeoff

Goal: full test-set validation of INT8 model (same rigor as Week 16 Session 5's
ONNX validation) + three-way benchmark: FP32 PyTorch vs FP32 ONNX vs INT8.

Deliverable: prediction-agreement table, latency/size comparison table.
Self-contained — rebuilds the INT8 pipeline inline, doesn't import Session 1.
"""

import torch
import torch.nn as nn
import torch.quantization
import onnxruntime as ort
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time
import os
import onnx
from pathlib import Path

# ---------------------------------------------------------------------------
# Transforms — clean only, no augmentation anywhere in this script
# ---------------------------------------------------------------------------

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

# ---------------------------------------------------------------------------
# Model definition — identical to Session 1
# ---------------------------------------------------------------------------

class QuantizedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = nn.Sequential(
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
            nn.Linear(512, 10),
        )
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


# ---------------------------------------------------------------------------
# Build FP32 model
# ---------------------------------------------------------------------------

def load_fp32_model(checkpoint_path):
    model = QuantizedCNN()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.model.load_state_dict(checkpoint)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Build INT8 model — same pipeline as Session 1, inlined here for
# self-containment
# ---------------------------------------------------------------------------

def build_int8_model(checkpoint_path, calibration_loader):
    model = load_fp32_model(checkpoint_path)

    torch.quantization.fuse_modules(model.model, [
        ['0', '1', '2'],
        ['4', '5', '6'],
        ['8', '9', '10'],
    ], inplace=True)

    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)

    model.eval()
    with torch.no_grad():
        for images, _ in calibration_loader:
            model(images)

    torch.quantization.convert(model, inplace=True)
    return model


# ---------------------------------------------------------------------------
# Load ONNX model — from Week 16 Session 4's export
# ---------------------------------------------------------------------------

def load_onnx_model(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    return ort.InferenceSession(onnx_path)



# ---------------------------------------------------------------------------
# Full test-set validation — THE core deliverable of this session.
# Track FP32 accuracy, INT8 accuracy, ONNX accuracy, AND pairwise agreement.
# Reference pattern: Week 16 Session 5's validation loop.
# ---------------------------------------------------------------------------

def validate_all(fp32_model, int8_model, onnx_session, test_loader):
    fp32_correct = int8_correct = onnx_correct = 0
    fp32_int8_agree = fp32_onnx_agree = 0
    total = 0

    onnx_input_name = onnx_session.get_inputs()[0].name

    with torch.no_grad():
        for images, labels in test_loader:
            fp32_preds = torch.argmax(fp32_model(images), dim=1)
            int8_preds = torch.argmax(int8_model(images), dim=1)

            onnx_output = onnx_session.run(None, {onnx_input_name: images.numpy()})[0]
            onnx_preds = np.argmax(onnx_output, axis=1)

            fp32_correct += (fp32_preds == labels).sum().item()
            int8_correct += (int8_preds == labels).sum().item()
            onnx_correct += (onnx_preds == labels.numpy()).sum().item()

            fp32_int8_agree += (fp32_preds == int8_preds).sum().item()
            fp32_onnx_agree += (fp32_preds.numpy() == onnx_preds).sum().item()

            total += labels.size(0)

    return {
        'fp32_acc': fp32_correct / total,
        'int8_acc': int8_correct / total,
        'onnx_acc': onnx_correct / total,
        'fp32_int8_agreement': fp32_int8_agree / total,
        'fp32_onnx_agreement': fp32_onnx_agree / total,
    }


# ---------------------------------------------------------------------------
# Latency benchmark — single-image AND batched, same as Week 16 Session 4/5.
# Measure FP32 PyTorch, INT8 PyTorch, ONNX Runtime.
# ---------------------------------------------------------------------------

def benchmark_latency_torch(model, sample_input, n_runs=100):
    n_warmup = 20

    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample_input)

    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(n_runs):
            model(sample_input)
        model_time = (time.perf_counter() - start) / n_runs
    
    return model_time

def benchmark_latency_onnx(session, sample_input, n_runs=100):
    n_warmup = 20

    for _ in range(n_warmup):
        session.run(None, {'input': sample_input})

    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {'input': sample_input})
    onnx_time = (time.perf_counter() - start) / n_runs
    
    return onnx_time


def benchmark_throughput_torch(model, loader):
    batch_size = 64
    n_warmup_batches = 5
    n_bench_batches  = 50

    images, labels = next(iter(loader))

    with torch.no_grad():
        for _ in range(n_warmup_batches):
            model(images)

    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(n_bench_batches):
            model(images)
    model_batch_time = time.perf_counter() - start
    
    return (n_bench_batches * batch_size) / model_batch_time

def benchmark_throughput_onnx(session, loader):
    batch_size = 64
    n_warmup_batches = 5
    n_bench_batches  = 50

    images, labels = next(iter(loader))
    bench_batch_numpy = images.numpy()

    for _ in range(n_warmup_batches):
        session.run(None, {'input': bench_batch_numpy})

    start = time.perf_counter()
    for _ in range(n_bench_batches):
        session.run(None, {'input': bench_batch_numpy})
    onnx_batch_time = time.perf_counter() - start
    
    return (n_bench_batches * batch_size) / onnx_batch_time


# ---------------------------------------------------------------------------
# Size comparison — reuse Session 1's file-size pattern
# ---------------------------------------------------------------------------

def get_model_size_kb_torch(model, tmp_path):
    torch.save(model.state_dict(), tmp_path)
    size_kb = os.path.getsize(tmp_path) / 1024
    os.remove(tmp_path)  # cleanup temp file
    return size_kb

def get_model_size_kb_onnx(onnx_path):
    size_kb = os.path.getsize(onnx_path) / 1024
    data_file_path = Path(onnx_path + ".data")
    if data_file_path.exists():
        size_kb += os.path.getsize(data_file_path) / 1024
    return size_kb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    calibration_dataset = datasets.CIFAR10(
        root='data/', train=True, download=True, transform=test_transform
    )
    calibration_loader = DataLoader(
        Subset(calibration_dataset, range(500)), batch_size=64, shuffle=False
    )

    test_dataset = datasets.CIFAR10(
        root='data/', train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    fp32_model = load_fp32_model("models/best_model_week16_session3.pth")
    int8_model = build_int8_model("models/best_model_week16_session3.pth", calibration_loader)
    onnx_session = load_onnx_model("models/cnn_week16_session3.onnx")


    results = validate_all(fp32_model, int8_model, onnx_session, test_loader)

    print(f"FP32 PyTorch accuracy: {results['fp32_acc']:.4f}")
    print(f"INT8 PyTorch accuracy: {results['int8_acc']:.4f}")
    print(f"ONNX Runtime accuracy: {results['onnx_acc']:.4f}")
    print(f"FP32 and INT8 Prediction agreement:  {results['fp32_int8_agreement']:.4f}  (fraction of images where both engines predict the same class)")
    print(f"FP32 and ONNX Prediction agreement:  {results['fp32_onnx_agreement']:.4f}  (fraction of images where both engines predict the same class)")

    bench_input_torch = torch.randn(1, 3, 32, 32)
    bench_input_numpy = bench_input_torch.numpy()


    FP32_model_latency_time = benchmark_latency_torch(fp32_model, bench_input_torch, n_runs=100)
    INT8_model_latency_time = benchmark_latency_torch(int8_model, bench_input_torch, n_runs=100)
    ONNX_model_latency_time = benchmark_latency_onnx(onnx_session, bench_input_numpy, n_runs=100)

    print(f"  FP32 PyTorch Latency time :      {FP32_model_latency_time * 1000:.3f} ms")
    print(f"  INT8 PyTorch Latency time :      {INT8_model_latency_time * 1000:.3f} ms")
    print(f"  ONNX Latency time : {ONNX_model_latency_time * 1000:.3f} ms")

    FP32_model_throughput = benchmark_throughput_torch(fp32_model, test_loader)
    INT8_model_throughput = benchmark_throughput_torch(int8_model, test_loader)
    ONNX_model_throughput = benchmark_throughput_onnx(onnx_session, test_loader)

    print(f"  FP32 PyTorch Throughput :      {FP32_model_throughput:.3f} img/s")
    print(f"  INT8 PyTorch Throughput :      {INT8_model_throughput:.3f} img/s")
    print(f"  ONNX Batch Throughput : {ONNX_model_throughput:.3f} img/s")

    FP32_model_size = get_model_size_kb_torch(fp32_model, "temp_fp32.pth")
    INT8_model_size = get_model_size_kb_torch(int8_model, "temp_int8.pth")
    ONNX_model_size = get_model_size_kb_onnx("models/cnn_week16_session3.onnx")

    print(f"FP32 model size: {FP32_model_size:.1f} KB")
    print(f"INT8 model size: {INT8_model_size:.1f} KB")
    print(f"ONNX model size: {ONNX_model_size:.1f} KB")

