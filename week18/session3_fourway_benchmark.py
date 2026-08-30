"""
Week 18 Session 3 — Four-way latency/throughput/size benchmark

Goal: benchmark PyTorch FP32, ONNX Runtime, TFLite-FP32, and TFLite-INT8
side by side on the Pi. Same methodology as Week 17 Session 2's three-way
benchmark (n_warmup=20/n_runs=100 for single-image latency, batch=64/
n_warmup_batches=5/n_bench_batches=50 for throughput), extended with the
two TFLite engines from Sessions 1-2.

Scope: latency, throughput, and size only. Accuracy is NOT recomputed here
-- per project convention, comparison numbers come from a single script
run, so this session's table stays limited to what's actually measured in
this run. Accuracy numbers (PyTorch 80.12%, TFLite-FP32 80.12%,
TFLite-INT8 80.24%) already exist from Session 2's full validation loop
and are referenced separately, not blended into this script's table.

Gotcha (new): TFLite's input tensor shape is fixed at conversion time
(batch=1), unlike PyTorch/ONNX which accept dynamic batch out of the box.
Batched throughput requires interpreter.resize_tensor_input() followed by
a second allocate_tensors() call before the batch will run -- confirmed
this works on both the FP32 and INT8 TFLite models produced in Sessions 1-2.
"""

import os
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

CHECKPOINT_PATH = "models/best_model_week16_session3.pth"
ONNX_PATH = "models/cnn_week16_session3.onnx"
TFLITE_FP32_PATH = "models/tflite_week18/cnn_week16_session3_float32.tflite"
TFLITE_INT8_PATH = "models/tflite_week18/cnn_week16_session3_integer_quant.tflite"

BATCH_SIZE = 64
N_WARMUP = 20
N_RUNS = 100
N_WARMUP_BATCHES = 5
N_BENCH_BATCHES = 50


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pytorch_model(checkpoint_path):
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
        nn.Linear(512, 10),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def load_onnx_model(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    return ort.InferenceSession(onnx_path)


def load_tflite_interpreter(tflite_path):
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


# ---------------------------------------------------------------------------
# Latency benchmarks (single-image, batch=1)
# ---------------------------------------------------------------------------

def benchmark_latency_torch(model, sample_input, n_runs=N_RUNS):
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(sample_input)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(sample_input)
        return (time.perf_counter() - start) / n_runs


def benchmark_latency_onnx(session, sample_input, n_runs=N_RUNS):
    for _ in range(N_WARMUP):
        session.run(None, {"input": sample_input})
    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {"input": sample_input})
    return (time.perf_counter() - start) / n_runs


def benchmark_latency_tflite(interpreter, sample_input_nhwc, n_runs=N_RUNS):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    for _ in range(N_WARMUP):
        interpreter.set_tensor(input_details["index"], sample_input_nhwc)
        interpreter.invoke()

    start = time.perf_counter()
    for _ in range(n_runs):
        interpreter.set_tensor(input_details["index"], sample_input_nhwc)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
    return (time.perf_counter() - start) / n_runs


# ---------------------------------------------------------------------------
# Throughput benchmarks (batched)
# ---------------------------------------------------------------------------

def benchmark_throughput_torch(model, loader, batch_size=BATCH_SIZE):
    images, _ = next(iter(loader))
    with torch.no_grad():
        for _ in range(N_WARMUP_BATCHES):
            model(images)
        start = time.perf_counter()
        for _ in range(N_BENCH_BATCHES):
            model(images)
        elapsed = time.perf_counter() - start
    return (N_BENCH_BATCHES * batch_size) / elapsed


def benchmark_throughput_onnx(session, loader, batch_size=BATCH_SIZE):
    images, _ = next(iter(loader))
    batch_numpy = images.numpy()

    for _ in range(N_WARMUP_BATCHES):
        session.run(None, {"input": batch_numpy})
    start = time.perf_counter()
    for _ in range(N_BENCH_BATCHES):
        session.run(None, {"input": batch_numpy})
    elapsed = time.perf_counter() - start
    return (N_BENCH_BATCHES * batch_size) / elapsed


def benchmark_throughput_tflite(interpreter, loader, batch_size=BATCH_SIZE):
    images, _ = next(iter(loader))
    batch_nhwc = images.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # GOTCHA: TFLite's input shape is fixed at conversion time (batch=1
    # for these models). Must resize + reallocate before a batch will run.
    interpreter.resize_tensor_input(input_details["index"], [batch_size, 32, 32, 3])
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]  # index may shift after realloc
    output_details = interpreter.get_output_details()[0]

    for _ in range(N_WARMUP_BATCHES):
        interpreter.set_tensor(input_details["index"], batch_nhwc)
        interpreter.invoke()
    start = time.perf_counter()
    for _ in range(N_BENCH_BATCHES):
        interpreter.set_tensor(input_details["index"], batch_nhwc)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
    elapsed = time.perf_counter() - start

    # Reset back to batch=1 so a later latency call on the same interpreter
    # (if any) doesn't silently run against a batch=64 input tensor.
    interpreter.resize_tensor_input(input_details["index"], [1, 32, 32, 3])
    interpreter.allocate_tensors()

    return (N_BENCH_BATCHES * batch_size) / elapsed


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------

def get_model_size_kb_torch(model, tmp_path="temp_fp32.pth"):
    torch.save(model.state_dict(), tmp_path)
    size_kb = os.path.getsize(tmp_path) / 1024
    os.remove(tmp_path)
    return size_kb


def get_model_size_kb_onnx(onnx_path):
    size_kb = os.path.getsize(onnx_path) / 1024
    data_file_path = Path(onnx_path + ".data")
    if data_file_path.exists():
        size_kb += os.path.getsize(data_file_path) / 1024
    return size_kb


def get_model_size_kb_tflite(tflite_path):
    return os.path.getsize(tflite_path) / 1024


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_dataset = datasets.CIFAR10(
        root="data/", train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    pytorch_model = load_pytorch_model(CHECKPOINT_PATH)
    onnx_session = load_onnx_model(ONNX_PATH)
    tflite_fp32 = load_tflite_interpreter(TFLITE_FP32_PATH)
    tflite_int8 = load_tflite_interpreter(TFLITE_INT8_PATH)

    # --- Single-image latency ---
    bench_input_torch = torch.randn(1, 3, 32, 32)
    bench_input_numpy = bench_input_torch.numpy()
    bench_input_nhwc = bench_input_numpy.transpose(0, 2, 3, 1)

    pytorch_latency = benchmark_latency_torch(pytorch_model, bench_input_torch)
    onnx_latency = benchmark_latency_onnx(onnx_session, bench_input_numpy)
    tflite_fp32_latency = benchmark_latency_tflite(tflite_fp32, bench_input_nhwc)
    tflite_int8_latency = benchmark_latency_tflite(tflite_int8, bench_input_nhwc)

    print("--- Single-image latency (batch=1) ---")
    print(f"PyTorch FP32:   {pytorch_latency * 1000:.3f} ms")
    print(f"ONNX Runtime:   {onnx_latency * 1000:.3f} ms")
    print(f"TFLite FP32:    {tflite_fp32_latency * 1000:.3f} ms")
    print(f"TFLite INT8:    {tflite_int8_latency * 1000:.3f} ms")

    # --- Batched throughput ---
    pytorch_throughput = benchmark_throughput_torch(pytorch_model, test_loader)
    onnx_throughput = benchmark_throughput_onnx(onnx_session, test_loader)
    tflite_fp32_throughput = benchmark_throughput_tflite(tflite_fp32, test_loader)
    tflite_int8_throughput = benchmark_throughput_tflite(tflite_int8, test_loader)

    print(f"\n--- Batched throughput (batch={BATCH_SIZE}) ---")
    print(f"PyTorch FP32:   {pytorch_throughput:.1f} img/s")
    print(f"ONNX Runtime:   {onnx_throughput:.1f} img/s")
    print(f"TFLite FP32:    {tflite_fp32_throughput:.1f} img/s")
    print(f"TFLite INT8:    {tflite_int8_throughput:.1f} img/s")

    # --- Size ---
    pytorch_size = get_model_size_kb_torch(pytorch_model)
    onnx_size = get_model_size_kb_onnx(ONNX_PATH)
    tflite_fp32_size = get_model_size_kb_tflite(TFLITE_FP32_PATH)
    tflite_int8_size = get_model_size_kb_tflite(TFLITE_INT8_PATH)

    print(f"\n--- Size ---")
    print(f"PyTorch FP32:   {pytorch_size:.1f} KB")
    print(f"ONNX Runtime:   {onnx_size:.1f} KB")
    print(f"TFLite FP32:    {tflite_fp32_size:.1f} KB")
    print(f"TFLite INT8:    {tflite_int8_size:.1f} KB")