import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import time

# --- Rebuild the exact Session 3 architecture ---
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

# --- Export to ONNX ---
dummy_input = torch.randn(1, 3, 32, 32)

torch.onnx.export(
    model,
    dummy_input,
    'models/cnn_week16_session3.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':  {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=13
)

print("Exported to models/cnn_week16_session3.onnx")

# --- Verify structural validity ---
onnx_model = onnx.load('models/cnn_week16_session3.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX model structure check: passed")

# --- Verify numerical correctness ---
test_input = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    pytorch_output = model(test_input).numpy()

ort_session = ort.InferenceSession('models/cnn_week16_session3.onnx')
onnx_output = ort_session.run(None, {'input': test_input.numpy()})[0]

if np.allclose(pytorch_output, onnx_output, atol=1e-5):
    print("Correctness check: PASSED -- PyTorch and ONNX outputs match")
else:
    max_diff = np.abs(pytorch_output - onnx_output).max()
    print(f"Correctness check: FAILED -- max difference: {max_diff}")

# --- Benchmark: single-image inference latency ---
n_warmup = 20
n_runs   = 200
bench_input_torch = torch.randn(1, 3, 32, 32)
bench_input_numpy = bench_input_torch.numpy()

with torch.no_grad():
    for _ in range(n_warmup):
        model(bench_input_torch)

with torch.no_grad():
    start = time.perf_counter()
    for _ in range(n_runs):
        model(bench_input_torch)
    pytorch_time = (time.perf_counter() - start) / n_runs

for _ in range(n_warmup):
    ort_session.run(None, {'input': bench_input_numpy})

start = time.perf_counter()
for _ in range(n_runs):
    ort_session.run(None, {'input': bench_input_numpy})
onnx_time = (time.perf_counter() - start) / n_runs

print(f"\nSingle-image inference latency (avg over {n_runs} runs):")
print(f"  PyTorch:      {pytorch_time * 1000:.3f} ms")
print(f"  ONNX Runtime: {onnx_time * 1000:.3f} ms")
print(f"  Speedup:      {pytorch_time / onnx_time:.2f}x")