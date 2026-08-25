"""
Week 18 Session 1 — Convert the CIFAR-10 CNN to TensorFlow Lite (FP32)

Goal: get the Week 16 Session 3 checkpoint into .tflite form and verify it
produces the same outputs as the PyTorch model. No quantization yet — that's
Session 2. This session is conversion + correctness only.

Path: PyTorch -> ONNX (already exported in Week 16 Session 4) -> TF SavedModel
-> TFLite, via onnx2tf. PyTorch no longer exports directly to TFLite, so the
ONNX file is the bridge.

Install (one-time, on Pi):
    pip install tensorflow onnx2tf onnx onnxruntime sng4onnx onnx_graphsurgeon --break-system-packages
Note: this pulls in full desktop TensorFlow, which is a heavier footprint
than anything installed so far this curriculum (torch/onnxruntime are much
smaller). If install size/RAM on the Pi becomes a problem, that's worth
flagging back — flag it, don't just push through it.
"""

import numpy as np
import torch
import torch.nn as nn
import onnx2tf

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# --- Rebuild the exact Week 16 Session 3 architecture ---
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

# --- Convert: reuse the Week 16 Session 4 ONNX export (legacy exporter, dynamo=False) ---
# Same standing rule as Week 17 Session 3: dynamo=False avoids the Flatten ->
# Reshape+Constant representation that trips up downstream tooling.
onnx_path = 'models/cnn_week16_session3.onnx'

onnx2tf.convert(
    input_onnx_file_path=onnx_path,
    output_folder_path='models/tflite_week18',
    output_signaturedefs=True,
    non_verbose=True,
)
# This writes both models/tflite_week18/cnn_week16_session3_float32.tflite
# and a float16 variant. We only use the float32 one this session — the
# float16 file can be ignored for now, it's not the INT8 quantization
# Session 2 will do.
tflite_path = 'models/tflite_week18/cnn_week16_session3_float32.tflite'

print(f"Converted: {onnx_path} -> {tflite_path}")

# --- Load TFLite model and inspect the interface ---
interpreter = Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(f"\nTFLite input : {input_details['shape']} ({input_details['dtype'].__name__})")
print(f"TFLite output: {output_details['shape']} ({output_details['dtype'].__name__})")

# GOTCHA: TFLite/TensorFlow's native layout is NHWC (batch, height, width,
# channels). PyTorch and ONNX use NCHW. onnx2tf does NOT change the model's
# math, but the interpreter now expects channels-last input, so every input
# tensor fed to the interpreter must be transposed before use:
#   NCHW (1, 3, 32, 32) -> NHWC (1, 32, 32, 3)
# Skipping this produces a shape mismatch error, or worse, silently wrong
# output if shapes happen to be compatible by coincidence.

# --- Correctness check: single random input, PyTorch vs TFLite ---
test_input = torch.randn(1, 3, 32, 32)

with torch.no_grad():
    pytorch_output = model(test_input).numpy()

tflite_input = test_input.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC
interpreter.set_tensor(input_details['index'], tflite_input)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details['index'])

if np.allclose(pytorch_output, tflite_output, atol=1e-4):
    max_diff = np.abs(pytorch_output - tflite_output).max()
    print(f"\nCorrectness check: PASSED -- max diff: {max_diff:.2e}")
else:
    max_diff = np.abs(pytorch_output - tflite_output).max()
    print(f"\nCorrectness check: FAILED -- max diff: {max_diff:.2e}")

# --- Size check ---
import os
onnx_size = os.path.getsize(onnx_path) / 1024
tflite_size = os.path.getsize(tflite_path) / 1024
print(f"\nONNX size:   {onnx_size:.1f} KB")
print(f"TFLite size: {tflite_size:.1f} KB")