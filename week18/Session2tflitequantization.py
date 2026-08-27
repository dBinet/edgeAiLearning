"""
Week 18 Session 2 — TFLite post-training INT8 quantization + full validation

Goal: quantize the TFLite model from Session 1 to INT8 using onnx2tf's
built-in quantization path, then validate accuracy/agreement across the
full 10,000-image CIFAR-10 test set (same rigor as Week 17 Session 2's
FP32/INT8/ONNX three-way check) and compare size/accuracy against Week 17's
PyTorch static quant (80.00%, 1130.5 KB) and ONNX dynamic quant
(79.83%, 1129.9 KB).

onnx2tf's -oiqt (output_integer_quantized_tflite) path produces several
INT8 variants in one pass. This session uses cnn_integer_quant.tflite:
weights AND activations are int8 internally, but the model still exposes
a float32 input/output interface -- the same shape of comparison as your
PyTorch/ONNX quantized models, so accuracy numbers are apples-to-apples.

onnx2tf also produces cnn_full_integer_quant.tflite (int8 in, int8 out,
no float interface at all). That's the form some NPUs/microcontrollers
actually require, but comparing it needs manual quantize/dequantize of
inputs and outputs using its scale/zero-point -- deliberately deferred to
a later session so this one stays focused on accuracy/size, not I/O
quantization mechanics.

Requires Session 1's dependencies already installed (tensorflow, onnx2tf,
ai_edge_litert/tf.lite).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import onnx2tf

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

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

# --- Build calibration data: 500 unaugmented training images, RAW [0,1] ---
# Matches Week 17 Session 1's "calibrated on 500 unaugmented training images."
# IMPORTANT: onnx2tf's -oiqt path expects calibration data pre-normalized to
# [0, 1] (i.e. just ToTensor, no Normalize) and applies (x - mean) / std
# itself internally using the mean/std you pass in. Passing already-CIFAR-
# normalized data here would double-normalize and silently miscalibrate.
raw_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(
    root='data/', train=True, download=True, transform=raw_transform
)

calib_images = torch.stack([train_dataset[i][0] for i in range(500)])  # (500, 3, 32, 32), [0,1]
calib_nhwc = calib_images.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC, onnx2tf's expected layout
calib_path = 'models/calib_cifar_raw.npy'
np.save(calib_path, calib_nhwc.astype(np.float32))
print(f"Calibration data: {calib_nhwc.shape}, range [{calib_nhwc.min():.3f}, {calib_nhwc.max():.3f}]")

# --- Convert with INT8 quantization enabled ---
onnx_path = 'models/cnn_week16_session3.onnx'

onnx2tf.convert(
    input_onnx_file_path=onnx_path,
    output_folder_path='models/tflite_week18',   # same folder as Session 1
    output_signaturedefs=True,
    output_integer_quantized_tflite=True,
    custom_input_op_name_np_data_path=[
        ['input', calib_path,
         [[[list(CIFAR_MEAN)]]],
         [[[list(CIFAR_STD)]]]]
    ],
    non_verbose=True,
)

fp32_path = 'models/tflite_week18/cnn_week16_session3_float32.tflite'
int8_path = 'models/tflite_week18/cnn_week16_session3_integer_quant.tflite'
print(f"\nConverted INT8 model: {int8_path}")

# --- Load both TFLite interpreters ---
fp32_interp = Interpreter(model_path=fp32_path)
fp32_interp.allocate_tensors()
fp32_in = fp32_interp.get_input_details()[0]
fp32_out = fp32_interp.get_output_details()[0]

int8_interp = Interpreter(model_path=int8_path)
int8_interp.allocate_tensors()
int8_in = int8_interp.get_input_details()[0]
int8_out = int8_interp.get_output_details()[0]

print(f"\nTFLite INT8 input : {int8_in['shape']} ({int8_in['dtype'].__name__})")
print(f"TFLite INT8 output: {int8_out['shape']} ({int8_out['dtype'].__name__})")
# Float32 in/out confirms this is the "float interface, int8 internals"
# variant -- comparable to PyTorch/ONNX quantized models, not the fully
# integer cnn_full_integer_quant.tflite variant.

# --- Full test-set validation: PyTorch vs TFLite-FP32 vs TFLite-INT8 ---
# Single loop tracking all three simultaneously, same reasoning as Week 17
# Session 2: separate passes risk desync, and aggregate accuracy alone
# can't surface per-image disagreement the way pairwise agreement can.
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
test_dataset = torchvision.datasets.CIFAR10(
    root='data/', train=False, download=True, transform=test_transform
)

pytorch_correct = fp32_correct = int8_correct = 0
pt_fp32_agree = pt_int8_agree = fp32_int8_agree = 0
total = len(test_dataset)

for i in range(total):
    image, label = test_dataset[i]
    image_batch = image.unsqueeze(0)  # (1, 3, 32, 32)

    with torch.no_grad():
        pytorch_pred = torch.argmax(model(image_batch), dim=1).item()

    nhwc = image_batch.numpy().transpose(0, 2, 3, 1)

    fp32_interp.set_tensor(fp32_in['index'], nhwc)
    fp32_interp.invoke()
    fp32_pred = int(np.argmax(fp32_interp.get_tensor(fp32_out['index']), axis=1)[0])

    int8_interp.set_tensor(int8_in['index'], nhwc)
    int8_interp.invoke()
    int8_pred = int(np.argmax(int8_interp.get_tensor(int8_out['index']), axis=1)[0])

    pytorch_correct += (pytorch_pred == label)
    fp32_correct += (fp32_pred == label)
    int8_correct += (int8_pred == label)
    pt_fp32_agree += (pytorch_pred == fp32_pred)
    pt_int8_agree += (pytorch_pred == int8_pred)
    fp32_int8_agree += (fp32_pred == int8_pred)

    if (i + 1) % 2000 == 0:
        print(f"  ...{i + 1}/{total} images processed")

print(f"\n--- Full test-set results ({total} images) ---")
print(f"PyTorch accuracy:    {pytorch_correct / total:.4%}")
print(f"TFLite-FP32 accuracy:{fp32_correct / total:.4%}")
print(f"TFLite-INT8 accuracy:{int8_correct / total:.4%}")
print(f"\nPyTorch <-> TFLite-FP32 agreement: {pt_fp32_agree / total:.4%}")
print(f"PyTorch <-> TFLite-INT8 agreement: {pt_int8_agree / total:.4%}")
print(f"TFLite-FP32 <-> TFLite-INT8 agreement: {fp32_int8_agree / total:.4%}")

# --- Size comparison ---
fp32_size = os.path.getsize(fp32_path) / 1024
int8_size = os.path.getsize(int8_path) / 1024
print(f"\nTFLite FP32 size: {fp32_size:.1f} KB")
print(f"TFLite INT8 size: {int8_size:.1f} KB ({fp32_size / int8_size:.2f}x reduction)")

print("\n--- Comparison to Week 17 quantization results ---")
print("PyTorch static INT8 (Week 17 S1): 80.00% accuracy, 1130.5 KB")
print("ONNX dynamic INT8    (Week 17 S3): 79.83% accuracy, 1129.9 KB")
print(f"TFLite INT8          (Week 18 S2): {int8_correct / total:.2%} accuracy, {int8_size:.1f} KB")