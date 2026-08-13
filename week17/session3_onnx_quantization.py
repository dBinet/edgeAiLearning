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
from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization import quant_pre_process

# ---------------------------------------------------------------------------
# Transforms — clean only, no augmentation anywhere in this script
# ---------------------------------------------------------------------------

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

# ---------------------------------------------------------------------------
# Load ONNX model — from Week 16 Session 4's export
# ---------------------------------------------------------------------------

def load_onnx_model(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    return ort.InferenceSession(onnx_path)



# ---------------------------------------------------------------------------
# Validate onnx
# ---------------------------------------------------------------------------

def validate_onnx(onnx_session, test_loader):
    total = 0
    onnx_correct = 0

    onnx_input_name = onnx_session.get_inputs()[0].name

    with torch.no_grad():
        for images, labels in test_loader:

            onnx_output = onnx_session.run(None, {onnx_input_name: images.numpy()})[0]
            onnx_preds = np.argmax(onnx_output, axis=1)

            onnx_correct += (onnx_preds == labels.numpy()).sum().item()

            total += labels.size(0)

    accuracy =onnx_correct / total
    return accuracy

def get_model_size_kb_onnx(onnx_path):
    size_kb = os.path.getsize(onnx_path) / 1024
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


    
    quantize_dynamic("models/cnn_week16_session3.onnx", "models/cnn_week17_session3_quantize.onnx")

    onnx_session_quantize = load_onnx_model("models/cnn_week17_session3_quantize.onnx")

    # Check quantization
    for node in onnx_session_quantize.graph.node:
        print(node.op_type, node.name)

    accuracy = validate_onnx(onnx_session_quantize, test_loader)
    print(f"ONNX Runtime accuracy: {accuracy:.4f}")

    onnx_session_size = get_model_size_kb_onnx("models/cnn_week16_session3.onnx")
    onnx_session_quantize_size = get_model_size_kb_onnx("models/cnn_week17_session3_quantize.onnx")


    print(f"ONNX FP32 model size: {onnx_session_size:.1f} KB")
    print(f"ONNX INT8 model size: {onnx_session_quantize_size:.1f} KB")