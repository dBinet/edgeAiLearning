"""
Week 17 Session 1 — Post-Training Static Quantization (INT8)

Goal: fuse Conv+BN+ReLU, calibrate, convert Week 16 Session 3 CNN to INT8.
Deliverable: one accuracy comparison (FP32 vs INT8), one size comparison.
Explicitly OUT of scope: full-dataset prediction agreement, latency, ONNX.
"""

import torch
import torch.nn as nn
import torch.quantization
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os

# ---------------------------------------------------------------------------
# 1. Model definition — must match Week 16 Session 3 exactly (same architecture,
#    since we're loading its state_dict). Add QuantStub/DeQuantStub wrapping.
# ---------------------------------------------------------------------------

class QuantizedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()

        # Identical, unnamed Sequential — must match Session 3 exactly
        # so state_dict keys ('0.weight', '1.weight', ...) load without remapping.
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
            nn.Linear(512, 10)
)

        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


# ---------------------------------------------------------------------------
# 2. Load FP32 checkpoint
# ---------------------------------------------------------------------------

def load_fp32_model(checkpoint_path):
    model = QuantizedCNN()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.model.load_state_dict(checkpoint)   # load into the inner Sequential, not the wrapper
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 3. Fuse Conv + BN + ReLU
#    torch.quantization.fuse_modules(model, [[...layer names...], [...], ...])
#    One list per fusable triplet, per conv block. Layer names must match
#    the names you used in QuantizedCNN.__init__ (e.g. ['conv1', 'bn1', 'relu1']).
# ---------------------------------------------------------------------------

def fuse_model(model):
    torch.quantization.fuse_modules(model.model, [
        ['0', '1', '2'],    # conv1, bn1, relu1
        ['4', '5', '6'],    # conv2, bn2, relu2
        ['8', '9', '10'],   # conv3, bn3, relu3
    ], inplace=True)
    return model


# ---------------------------------------------------------------------------
# 4. Prepare for static quantization (attach observers)
# ---------------------------------------------------------------------------

def prepare_model(model):
    torch.backends.quantized.engine = 'qnnpack'   # tell PyTorch which kernels to use at inference
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)
    return model


# ---------------------------------------------------------------------------
# 5. Calibration — run ~500 images through in observer mode.
#    No augmentation transform, no gradients, no backward pass.
#    This is NOT training — just forward passes so observers see activation ranges.
# ---------------------------------------------------------------------------

def calibrate(model, calibration_loader):
    with torch.no_grad():
        for images, labels in calibration_loader:
            model(images)
        pass


# ---------------------------------------------------------------------------
# 6. Convert to INT8
# ---------------------------------------------------------------------------

def convert_model(model):
    torch.quantization.convert(model, inplace=True)
    return model


# ---------------------------------------------------------------------------
# 7. Evaluate accuracy — reuse Week 16's evaluate() pattern
# ---------------------------------------------------------------------------

def evaluate(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds   = torch.argmax(model(images), dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total



# ---------------------------------------------------------------------------
# 8. Size comparison — file size on disk, not parameter count
# ---------------------------------------------------------------------------

def get_model_size_kb(model, path):
    torch.save(model.state_dict(), path)
    size_kb = os.path.getsize(path) / 1024
    os.remove(path)  # cleanup temp file
    return size_kb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    calibration_dataset = datasets.CIFAR10(
        root='data/', train=True, download=True, transform=test_transform
    )
    calibration_subset = Subset(calibration_dataset, range(500))
    calibration_loader = DataLoader(calibration_subset, batch_size=64, shuffle=False)

    test_dataset = datasets.CIFAR10(
        root='data/', train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # --- FP32 baseline ---
    fp32_model = load_fp32_model("models/best_model_week16_session3.pth")
    fp32_acc = evaluate(fp32_model, test_loader)
    fp32_size = get_model_size_kb(fp32_model, "temp_fp32.pth")

    # --- INT8 pipeline ---
    int8_model = load_fp32_model("models/best_model_week16_session3.pth")
    int8_model = fuse_model(int8_model)
    int8_model = prepare_model(int8_model)
    calibrate(int8_model, calibration_loader)
    int8_model = convert_model(int8_model)

    int8_acc = evaluate(int8_model, test_loader)
    int8_size = get_model_size_kb(int8_model, "temp_int8.pth")

    # --- Report ---
    print(f"FP32 accuracy: {fp32_acc*100:.2f}%  |  size: {fp32_size:.1f} KB")
    print(f"INT8 accuracy: {int8_acc*100:.2f}%  |  size: {int8_size:.1f} KB")
    print(f"Size reduction: {fp32_size / int8_size:.2f}x")