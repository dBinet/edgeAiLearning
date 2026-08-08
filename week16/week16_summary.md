# Week 16 Complete — Full Summary for Week 17

---

## Context & Goal
**12-month goal:** Build employable Edge AI / Embedded AI skills to improve job security and open new career doors beyond current 5G software development role.

**Current position: End of month 4, entering Months 5–7 (Edge Deployment phase)**

---

## Environment
- **SSH:** `ssh dbin@PiDavid.local` (Bonjour Print Services installed on Windows for reliable `.local` resolution; Fing app as backup for IP discovery if DHCP lease changes)
- **Alias:** `aiwork` → activates venv + navigates to project folder
- **Repo:** `~/edgeAiLearning/`
- **Venv:** `source ~/edge-ai/bin/activate`
- **GitHub:** https://github.com/dBinet/edgeAiLearning
- **PyTorch:** 2.11.0, torchvision 0.22.0
- **New this week:** `onnx`, `onnxruntime`, `onnxscript` installed on Pi

---

## What Was Accomplished in Week 16

### Session 1 — Batch Normalization
- Added `nn.BatchNorm2d` after each Conv2d, before ReLU — only change vs Week 15 Session 5 baseline
- BatchNorm normalizes per-channel activations across the batch (mean 0, var 1), then applies learnable scale (`gamma`) and shift (`beta`)
- Applied to every conv layer, not just the first — instability compounds layer to layer as each layer's weights update simultaneously
- Not applied to the final output layer — raw logits need to flow into CrossEntropyLoss unnormalized
- `model.train()`/`model.eval()` now matters for two reasons, not one: Dropout *and* BatchNorm's running-stats switch
- Result: ceiling barely moved (74.44% → 74.75%), but train/test gap cut roughly in half (21.9pp → 11.2pp)
- Side effect: batch-statistics noise acts as implicit regularization, despite BatchNorm's stated purpose being training stability, not overfitting reduction
- File: `week16/session1_batchnorm.py`

### Session 2 — Data Augmentation
- Added `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` to train transform only; test transform stayed clean
- Kept BatchNorm from Session 1 — established improvement, not re-tested
- Result: ceiling still flat (74.36%); train/test gap flipped negative (train measured against harder, augmented images, no longer directly comparable to test)
- Confirmed overfitting fully eliminated by this point — and the ceiling still hadn't moved
- File: `week16/session2_augmentation.py`

### Session 3 — Deeper Architecture
- Added third conv block (64→128 channels), widened classifier head to `Linear(2048, 512)`
- Spatial dimension trace (explicit, to avoid Week 15's flatten-size bug):
```
32×32 → 16×16 (32ch) → 8×8 (64ch) → 4×4 (128ch)
Flatten: 128 × 4 × 4 = 2048
```
- Result: **80.12% best test accuracy (epoch 13)** — +5.4pp from ~77K extra parameters (7% size increase)
- Confirms the two-session elimination process (Sessions 1–2) correctly diagnosed the problem: capacity, not regularization
- Best epoch ≠ last epoch again (13 vs 15) — checkpoint saving caught it
- File: `week16/session3_deeper.py`

### Session 4 — ONNX Export
- Exported Session 3's best checkpoint to `models/cnn_week16_session3.onnx`
- Used dynamo-based exporter (`onnxscript` installed) after `dynamo=False` legacy path was offered as an alternative
- Correctness verified on a single random input: PyTorch and ONNX Runtime outputs matched (`atol=1e-5`)
- Single-image latency benchmark:

| Engine | Latency |
|---|---|
| PyTorch | 4.821 ms |
| ONNX Runtime | 0.767 ms |
| **Speedup** | **6.28×** |

- Direct resolution of Week 15 Session 4's finding: numpy conv layers were 200× slower than PyTorch, making hand-written numpy inference unviable for CNNs. ONNX Runtime uses an optimized C++ execution engine instead of reimplementing convolution — no numpy detour needed.
- File: `week16/session4_onnx_export.py`

### Session 5 — Full Test-Set ONNX Validation
- Session 4 only verified correctness on one random tensor — this session validated across the full 10,000-image CIFAR-10 test set
- Results:

| Metric | Result |
|---|---|
| PyTorch accuracy | 80.12% |
| ONNX Runtime accuracy | 80.12% |
| Prediction agreement | **100.00%** |
| Accuracy difference | 0.000000 |

- 100% prediction agreement is the strongest possible signal — both engines agree on every single image, correct or incorrect, ruling out compensating-errors scenarios
- Batch throughput (batch=64): ONNX 2838 img/s vs PyTorch 1360 img/s → **2.09× speedup**
- Speedup dropped from Session 4's 6.28× (single-image) to 2.09× (batched) — PyTorch's fixed per-call overhead amortizes across a batch, narrowing ONNX Runtime's relative advantage. Matches the Week 14 finding that fixed overhead, not proportional cost, dominates small-scale PyTorch inference.
- File: `week16/session5_onnx_validation.py`

---

## Self-Assessment (end of Week 16)

Compared to Week 15's ranking, here's what moved:

1. **Batch training loop** — still strongest, unchanged (8/10)
2. **Architectural ceiling diagnosis** — moved up significantly. Week 15 could read results but not design fixes (5/10). Now able to design and execute a full elimination process (BatchNorm → augmentation → depth) and correctly predict which lever would work (7-8/10)
3. **Overfitting in CNN context** — reinforced further; now comfortable distinguishing "shrinking gap" from "solved ceiling" (8/10)
4. **BatchNorm mechanics** — new this week: placement, per-layer application, train/eval interaction, regularization side effect (7/10)
5. **Deployment pipeline (ONNX)** — new this week: export, structural + numerical + full-dataset validation, latency vs throughput distinction (7/10)
6. **CNN architecture / spatial dimension tracing** — improved from Week 15's weakest area (5/10 → 7/10), no arithmetic errors this week after explicit tracing discipline
7. **Convolution operation fundamentals** — unchanged, not directly exercised this week (5/10)

**Weakest areas heading into Week 17:** no direct exposure yet to quantization (INT8, post-training vs quantization-aware training) or model pruning — both Months 5–7 material, both untouched so far.

---

## PyTorch Reference (new this week)

**CNN block with BatchNorm:**
```python
nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
nn.BatchNorm2d(out_ch),     # placed AFTER conv, BEFORE ReLU
nn.ReLU(),
nn.MaxPool2d(2)
```

**Train-only augmentation (test transform stays clean):**
```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)   # no augmentation
])
```

**ONNX export:**
```python
model.eval()
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(
    model, dummy_input, 'model.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=13
    # add dynamo=False to use the legacy TorchScript exporter and avoid
    # the onnxscript dependency, if preferred
)
```

**ONNX Runtime inference + correctness check:**
```python
import onnxruntime as ort
ort_session = ort.InferenceSession('model.onnx')
onnx_output = ort_session.run(None, {'input': test_input.numpy()})[0]

np.allclose(pytorch_output, onnx_output, atol=1e-5)   # correctness check
```

**Full-dataset validation pattern (accuracy + per-image agreement):**
```python
pytorch_correct = onnx_correct = agree_count = total = 0
for images, labels in test_loader:
    pytorch_preds = torch.argmax(model(images), dim=1).numpy()
    onnx_preds = np.argmax(ort_session.run(None, {'input': images.numpy()})[0], axis=1)
    pytorch_correct += (pytorch_preds == labels.numpy()).sum()
    onnx_correct    += (onnx_preds == labels.numpy()).sum()
    agree_count      += (pytorch_preds == onnx_preds).sum()   # stronger check than accuracy alone
    total            += labels.size(0)
```

---

## Deployment Finding (confirmed Week 16)

**ONNX Runtime is the correct deployment path for CNNs — numpy is not.**

| Layer type | Best deployment path | Evidence |
|---|---|---|
| Dense (MLP) | numpy | 34× faster than PyTorch (Week 14) |
| Conv (CNN) | ONNX Runtime | 6.28× faster single-image, 2.09× faster batched (Week 16); numpy was 200× *slower* (Week 15) |

**Updated production edge pattern:**
```
Dense layers:  Train PyTorch → Extract weights → Deploy numpy       ✓
Conv layers:   Train PyTorch → Export ONNX → Deploy ONNX Runtime    ✓
Conv layers:   Train PyTorch → Deploy numpy                         ✗ (200× slower)
```

**Latency vs throughput — a distinction worth remembering for deployment decisions:**
- Single-image, latency-sensitive use case (e.g. real-time camera classification) → ONNX Runtime's advantage is large (6.28×)
- Batch-processing use case (e.g. classifying many images at once) → advantage is real but more modest (2.09×), since PyTorch's fixed per-call overhead amortizes across the batch

---

## 12-Month Roadmap

| Phase | Months | Focus |
|---|---|---|
| Foundations | 1–4 | ML fundamentals on Pi — **complete** |
| Edge Deployment | 5–7 | TinyML, model optimization, quantization, ONNX, AI HAT — **starting Week 17** |
| Real Project | 8–10 | Portfolio-worthy edge AI project on Pi |
| Job Search | 11–12 | GitHub tells a coherent Edge AI story, interviews |

---

## Week 17 Plan (draft)

**Theme: Entering the Edge Deployment phase — model compression and further hardware benchmarking**

### Session 1 — Post-training quantization (INT8)
- Why quantization matters for edge: 4× smaller model size (FP32 → INT8), often faster inference on CPU
- PyTorch's `torch.quantization` static/dynamic quantization APIs
- Quantize the Week 16 Session 3 CNN, compare accuracy before/after

### Session 2 — Quantization accuracy/speed tradeoff
- Full test-set validation of the quantized model (same rigor as Week 16 Session 5)
- Benchmark: FP32 PyTorch vs FP32 ONNX vs INT8 quantized — size, latency, accuracy side by side

### Session 3 — ONNX quantization path
- Quantize directly in ONNX (`onnxruntime.quantization`) as an alternative to PyTorch-side quantization
- Compare results against Session 1–2's PyTorch-quantization path

### Session 4 — Model pruning (exploratory)
- Structured vs unstructured pruning concepts
- Try pruning the Week 16 CNN, observe accuracy/size tradeoff

### Session 5 — Consolidation
- Pull together FP32 / ONNX / quantized / pruned results into one comparison table
- Decide which variant is the right baseline going into the Month 8–10 portfolio project

### Session 6 — Week 17 review + Week 18 plan

*(This is a draft — adjust scope once Session 1's actual results are in, same as Week 16 diverged from its own original plan.)*

---

## Full Concepts Mastered So Far

*(carried forward from Week 15, plus new additions below)*

- Linear regression (single + multi-feature)
- Gradient descent (loop and vectorized)
- NumPy vectorization and why it matters
- Train/test split
- MSE and R² — definition and interpretation
- Normalization / StandardScaler — fit on train only
- Multicollinearity
- Synthetic vs real data performance gap
- Edge deployment and benchmarking on Pi
- sklearn vs from-scratch validation
- Normal Equation vs gradient descent tradeoffs
- Feature engineering (execution + systematic process)
- Classification vs regression (conceptual)
- Logistic regression (usage + from scratch)
- Sigmoid function — internals and why it works
- Binary cross-entropy loss — why it pairs with sigmoid (convex surface)
- Forward pass concept
- Accuracy score
- Confusion matrix — all four cells, interpretation
- Precision, Recall, F1 — definition and when each matters
- Class imbalance problem
- Unique variance — low target correlation ≠ safe to drop
- Overfitting in practice — train/test gap, not just bad test accuracy
- L2 regularization — loss penalty + gradient update, lambda tuning
- Threshold tuning — 0.5 is not optimal, sweep to find best threshold
- ROC curve — TPR vs FPR across all thresholds
- AUC — threshold-independent model quality metric
- ReLU activation — element-wise max(0, z), used in hidden layers
- Neural network architecture — Input → Hidden (ReLU) → Output (Sigmoid/Softmax)
- Weight initialization — He init for ReLU (`* sqrt(2/n_in)`), not `* 0.01`
- Backpropagation — chain rule through two and three layers, verified by hand
- predict_proba vs predict — separate raw probabilities from thresholded output
- Non-linear decision boundaries — what hidden layers actually do
- Architecture sizing — right-size for the problem, more capacity ≠ better
- Momentum optimizer — velocity accumulation, beta hyperparameter
- Adam optimizer — adaptive lr, first + second moments, bias correction
- Optimizer tradeoffs — training-time cost vs inference-time cost
- Pi benchmarking — µs/sample, training vs inference cost, depth scaling
- PyTorch training loop — nn.Sequential, CrossEntropyLoss, Adam, forward/backward/step
- Mini-batch training — DataLoader, TensorDataset, batch size tradeoffs
- Autograd — computation graph, .backward(), verified against manual gradients
- Framework overhead — constant per-call cost, 11–34× inference penalty on Pi at small scale
- Production edge pattern — train in PyTorch, deploy in numpy/ONNX/TFLite
- Multi-class classification — CrossEntropyLoss, argmax, torch.long labels
- Real data pipeline — pd.read_csv, label remapping, StandardScaler
- Overfitting signature — train/test loss divergence visible epoch by epoch
- Dropout — model.train()/model.eval(), gap reduction cost
- Model save/load — state_dict, verified identical outputs
- Weight export — numpy_forward, single-sample speedup, batch amortization
- Convolution operation — kernel, stride, padding, output size formula
- CNN architecture — Conv2d, MaxPool2d, Flatten, spatial dimension tracing
- Weight sharing — same kernel at every position, spatial invariance
- Batch training loop — inner loop over DataLoader, running_loss, evaluate()
- Filter count vs spatial size — independent knobs
- Overfitting in CNN context — train/test loss curves, best epoch ≠ last epoch
- Best checkpoint saving — save on test metric improvement, not at end
- Dropout in CNN context — after linear layers, same train/eval pattern
- Epoch noise — SGD shuffle + Dropout stochasticity, why curves aren't smooth
- Conv weight extraction — 4D tensors (out_ch, in_ch, kH, kW), no .T needed
- numpy export boundary — works for dense (34×), fails for conv (200× slower)
- Architectural ceiling — distinct from overfitting, fixed by depth/BatchNorm/augmentation
- CIFAR-10 vs MNIST — real-world image complexity, human accuracy ~94%
- torchvision — datasets, transforms, ToTensor, Normalize
- **Batch Normalization — per-channel batch statistics, gamma/beta, placement rule, train/eval running-stats switch**
- **BatchNorm as implicit regularizer — side effect distinct from its stated purpose**
- **Data augmentation — RandomCrop/RandomHorizontalFlip, train-only application**
- **Elimination-based debugging — ruling out regularization causes before blaming architecture**
- **ONNX export — torch.onnx.export, dynamic_axes, opset_version, dynamo vs legacy exporter**
- **ONNX correctness validation — single-sample np.allclose, full-dataset prediction agreement**
- **ONNX Runtime — InferenceSession, run(), CPU execution without PyTorch overhead**
- **Latency vs throughput — single-image vs batched inference, fixed-overhead amortization**