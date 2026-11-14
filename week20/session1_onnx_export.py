import time

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn

DATA_DIR    = 'data/CMAPSSData'
MODEL_PATH  = 'models/cnn1d_cmapss_fd001.pth'
SCALER_PATH = 'models/cnn1d_cmapss_fd001_scaler.npz'
ONNX_PATH   = 'models/week20_session1.onnx'

WINDOW_SIZE = 30

COL_NAMES = [
    'unit_number', 'time_in_cycles', 'altitude', 'mach_number', 'throttle_resolver_angle',
    't2', 't24', 't30', 't50', 'p2', 'p15', 'p30', 'nf', 'nc', 'epr', 'ps30', 'phi',
    'nrf', 'nrc', 'bpr', 'farb', 'ht_bleed', 'nf_dmd', 'pcnfr_dmd', 'w31', 'w32',
]
FEATURE_COLS = [
    'altitude', 'mach_number', 't24', 't30', 't50', 'p15', 'p30', 'nf', 'nc', 'ps30', 'phi',
    'nrf', 'nrc', 'bpr', 'ht_bleed', 'w31', 'w32',
]
N_CHANNELS = len(FEATURE_COLS)   # 17


def make_model():
    return nn.Sequential(
        nn.Conv1d(N_CHANNELS, 32, kernel_size=3, padding=1),   # (batch, 32, 30)
        nn.ReLU(),
        nn.MaxPool1d(2),                                       # (batch, 32, 15)
        nn.Conv1d(32, 64, kernel_size=3, padding=1),           # (batch, 64, 15)
        nn.ReLU(),
        nn.MaxPool1d(2),                                       # (batch, 64, 7)
        nn.Flatten(),                                          # (batch, 448)
        nn.Linear(448, 1),                                     # (batch, 1)
    )


def load_test_windows():
    """One scaled window per test engine (its last WINDOW_SIZE cycles) plus true RUL labels."""
    test_set = pd.read_csv(f'{DATA_DIR}/test_FD001.txt', header=None, names=COL_NAMES, sep=r'\s+')
    rul = pd.read_csv(f'{DATA_DIR}/RUL_FD001.txt', header=None, names=['RUL'])['RUL'].values

    scaler = np.load(SCALER_PATH)
    mean, scale = scaler['mean'], scaler['scale']

    # groupby sorts by unit_number, which matches the order of RUL_FD001.txt
    windows = [
        (engine[FEATURE_COLS].values[-WINDOW_SIZE:] - mean) / scale
        for _, engine in test_set.groupby('unit_number')
    ]

    X = torch.FloatTensor(np.array(windows)).transpose(1, 2).contiguous()   # (100, 17, 30)
    y = torch.FloatTensor(rul)
    return X, y


def rmse(preds, targets):
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def time_ms(fn, n_warmup=20, n_runs=200, repeats=5):
    """Median, min and max of the per-call latency (ms) across several repeats."""
    for _ in range(n_warmup):
        fn()
    results = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(n_runs):
            fn()
        results.append((time.perf_counter() - start) / n_runs * 1000)
    return np.median(results), min(results), max(results)


if __name__ == '__main__':

    X_test, y_test = load_test_windows()
    print("X_test shape:", tuple(X_test.shape))   # expect (100, 17, 30)

    model = make_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # --- Export ---
    torch.onnx.export(
        model,
        torch.randn(1, N_CHANNELS, WINDOW_SIZE),
        ONNX_PATH,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=13,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(ONNX_PATH))
    print("ONNX model structure check: passed")

    # --- Correctness on the 100 real test windows ---
    ort_session = ort.InferenceSession(ONNX_PATH)

    with torch.no_grad():
        pytorch_preds = model(X_test).squeeze(1).numpy()
    onnx_preds = ort_session.run(None, {'input': X_test.numpy()})[0].squeeze(1)
    y_true = y_test.numpy()

    print(f"PyTorch RMSE:      {rmse(pytorch_preds, y_true):.3f} cycles")
    print(f"ONNX Runtime RMSE: {rmse(onnx_preds, y_true):.3f} cycles")
    print(f"Max per-window prediction difference: {np.abs(pytorch_preds - onnx_preds).max():.2e}")

    if np.allclose(pytorch_preds, onnx_preds, atol=1e-4):
        print("Correctness check: PASSED -- all 100 predictions match")
    else:
        print("Correctness check: FAILED")

    # --- Single-window latency ---
    window_torch = X_test[:1]
    window_numpy = window_torch.numpy()

    def run_torch():
        with torch.no_grad():
            model(window_torch)

    def run_onnx():
        ort_session.run(None, {'input': window_numpy})

    t_med, t_min, t_max = time_ms(run_torch)
    o_med, o_min, o_max = time_ms(run_onnx)

    print("\nSingle-window latency (median [min-max] over 5 repeats of 200 runs):")
    print(f"  PyTorch:      {t_med:.3f} ms  [{t_min:.3f}-{t_max:.3f}]")
    print(f"  ONNX Runtime: {o_med:.3f} ms  [{o_min:.3f}-{o_max:.3f}]")
    print(f"  Speedup:      {t_med / o_med:.2f}x")