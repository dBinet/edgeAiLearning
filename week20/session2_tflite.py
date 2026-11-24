import os

import numpy as np
import onnx2tf
import pandas as pd
import torch
import torch.nn as nn

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

DATA_DIR    = 'data/CMAPSSData'
MODEL_PATH  = 'models/cnn1d_cmapss_fd001.pth'
SCALER_PATH = 'models/cnn1d_cmapss_fd001_scaler.npz'   # mean/scale saved in Week 19
ONNX_PATH   = 'models/week20_session1.onnx'            # exported in Session 1
OUT_DIR     = 'models/tflite_week20'
CALIB_PATH  = 'models/calib_cmapss.npy'

WINDOW_SIZE = 30   # chosen in Week 19; hard limit is 31 (shortest test engine)

N_CALIB = 500

COL_NAMES = [
    'unit_number', 'time_in_cycles', 'altitude', 'mach_number', 'throttle_resolver_angle',
    't2', 't24', 't30', 't50', 'p2', 'p15', 'p30', 'nf', 'nc', 'epr', 'ps30', 'phi',
    'nrf', 'nrc', 'bpr', 'farb', 'ht_bleed', 'nf_dmd', 'pcnfr_dmd', 'w31', 'w32',
]
# The 17 inputs the model was trained on (the 7 constant sensors are left out)
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
    """One standardized window per test engine (last WINDOW_SIZE cycles) plus true RUL."""
    test_set = pd.read_csv(f'{DATA_DIR}/test_FD001.txt', header=None, names=COL_NAMES, sep=r'\s+')
    rul = pd.read_csv(f'{DATA_DIR}/RUL_FD001.txt', header=None, names=['RUL'])['RUL'].values

    saved = np.load(SCALER_PATH)
    mean, scale = saved['mean'], saved['scale']

    # groupby sorts by unit_number, which matches the order of RUL_FD001.txt
    windows = [
        (engine[FEATURE_COLS].values[-WINDOW_SIZE:] - mean) / scale
        for _, engine in test_set.groupby('unit_number')
    ]
    X = torch.FloatTensor(np.array(windows)).transpose(1, 2).contiguous()   # (100, 17, 30)
    y = torch.FloatTensor(rul)
    return X, y


def load_train_set():
    """Training table with a per-row RUL column (labels are unused for calibration)."""
    train_set = pd.read_csv(f'{DATA_DIR}/train_FD001.txt', header=None, names=COL_NAMES, sep=r'\s+')
    train_set['RUL'] = (
        train_set.groupby('unit_number')['time_in_cycles'].transform('max') - train_set['time_in_cycles']
    )
    return train_set


def build_windows(df, X_scaled, window_size=WINDOW_SIZE):
    """Slide a fixed-length window per engine, labeling each with RUL at the window's last cycle."""
    windows, labels = [], []

    for unit in df['unit_number'].unique():
        engine_data = df[df['unit_number'] == unit]
        size = len(engine_data)
        start_idx = engine_data.index[0]
        rul_values = engine_data['RUL'].values

        for i in range(size - window_size + 1):
            windows.append(X_scaled[start_idx + i: start_idx + i + window_size])
            labels.append(rul_values[i + window_size - 1])

    return windows, labels


def rmse(preds, targets):
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


if __name__ == '__main__':

    # --- Test windows ---
    X_test, y_test = load_test_windows()
    print("X_test shape:", tuple(X_test.shape))   # expect (100, 17, 30)

    # --- Training windows (for calibration) ---
    train_set = load_train_set()
    saved = np.load(SCALER_PATH)
    X_train_scaled = (train_set[FEATURE_COLS].values - saved['mean']) / saved['scale']
    print("X_train_scaled shape:", X_train_scaled.shape)   # expect (20631, 17)

    all_windows, all_labels = build_windows(train_set, X_train_scaled)

    # Calibration windows.
   
    picks = np.linspace(0, len(all_windows) - 1, N_CALIB).astype(int)
    
    # Stack it into one (17731, 30, 17) array first,
    # because a list can't be indexed with an array of positions.
    calib_windows_nlc = np.array(all_windows, dtype=np.float32)[picks]   # (500, 30, 17)
    np.save(CALIB_PATH, calib_windows_nlc.astype(np.float32))
    print(f"Calibration data: {calib_windows_nlc.shape}, range [{calib_windows_nlc.min():.3f}, {calib_windows_nlc.max():.3f}]")

    onnx2tf.convert(
        input_onnx_file_path=ONNX_PATH,
        output_folder_path=OUT_DIR,
        output_signaturedefs=True,
        output_integer_quantized_tflite=True,
        custom_input_op_name_np_data_path=[
            ['input', CALIB_PATH,
             np.zeros((1, 1, N_CHANNELS), dtype=np.float32),   # mean = 0
             np.ones((1, 1, N_CHANNELS), dtype=np.float32)]    # std  = 1
        ],
        non_verbose=True,
    )

    # --- Rename FP32 output for session 2 ---
    os.replace(f'{OUT_DIR}/week20_session1_float32.tflite',
               f'{OUT_DIR}/week20_session2_float32.tflite')
    tflite_path = f'{OUT_DIR}/week20_session2_float32.tflite'
    tflite_int8_path = f'{OUT_DIR}/week20_session1_integer_quant.tflite'

    # --- Load TFLite FP32 model ---
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_index = input_details['index']
    output_index = output_details['index']

    print(f"\nTFLite input : {input_details['shape']} ({input_details['dtype'].__name__})")
    print(f"TFLite output: {output_details['shape']} ({output_details['dtype'].__name__})")

    # --- PyTorch reference ---
    model = make_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        pytorch_preds = model(X_test).squeeze(1).numpy()
    y_true = y_test.numpy()

    # --- TFLite FP32 predictions, one window at a time (batch is fixed at 1) ---
    X_np = X_test.numpy().transpose(0, 2, 1)   # (100, 17, 30) -> (100, 30, 17)
    preds = []
    for i in range(len(X_np)):
        interpreter.set_tensor(input_index, np.ascontiguousarray(X_np[i:i + 1]))
        interpreter.invoke()
        preds.append(interpreter.get_tensor(output_index)[0, 0])
    preds = np.array(preds)

    # --- Load TFLite int8 model ---
    interpreter_int8 = Interpreter(model_path=tflite_int8_path)
    interpreter_int8.allocate_tensors()
    input_details_int8 = interpreter_int8.get_input_details()[0]
    output_details_int8 = interpreter_int8.get_output_details()[0]
    input_index_int8 = input_details_int8['index']
    output_index_int8 = output_details_int8['index']

    print(f"\nTFLite int8 input : {input_details_int8['shape']} ({input_details_int8['dtype'].__name__})")
    print(f"TFLite int8 output: {output_details_int8['shape']} ({output_details_int8['dtype'].__name__})")

    # --- TFLite int8 predictions, one window at a time (batch is fixed at 1) ---
    int8_preds = []
    for i in range(len(X_np)):
        interpreter_int8.set_tensor(input_index_int8, np.ascontiguousarray(X_np[i:i + 1]))
        interpreter_int8.invoke()
        int8_preds.append(interpreter_int8.get_tensor(output_index_int8)[0, 0])
    int8_preds = np.array(int8_preds)

    # --- Results ---
    print(f"\nONNX size:   {os.path.getsize(ONNX_PATH) / 1024:.1f} KB")
    print(f"TFLite size: {os.path.getsize(tflite_path) / 1024:.1f} KB")
    print(f"PyTorch RMSE:        {rmse(pytorch_preds, y_true):.3f} cycles")
    print(f"TFLite Runtime RMSE: {rmse(preds, y_true):.3f} cycles")
    print(f"Max per-window prediction difference: {np.abs(pytorch_preds - preds).max():.2e}")

    print(f"TFLite int8 size: {os.path.getsize(tflite_int8_path) / 1024:.1f} KB")
    print(f"TFLite int8 Runtime RMSE: {rmse(int8_preds, y_true):.3f} cycles")
    print(f"Max per-window prediction difference int_8: {np.abs(pytorch_preds - int8_preds).max():.2e}")