import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

COL_NAMES = [
    'unit_number', 'time_in_cycles', 'altitude', 'mach_number', 'throttle_resolver_angle',
    't2', 't24', 't30', 't50', 'p2', 'p15', 'p30', 'nf', 'nc', 'epr', 'ps30', 'phi',
    'nrf', 'nrc', 'bpr', 'farb', 'ht_bleed', 'nf_dmd', 'pcnfr_dmd', 'w31', 'w32'
]

FEATURE_COLS = [
    'altitude', 'mach_number', 't24', 't30', 't50', 'p15', 'p30', 'nf', 'nc', 'ps30', 'phi',
    'nrf', 'nrc', 'bpr', 'ht_bleed', 'w31', 'w32'
]

DEAD_SENSORS = ['throttle_resolver_angle', 't2', 'epr', 'nf_dmd', 'pcnfr_dmd', 'p2', 'farb']

DATA_DIR = 'data/CMAPSSData'
MODEL_PATH = 'models/cnn1d_cmapss_fd001.pth'
SCALER_PATH = 'models/cnn1d_cmapss_fd001_scaler.npz'

# Tuned config (Week 19 Session 5 experiments)
WINDOW_SIZE = 30
RUL_CAP = 125
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 30
SEED = 0


def load_fd001():
    train_set = pd.read_csv(f'{DATA_DIR}/train_FD001.txt', header=None, names=COL_NAMES, sep=r'\s+')
    test_set = pd.read_csv(f'{DATA_DIR}/test_FD001.txt', header=None, names=COL_NAMES, sep=r'\s+')
    rul_set = pd.read_csv(f'{DATA_DIR}/RUL_FD001.txt', header=None, names=['RUL'])

    train_set['RUL'] = (
        train_set.groupby('unit_number')['time_in_cycles'].transform('max') - train_set['time_in_cycles']
    )

    test_last_cycle = test_set.groupby('unit_number').tail(1).reset_index(drop=True)
    test_last_cycle['RUL'] = rul_set['RUL'].values

    return train_set, test_set, test_last_cycle


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


def build_test_windows(df, X_scaled, rul_values, window_size=WINDOW_SIZE):
    """Grab each test engine's final window only, labeled with its true RUL at the cutoff point."""
    windows, labels = [], []

    for i, unit in enumerate(df['unit_number'].unique()):
        engine_data = df[df['unit_number'] == unit]
        size = len(engine_data)
        start_idx = engine_data.index[0]

        windows.append(X_scaled[start_idx + size - window_size: start_idx + size])
        labels.append(rul_values[i])

    return windows, labels


def to_model_tensors(windows, labels):
    """Stack windows/labels into tensors, reshaped to (batch, channels, sequence_length) for Conv1d."""
    windows_tensor = torch.FloatTensor(np.array(windows))
    windows_tensor = torch.transpose(windows_tensor, 2, 1)
    labels_tensor = torch.FloatTensor(labels)
    return windows_tensor, labels_tensor


def make_model():
    return nn.Sequential(
        nn.Conv1d(17, 32, kernel_size=3, padding=1),   # (batch, 32, 30)
        nn.ReLU(),
        nn.MaxPool1d(2),                               # (batch, 32, 15)
        nn.Conv1d(32, 64, kernel_size=3, padding=1),   # (batch, 64, 15)
        nn.ReLU(),
        nn.MaxPool1d(2),                               # (batch, 64, 7)
        nn.Flatten(),                                  # (batch, 448)
        nn.Linear(448, 1),                             # (batch, 1) -> single RUL prediction
    )


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze(1)
        return torch.sqrt(nn.functional.mse_loss(preds, y)).item()


def train(model, loader, X_test, y_test):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            loss = criterion(model(X_batch).squeeze(1), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_rmse = (epoch_loss / len(loader)) ** 0.5
        test_rmse = evaluate(model, X_test, y_test)
        history.append(test_rmse)
        print(f"Epoch {epoch+1:2d} | train RMSE: {train_rmse:6.2f} | "
              f"test RMSE: {test_rmse:6.2f} | {time.time() - start:5.1f}s")

    return history


if __name__ == '__main__':
    torch.manual_seed(SEED)

    train_set, test_set, test_last_cycle = load_fd001()

    train_set = train_set.drop(columns=DEAD_SENSORS)
    train_set['RUL'] = train_set['RUL'].clip(upper=RUL_CAP)   # train labels only, test stays true
    test_set = test_set.drop(columns=DEAD_SENSORS)
    test_last_cycle = test_last_cycle.drop(columns=DEAD_SENSORS)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_set[FEATURE_COLS])
    X_test_scaled = scaler.transform(test_set[FEATURE_COLS])

    all_windows, all_labels = build_windows(train_set, X_train_scaled)
    test_windows, test_labels = build_test_windows(test_set, X_test_scaled, test_last_cycle['RUL'].values)

    X_train, y_train = to_model_tensors(all_windows, all_labels)
    X_test, y_test = to_model_tensors(test_windows, test_labels)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model = make_model()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    history = train(model, loader, X_test, y_test)

    final_rmse = evaluate(model, X_test, y_test)
    print(f"\nFinal test RMSE (epoch {EPOCHS}): {final_rmse:.2f} cycles")
    print(f"Mean of last 5 epochs: {sum(history[-5:]) / 5:.2f} cycles")

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    np.savez(SCALER_PATH, mean=scaler.mean_, scale=scaler.scale_)
    print(f"Saved {MODEL_PATH} and {SCALER_PATH}")