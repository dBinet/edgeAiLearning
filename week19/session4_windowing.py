import pandas as pd
import numpy as np
import torch
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
WINDOW_SIZE = 30
STRIDE = 1  # placeholder for the planned stride sweep


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
            window = X_scaled[start_idx + i: start_idx + i + window_size]
            label = rul_values[i + window_size - 1]
            windows.append(window)
            labels.append(label)

    return windows, labels


def build_test_windows(df, X_scaled, rul_values, window_size=WINDOW_SIZE):
    """Grab each test engine's final window only, labeled with its true RUL at the cutoff point."""
    windows, labels = [], []

    for i, unit in enumerate(df['unit_number'].unique()):
        engine_data = df[df['unit_number'] == unit]
        size = len(engine_data)
        start_idx = engine_data.index[0]

        window = X_scaled[start_idx + size - window_size: start_idx + size]
        label = rul_values[i]
        windows.append(window)
        labels.append(label)

    return windows, labels


def to_model_tensors(windows, labels):
    """Stack windows/labels into tensors, reshaped to (batch, channels, sequence_length) for Conv1d."""
    windows_tensor = torch.FloatTensor(np.array(windows))
    windows_tensor = torch.transpose(windows_tensor, 2, 1)
    labels_tensor = torch.FloatTensor(labels)
    return windows_tensor, labels_tensor


if __name__ == '__main__':
    train_set, test_set, test_last_cycle = load_fd001()

    train_set = train_set.drop(columns=DEAD_SENSORS)
    test_set = test_set.drop(columns=DEAD_SENSORS)
    test_last_cycle = test_last_cycle.drop(columns=DEAD_SENSORS)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_set[FEATURE_COLS])
    X_test_scaled = scaler.transform(test_set[FEATURE_COLS])

    all_windows, all_labels = build_windows(train_set, X_train_scaled)
    test_windows, test_labels = build_test_windows(test_set, X_test_scaled, test_last_cycle['RUL'].values)

    X_train, y_train = to_model_tensors(all_windows, all_labels)
    X_test, y_test = to_model_tensors(test_windows, test_labels)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)