import pandas as pd

COL_NAMES = [
    'unit_number', 'time_in_cycles', 'altitude', 'mach_number', 'throttle_resolver_angle',
    't2', 't24', 't30', 't50', 'p2', 'p15', 'p30', 'nf', 'nc', 'epr', 'ps30', 'phi',
    'nrf', 'nrc', 'bpr', 'farb', 'ht_bleed', 'nf_dmd', 'pcnfr_dmd', 'w31', 'w32'
]

DEAD_SENSORS = ['throttle_resolver_angle', 't2', 'epr', 'nf_dmd', 'pcnfr_dmd', 'p2', 'farb']

DATA_DIR = 'data/CMAPSSData'


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


if __name__ == '__main__':
    train_set, test_set, test_last_cycle = load_fd001()

    print(train_set.shape)
    print(train_set[train_set['unit_number'] == 1].tail())
    print(test_last_cycle[['unit_number', 'time_in_cycles', 'RUL']].head())