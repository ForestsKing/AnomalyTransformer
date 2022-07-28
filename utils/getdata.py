import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data(data_path='./MSL/'):
    train_valid_data = np.load(data_path + "MSL_train.npy")
    train_valid_label = np.zeros(len(train_valid_data))

    thre_test_data = np.load(data_path + "MSL_test.npy")
    thre_test_label = np.load(data_path + "MSL_test_label.npy").astype(int)

    scaler = StandardScaler()
    train_valid_data = scaler.fit_transform(train_valid_data)
    thre_test_data = scaler.transform(thre_test_data)

    data = {}

    data['train_data'] = train_valid_data[:int(0.7 * len(train_valid_data)), :]
    data['train_label'] = train_valid_label[:int(0.7 * len(train_valid_label))]
    data['valid_data'] = train_valid_data[int(0.7 * len(train_valid_data)):, :]
    data['valid_label'] = train_valid_label[int(0.7 * len(train_valid_label)):]
    data['thre_data'] = thre_test_data[:int(0.3 * len(thre_test_data)), :]
    data['thre_label'] = thre_test_label[:int(0.3 * len(thre_test_label))]
    data['test_data'] = thre_test_data[int(0.3 * len(thre_test_data)):, :]
    data['test_label'] = thre_test_label[int(0.3 * len(thre_test_label)):]
    return data
