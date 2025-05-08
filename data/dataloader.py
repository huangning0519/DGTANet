import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.utils import StandardScaler

def get_dataloader(data_dir, batch_size, time_steps, horizon, input_dim, output_dim):
    data_path = os.path.join(data_dir, 'data.npy')
    data = np.load(data_path)

    num_samples, num_nodes, num_features = data.shape
    assert num_features == input_dim, "Input dimension mismatch!"

    x_list, y_list = [], []
    for i in range(num_samples - time_steps - horizon):
        x = data[i:i+time_steps]
        y = data[i+time_steps:i+time_steps+horizon, :, -1:]
        x_list.append(x)
        y_list.append(y)

    x = np.stack(x_list, axis=0)
    y = np.stack(y_list, axis=0)

    num_train = int(x.shape[0] * 0.7)
    num_val = int(x.shape[0] * 0.1)
    num_test = x.shape[0] - num_train - num_val

    train_x = x[:num_train]
    train_y = y[:num_train]
    val_x = x[num_train:num_train+num_val]
    val_y = y[num_train:num_train+num_val]
    test_x = x[-num_test:]
    test_y = y[-num_test:]

    scaler = StandardScaler(train_x)

    train_x = scaler.transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    train_loader = DataLoader(TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y)), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
