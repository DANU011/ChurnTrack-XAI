import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split

class CustomerChurnDataset(Dataset):
    def __init__(self, time_series_df, meta_df, window_size=5):
        self.window_size = window_size
        self.time_series_data = []
        self.meta_data = []
        self.labels = []

        ts_numeric_cols = time_series_df.select_dtypes(include=[np.number]).columns.drop('month')
        meta_numeric_cols = meta_df.select_dtypes(include=[np.number]).columns.drop('churn')

        ts_base = time_series_df[['customer_id', 'month'] + list(ts_numeric_cols)].copy()
        meta_base = meta_df[['customer_id', 'churn'] + list(meta_numeric_cols)].copy()

        for cust_id in ts_base['customer_id'].unique():
            cust_ts = ts_base[ts_base['customer_id'] == cust_id].sort_values('month')
            cust_meta = meta_base[meta_base['customer_id'] == cust_id].iloc[0]
            label = cust_meta['churn']

            ts_vals = cust_ts[ts_numeric_cols].values.astype(np.float32)
            meta_vec = cust_meta[meta_numeric_cols].values.astype(np.float32)

            for i in range(len(ts_vals) - window_size + 1):
                window = ts_vals[i : i + window_size]
                self.time_series_data.append(window)
                self.meta_data.append(meta_vec)
                self.labels.append(np.float32(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_seq  = torch.from_numpy(self.time_series_data[idx])
        x_meta = torch.from_numpy(self.meta_data[idx])
        y      = torch.tensor(self.labels[idx])
        return x_seq, x_meta, y

def create_dataloaders(time_series_df, meta_df,
                       batch_size=64, window_size=5,
                       split_ratio=(0.7, 0.15, 0.15),
                       cache_path=None):
    # 캐시가 존재하면 불러오기
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Loading cached dataset from {cache_path}")
        data = np.load(cache_path)
        x_seq = torch.from_numpy(data['x_seq'])
        x_meta = torch.from_numpy(data['x_meta'])
        y = torch.from_numpy(data['y'])
        dataset = torch.utils.data.TensorDataset(x_seq, x_meta, y)
    else:
        dataset = CustomerChurnDataset(time_series_df, meta_df, window_size)
        if cache_path:
            preprocess_and_save_dataset(time_series_df, meta_df, window_size, cache_path)

    total_size = len(dataset)
    train_size = int(split_ratio[0] * total_size)
    val_size = int(split_ratio[1] * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    meta_info = {
        'num_features': dataset[0][0].shape[1],
        'meta_dim': dataset[0][1].shape[0],
    }

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set,   batch_size=batch_size, shuffle=False),
        DataLoader(test_set,  batch_size=batch_size, shuffle=False),
        meta_info
    )


def preprocess_and_save_dataset(time_series_df, meta_df, window_size=5, save_path='cached_dataset.npz'):
    dataset = CustomerChurnDataset(time_series_df, meta_df, window_size)
    np.savez_compressed(
        save_path,
        x_seq=np.array(dataset.time_series_data),
        x_meta=np.array(dataset.meta_data),
        y=np.array(dataset.labels)
    )
    print(f"Saved cached dataset to {save_path}")

# def create_cached_dataloaders(cache_path, batch_size=64, split_ratio=(0.7, 0.15, 0.15)):
#     data = np.load(cache_path)
#     x_seq = torch.from_numpy(data['x_seq'])
#     x_meta = torch.from_numpy(data['x_meta'])
#     y = torch.from_numpy(data['y'])
#
#     dataset = torch.utils.data.TensorDataset(x_seq, x_meta, y)
#
#     total_size = len(dataset)
#     train_size = int(split_ratio[0] * total_size)
#     val_size = int(split_ratio[1] * total_size)
#     test_size = total_size - train_size - val_size
#
#     generator = torch.Generator().manual_seed(42)
#     train_set, val_set, test_set = torch.utils.data.random_split(
#         dataset, [train_size, val_size, test_size], generator=generator
#     )
#
#     meta_info = {
#         'num_features': x_seq.shape[2],
#         'meta_dim': x_meta.shape[1],
#     }
#
#     return (
#         DataLoader(train_set, batch_size=batch_size, shuffle=True),
#         DataLoader(val_set, batch_size=batch_size, shuffle=False),
#         DataLoader(test_set, batch_size=batch_size, shuffle=False),
#         meta_info
#     )


def create_stratified_dataloaders(
    time_series_df=None,
    meta_df=None,
    window_size=5,
    cache_path=None,
    batch_size=64,
    split_ratio=(0.7, 0.15, 0.15)
):
    assert cache_path is not None, "cache_path must be provided."

    # 캐시 파일 없으면 생성
    if not os.path.exists(cache_path):
        if time_series_df is None or meta_df is None:
            raise ValueError("No cached file found and raw data not provided.")
        print(f"[INFO] Saving dataset to cache at {cache_path}")
        preprocess_and_save_dataset(time_series_df, meta_df, window_size, cache_path)

    print(f"[INFO] Loading cached dataset from {cache_path}")
    data = np.load(cache_path)
    x_seq = data['x_seq']
    x_meta = data['x_meta']
    y = data['y']

    # stratified split (train vs temp)
    x_seq_train, x_seq_temp, x_meta_train, x_meta_temp, y_train, y_temp = train_test_split(
        x_seq, x_meta, y, test_size=(1 - split_ratio[0]), stratify=y, random_state=42
    )

    # stratified split (val vs test)
    val_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
    x_seq_val, x_seq_test, x_meta_val, x_meta_test, y_val, y_test = train_test_split(
        x_seq_temp, x_meta_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=36
    )

    # 텐서 변환
    def to_tensor_dataset(x_seq, x_meta, y):
        return torch.utils.data.TensorDataset(
            torch.from_numpy(x_seq),
            torch.from_numpy(x_meta),
            torch.from_numpy(y)
        )

    train_set = to_tensor_dataset(x_seq_train, x_meta_train, y_train)
    val_set   = to_tensor_dataset(x_seq_val, x_meta_val, y_val)
    test_set  = to_tensor_dataset(x_seq_test, x_meta_test, y_test)

    meta_info = {
        'num_features': x_seq.shape[2],
        'meta_dim': x_meta.shape[1],
    }

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
        meta_info
    )

