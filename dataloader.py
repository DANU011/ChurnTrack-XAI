import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CustomerChurnDataset(Dataset):
    def __init__(self, time_series_df, meta_df, window_size=6):
        self.window_size = window_size
        self.time_series_data = []
        self.meta_data = []
        self.labels = []

        # 고객별 시계열 슬라이딩 윈도우 구성
        for customer_id in time_series_df['customer_id'].unique():
            customer_ts = time_series_df[time_series_df['customer_id'] == customer_id].sort_values('month')
            customer_meta = meta_df[meta_df['customer_id'] == customer_id].iloc[0]
            label = customer_meta['churn']

            ts_values = customer_ts.drop(columns=['customer_id', 'month']).values
            for i in range(len(ts_values) - window_size + 1):
                self.time_series_data.append(ts_values[i:i+window_size])
                self.meta_data.append(customer_meta.drop(['customer_id', 'churn']).values)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.time_series_data[idx], dtype=torch.float32),
            torch.tensor(self.meta_data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

# DataLoader 생성 함수
def create_dataloaders(time_series_df, meta_df, batch_size=64, window_size=6, split_ratio=(0.7, 0.15, 0.15)):
    dataset = CustomerChurnDataset(time_series_df, meta_df, window_size)
    total_size = len(dataset)
    train_size = int(split_ratio[0] * total_size)
    val_size = int(split_ratio[1] * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    meta_info = {
        'num_features': time_series_df.drop(columns=['customer_id', 'month']).shape[1],
        'meta_dim': meta_df.drop(columns=['customer_id', 'churn']).shape[1],
    }

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
        meta_info
    )
