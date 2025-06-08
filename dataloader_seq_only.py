import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class CustomerChurnSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, time_series_df, window_size=5):
        self.window_size = window_size
        self.x_seq = []
        self.labels = []
        self.customer_ids = []

        # 숫자형 특성만 필터링
        num_cols = time_series_df.select_dtypes(include=[np.number]).columns
        ts_numeric_cols = [c for c in num_cols if c not in ['month', 'customer_id']]

        # 고객별 시계열 구성
        for cid, group in time_series_df.groupby('customer_id'):
            group = group.sort_values('month')
            values = group[ts_numeric_cols].values
            if len(values) == window_size:
                self.x_seq.append(values)
                self.labels.append(group['churn'].iloc[-1])  # 마지막 달의 라벨 사용
                self.customer_ids.append(cid)

        self.x_seq = np.array(self.x_seq, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.customer_ids = np.array(self.customer_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.customer_ids[idx],
            torch.tensor(self.x_seq[idx]),
            torch.tensor(self.labels[idx])
        )

def create_stratified_dataloaders(
    time_series_df,
    window_size=5,
    batch_size=64,
    cache_path=None,
    test_size=0.2,
    val_size=0.2,
    random_state=42
):
    dataset = CustomerChurnSequenceDataset(time_series_df, window_size=window_size)

    # Stratified split
    y = dataset.labels
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=random_state
    )
    train_y = y[train_idx]
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size, stratify=train_y, random_state=random_state
    )

    def subset(indices):
        return torch.utils.data.Subset(dataset, indices)

    train_loader = DataLoader(subset(train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(subset(val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(subset(test_idx), batch_size=batch_size, shuffle=False)

    meta_info = {
        'num_features': dataset.x_seq.shape[2],
        'meta_dim': 0  # 메타 없음
    }

    return train_loader, val_loader, test_loader, meta_info
