import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset


class TsDataset(Dataset):
    """Timeseries dataset for training."""
    def __init__(self, cfg):
        """
        Load total data into memory and normalize it.
        """
        self.seq_len = cfg.SEQ_LEN
        df = pd.read_csv(cfg.DATA_PATH, index_col=0)
        df = df[cfg.COLS]
        df = df.dropna()
        df = normalize_data(df, cfg.COLS)
        self.data = np.array(df)
    
    def __len__(self):
        """Usable length: total length - sequence length - 1"""
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, index):
        """Get item based on sliding windows"""
        X = self.data[index: index + self.seq_len]   # e.g. points [0, 240)
        y = self.data[index + self.seq_len][-1]    # e.g. point 240
        X = torch.tensor(X).type(torch.FloatTensor)
        y = torch.tensor([y]).type(torch.FloatTensor)
        return X, y

def normalize_data(df, cols):
    """Normalize pandas dataframe"""
    for col in cols:
        min_max_scaler = preprocessing.MinMaxScaler()
        df[col] = min_max_scaler.fit_transform(df[col].values.reshape(-1,1))
    return df

