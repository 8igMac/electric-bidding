"""
Define Dataset and DataLoader on the training_data folder.

Preprocessing: 以天為單位，用前七天預測第八天，先試一個 target.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import params 

def get_train_vali_test():
    # Read training file.
    data_list = list()
    path = './training_data'
    for filename in os.listdir(path):
      if filename[-3:] == 'csv':
        f = os.path.join(path, filename)
        df = pd.read_csv(f, parse_dates=True, index_col=0, dtype=np.float32)
        df['gen-con'] = df['generation'] - df['consumption']
        data_list.append(df)

    # Prepare data.
    look_ahead_day = params.LOOK_AHEAD_DAY
    hour_per_day = 24
    combine = pd.DataFrame()
    for data in data_list:
      shifted_dict = {n: data['gen-con'].shift(n) for n in range(hour_per_day * (look_ahead_day + 1))}
      shifted = pd.DataFrame(shifted_dict)
      shifted.dropna(inplace=True)
      combine = pd.concat([combine, shifted])

    # Split train, vali, test.
    total_size = len(combine)
    '''
    train_size = int(total_size * params.TRAIN_RATIO)
    vali_size = int(total_size * params.VALI_RATIO)
    test_size = total_size - train_size - vali_size
    '''
    # debug
    train_size = 5000
    vali_size = 1000
    test_size = 1000

    train = combine.iloc[0:train_size]
    vali = combine.iloc[train_size:train_size + vali_size]
    test = combine.iloc[train_size + vali_size:]

    train = {'labels': train.iloc[:,:24], 'features': train.iloc[:, 24:]}
    vali = {'labels': vali.iloc[:,:24], 'features': vali.iloc[:, 24:]}
    test = {'labels': test.iloc[:,:24], 'features': test.iloc[:, 24:]}

    return train, vali, test

class MyDataset(Dataset):
  def __init__(self, x, y, scale_x=True):
    if scale_x:
      x = StandardScaler().fit_transform(x)
      self.x = torch.from_numpy(x.reshape((-1, 7, 24)))
      self.y = torch.from_numpy(y.to_numpy())
      self.y = self.y.reshape(-1, 1, 24)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    '''Return feature, label.
    '''
    return self.x[idx], self.y[idx]
