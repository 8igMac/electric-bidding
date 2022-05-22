"""
Training and save the model {model_name}.mdf5
"""
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import params
from model import MyLSTM
from data import get_train_vali_test, MyDataset

if __name__ == '__main__':

    # Prepare data.
    print('Prepare data.')
    train, vali, test = get_train_vali_test()
    dataset_dict = {
        'train': MyDataset(train['features'], train['labels']),
        'vali': MyDataset(vali['features'], vali['labels']),
        'test': MyDataset(test['features'], test['labels']),
    }

    dataloader_dict = {
        x: DataLoader(
            dataset=dataset_dict[x],
            batch_size=params.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        ) for x in ['train', 'vali', 'test']
    }

    # Setup model.
    print('Setup model.')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MyLSTM(params.INPUT_SIZE, params.OUTPUT_SIZE, params.HIDDEN_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=params.LEARNING_RATE)

    # Training.
    print('Training.')
    hist_dict = {
        'train': list(),
        'vali': list(),
    }
    best_epoch_loss = 1000000
    for epoch in tqdm(range(params.EPOCHS)):
        for phase in ['train', 'vali']:
            running_loss = 0.0
            for inputs, labels in dataloader_dict[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                model = model.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(dataloader_dict[phase].dataset)
            if phase == 'vali' and epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            hist_dict[phase].append(epoch_loss)
    print(f'\nBest validation epoch loss: {best_epoch_loss}')

    # Plot training result and save the plot.
    print('Plot training result.')
    plt.plot(np.arange(len(hist_dict['train'])), hist_dict['train'], label='train')
    plt.plot(np.arange(len(hist_dict['vali'])), hist_dict['vali'], label='vali')
    plt.legend()
    plt.savefig('training_result.png')

    # Evaluating.
    print('Evaluating.')
    with torch.no_grad():
        total_loss = 0.0
        for inputs, labels in tqdm(dataloader_dict['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
        total_loss /= len(dataloader_dict['test'].dataset)
    print(f'Testing epoch loss: {total_loss}')

    # Saving model.
    print('Saving model.')
    torch.save(best_model_wts, './model.hdf5')
