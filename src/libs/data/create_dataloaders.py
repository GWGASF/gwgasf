# src/libs/data/create_dataloaders.py

import torch
from torch.utils.data import DataLoader, Dataset
from libs.data.data_utils import set_seed

class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

def create_dataloaders(strains, targets, config):
    """Create DataLoader objects for training, testing, and validation sets."""
    batch_size = config['hyperparameters']['batch_size']
    seed = config['hyperparameters']['seed']


    set_seed(seed)
    training_data = DataLoader(
        MyDataset(strains['training'], targets['training']), 
        batch_size=batch_size, 
        shuffle=True
    )

    validation_data = DataLoader(
        MyDataset(strains['validation'], targets['validation']), 
        batch_size=batch_size, 
        shuffle=False
    )

    testing_data = DataLoader(
        MyDataset(strains['testing'], targets['testing']), 
        batch_size=batch_size, 
        shuffle=False
    )

    return training_data, validation_data, testing_data
