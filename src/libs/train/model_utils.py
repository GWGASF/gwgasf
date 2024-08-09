# src/libs/train/model_utils.py

import torch
import os
from libs.architecture.cnn_model import CNNModel
from libs.data.data_utils import set_seed

def save_checkpoint(path, model, optimizer, epoch, loss):
    """Save model checkpoint."""
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, os.path.join(path, f'checkpoint_epoch_{epoch}.pth'))

def load_checkpoint(filepath, model, optimizer):
    """Load model checkpoint for continued training."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def save_best_model(model, path):
    """Save the best model based on validation loss."""
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'best_model.pth'))

def load_best_model(config, device):
    """Load the best saved model for evaluation."""
    set_seed(config['hyperparameters']['seed'])
    path = config['paths']['models_path']
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(os.path.join(path, 'best_model.pth')))
    return model, device