# src/libs/train/model_utils.py

import torch
import os
from libs.utils.s3_helper import create_s3_filesystem
from libs.architecture.cnn_model import CNNModel
from libs.data.data_utils import set_seed

def save_checkpoint(path, model, optimizer, epoch, loss, config):
    """Save model checkpoint."""
    fs = create_s3_filesystem(config)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    with fs.open(os.path.join(path, f'checkpoint_epoch_{epoch}.pth'), 'wb') as f:
        torch.save(checkpoint, f)    

def load_checkpoint(filepath, model, optimizer, config):
    """Load model checkpoint for continued training."""
    fs = create_s3_filesystem(config)
    with fs.open(filepath, 'rb') as f:
        checkpoint = torch.load(f)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def save_best_model(model, path, config):
    """Save the best model based on validation loss."""
    fs = create_s3_filesystem(config)
    with fs.open(os.path.join(path, 'best_model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

def load_best_model(config, device):
    """Load the best saved model for evaluation."""
    set_seed(config['hyperparameters']['seed'])
    path = config['paths']['models_path']
    fs = create_s3_filesystem(config)
    with fs.open(os.path.join(path, 'best_model.pth'), 'rb') as f:
        model = CNNModel().to(device)
        model.load_state_dict(torch.load(f, map_location=device))
    return model, device