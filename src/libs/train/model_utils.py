# src/libs/train/model_utils.py

import torch
import os
import tempfile
import logging
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

# def save_best_model(model, config):
#     """Save the best model based on validation loss."""
#     fs = create_s3_filesystem(config)
#     path = config['paths']['models_path']
#     with fs.open(os.path.join(path, 'best_model.pth'), 'wb') as f:
#         torch.save(model.state_dict(), f)


def save_best_model(model, config):
    """Save the best model based on validation loss and upload to S3."""
    fs = create_s3_filesystem(config)
    s3_model_path = os.path.join(config['paths']['models_path'], 'best_model.pth')

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
        temp_model_path = tmp_file.name

    try:
        # Save model locally
        torch.save(model.state_dict(), temp_model_path)
        logging.info(f"Model saved locally at {temp_model_path}")

        # Get the file size explicitly
        file_size = os.path.getsize(temp_model_path)
        logging.info(f"File size: {file_size} bytes")

        # Open the file in binary mode and upload using fs.put
        with open(temp_model_path, "rb") as f:
            fs.put(temp_model_path, s3_model_path)

        logging.info(f"Successfully uploaded model to S3 at {s3_model_path}")
    
    except Exception as e:
        logging.error(f"Failed to upload model to S3: {e}")
    
    finally:
        # Clean up the local temporary file
        try:
            os.remove(temp_model_path)
            logging.info(f"Deleted temporary file: {temp_model_path}")
        except Exception as e:
            logging.error(f"Failed to delete temporary file {temp_model_path}: {e}")






def load_best_model(config, device):
    """Load the best saved model for evaluation."""
    set_seed(config['hyperparameters']['seed'])
    path = config['paths']['models_path']
    fs = create_s3_filesystem(config)
    with fs.open(os.path.join(path, 'best_model.pth'), 'rb') as f:
        model = CNNModel().to(device)
        model.load_state_dict(torch.load(f, map_location=device))
    return model, device