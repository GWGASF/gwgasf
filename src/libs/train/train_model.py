# src/libs/train/train_model.py

import torch
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from libs.train.model_utils import save_checkpoint, save_best_model, load_checkpoint
from libs.architecture.cnn_model import CNNModel
from libs.utils.plot_utils import plot_training_validation_loss
from libs.data.data_utils import set_seed

def train_model(config, device, training_data, validation_data):
    """Train the model using parameters from config."""
    set_seed(config['hyperparameters']['seed'])
    model = CNNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['hyperparameters']['learning_rate'], weight_decay=config['hyperparameters']['L2_reg'])
    criterion = torch.nn.CrossEntropyLoss()

    summary(model, (2, 194, 194), len(training_data.dataset))

    training_loss_values = np.empty(config['hyperparameters']['epochs'])
    validation_loss_values = np.empty(config['hyperparameters']['epochs'])
    

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['hyperparameters']['epochs']):
        model.train()
        training_loss = 0
        for j, (x, y) in enumerate(tqdm(training_data)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        avg_training_loss = training_loss / (j + 1)
        training_loss_values[epoch] = avg_training_loss

        model.eval()
        with torch.no_grad():
            validation_loss = 0
            for run, (x, y) in enumerate(validation_data):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                validation_loss += loss.item()

        avg_validation_loss = validation_loss / (run + 1)
        validation_loss_values[epoch] = avg_validation_loss

        print(f"Epoch {epoch+1}/{config['hyperparameters']['epochs']}, Training Loss: {avg_training_loss}, Validation Loss: {avg_validation_loss}")

        # # Save checkpoint
        # save_checkpoint(config['paths']['models_path'], model, optimizer, epoch + 1, avg_validation_loss)

        # Save the best model
        if avg_validation_loss < best_val_loss:
            best_val_loss = avg_validation_loss
            save_best_model(model, config)

    # Plot training and validation loss
    plot_training_validation_loss(training_loss_values, validation_loss_values, config)
