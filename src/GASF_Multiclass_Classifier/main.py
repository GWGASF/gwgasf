from prepare_data import prepare_data
from test_plots import plot_data
from model_training import train_model, CNNModel
from confusion_matrix_analysis import confusion_matrix_analysis
import torch

# Prepare data and create dataloaders
training_data, testing_data, validation_data, device, x_train, y_train, idx_train, img_x_train = prepare_data()

# Plot the data
plot_data(x_train, y_train, idx_train, img_x_train)

# Train the model
model_save_path = '/home/dfredin/gwgasf/models/gasf_model.pth'
train_model(training_data, validation_data, device, model_save_path)

# Load the trained model
model = CNNModel().to(device)
model.load_state_dict(torch.load(model_save_path))
print(f'Model loaded from {model_save_path}')

# Perform confusion matrix analysis
confusion_matrix_analysis(training_data, validation_data, testing_data, model, device)
