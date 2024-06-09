from GASF_Multiclass_Classifier.prepare_data import prepare_data
from GASF_Multiclass_Classifier.test_plots import plot_data
from GASF_Multiclass_Classifier.model_training import train_model, CNNModel, load_checkpoint
from GASF_Multiclass_Classifier.confusion_matrix_analysis import confusion_matrix_analysis
import torch

# User-defined hyperparameters
learning_rate = 0.0005
epochs = 25
L2_reg = 0.001

# Prepare data and create dataloaders
training_data, testing_data, validation_data, device, x_train, y_train, idx_train, img_x_train = prepare_data()

# Plot the data
plot_data(x_train, y_train, idx_train, img_x_train)

# Path to save/load the model
model_save_path = '/home/dfredin/gwgasf/models/gasf_model.pth'

# Train the model (set resume_training to True if you want to resume from a checkpoint)
resume_training = False
train_model(training_data, validation_data, device, learning_rate, epochs, L2_reg, model_save_path, resume_training)

# Load the trained model for evaluation
model = CNNModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_reg)
model, optimizer, _, _, _ = load_checkpoint(model_save_path, model, optimizer)

# Perform confusion matrix analysis
confusion_matrix_analysis(training_data, validation_data, testing_data, model, device)
