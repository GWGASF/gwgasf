# src/libs/utils/plot_utils.py

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os
import numpy as np
import logging
import tempfile
from libs.utils.s3_helper import create_s3_filesystem

def plot_training_validation_loss(training_loss, validation_loss, epochs, path):
    """Plot training and validation loss vs epochs and save the plot to S3."""
    fs = create_s3_filesystem()  # Create the S3 filesystem
    save_path_s3 = os.path.join(path, "train_val_Loss.png")

    # Plot the training and validation loss
    total_epoch = np.linspace(1, epochs, epochs)
    plt.title('Loss value to epoch')
    plt.plot(total_epoch, training_loss, label='Training Set')
    plt.plot(total_epoch, validation_loss, label='Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()

    # Save the plot to a temporary local file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        temp_file_path = tmp_file.name
        logging.info(f"Created temporary file for loss plot: {temp_file_path}")
        plt.savefig(temp_file_path, bbox_inches='tight')
        plt.close()

    # Upload the plot to S3
    try:
        logging.info(f"Uploading loss plot to S3 at {save_path_s3}")
        fs.put(temp_file_path, save_path_s3)  # Upload to S3
        logging.info(f"Successfully uploaded loss plot to {save_path_s3}")
    except Exception as e:
        logging.error(f"Failed to upload loss plot to S3: {e}")

    # Clean up the local temporary file
    try:
        os.remove(temp_file_path)
        logging.info(f"Deleted temporary file for loss plot: {temp_file_path}")
    except Exception as e:
        logging.error(f"Failed to delete temporary file {temp_file_path}: {e}")

def plot_confusion_matrix(conf_matrix, title, config):
    """Plot confusion matrix using sklearn's ConfusionMatrixDisplay and save the plot to S3."""
    fs = create_s3_filesystem()  # Create the S3 filesystem
    save_path_s3 = config['paths']['results_path'] + f'{title}_confusion_matrix.png'

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Glitch', 'Signal', 'Background'])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Wistia', values_format='.3f')
    ax.set_title(f'{title} Confusion Matrix')

    # Set the text color to black
    for text in disp.text_.ravel():
        text.set_color('black')

    # Custom labels
    lab = np.array([['True Positive', 'False Negative', 'False Negative'], 
                    ['False Positive', 'True Negative', 'True Negative'],
                    ['False Positive', 'True Negative', 'True Negative']])
    
    for i in range(3):
        for j in range(3):
            # Adjust the position of the custom label
            ax.text(j, i - 0.1, lab[i, j], ha='center', va='center', color='black')

    # Save the plot to a temporary local file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        temp_file_path = tmp_file.name
        logging.info(f"Created temporary file for plot: {temp_file_path}")
        plt.savefig(temp_file_path, bbox_inches='tight')
        plt.close()

    # Upload the plot to S3
    try:
        logging.info(f"Uploading plot to S3 at {save_path_s3}")
        fs.put(temp_file_path, save_path_s3)  # Upload to S3
        logging.info(f"Successfully uploaded plot to {save_path_s3}")
    except Exception as e:
        logging.error(f"Failed to upload plot to S3: {e}")

    # Clean up the local temporary file
    try:
        os.remove(temp_file_path)
        logging.info(f"Deleted temporary file for plot: {temp_file_path}")
    except Exception as e:
        logging.error(f"Failed to delete temporary file {temp_file_path}: {e}")

# def plot_gasf(data, title):
#     """Plot GASF data."""
#     fig, ax = plt.subplots()
#     cax = ax.imshow(data[0,0,:,:], cmap='rainbow', origin='lower')
#     plt.title(title)
#     fig.colorbar(cax)
#     return fig

# def plot_time_series(data, title):
#     """Plot time-series data."""
#     fig, ax = plt.subplots()
#     ax.plot(data[0])
#     plt.title(title)
#     return fig