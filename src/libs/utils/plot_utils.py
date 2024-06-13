# src/libs/utils/plot_utils.py

import matplotlib.pyplot as plt
import os
import numpy as np

def plot_confusion_matrix(conf_matrix, title, save_path):
    """Plot confusion matrix with custom labels and save the plot."""
    # Labels for a 3x3 confusion matrix
    lab = np.array([['True Positive', 'False Positive', 'False Positive'], 
                    ['False Negative', 'True Negative', 'False Positive'],
                    ['False Negative', 'False Negative', 'True Negative']])

    plt.figure()
    color = plt.pcolormesh([conf_matrix[2], conf_matrix[1], conf_matrix[0]], cmap='Wistia', vmin=0, vmax=1)

    # Adjust the loop for a 3x3 matrix
    for i in range(3):
        for j in range(3):
            plt.text(i + .5, j + .5, 
                     f'{lab[2 - j, i]}\n{round(conf_matrix[2 - j, i], 3)}', 
                     ha='center', 
                     va='center')

    # Update the ticks for three classes
    plt.xticks([.5, 1.5, 2.5], ['Glitch', 'Signal', 'Background'])
    plt.yticks([.5, 1.5, 2.5], ['Background', 'Signal', 'Glitch'], rotation=45)
    plt.xlabel('Predicted Value')
    plt.ylabel('Actual Value')
    plt.title(f'3x3 {title} Confusion Matrix')
    plt.colorbar(color)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{title}_confuMatrix.png', bbox_inches='tight')
    plt.show()
    plt.close()

def save_confusion_matrix(fig, filename):
    """Save plotted confusion matrix."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)

def plot_gasf(data, title):
    """Plot GASF data."""
    fig, ax = plt.subplots()
    cax = ax.imshow(data[0,0,:,:], cmap='rainbow', origin='lower')
    plt.title(title)
    fig.colorbar(cax)
    return fig

def plot_time_series(data, title):
    """Plot time-series data."""
    fig, ax = plt.subplots()
    ax.plot(data[0])
    plt.title(title)
    return fig

def save_plot(fig, filename):
    """Save plots to a file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)

def plot_training_validation_loss(training_loss, validation_loss, epochs, save_path):
    """Plot training and validation loss vs epochs and save the plot."""
    total_epoch = np.linspace(1, epochs, epochs)
    plt.title('Loss value to epoch')
    plt.plot(total_epoch, training_loss, label='Training Set')
    plt.plot(total_epoch, validation_loss, label='Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/train_val_Loss.png', bbox_inches='tight')
    plt.show()
    plt.close()
