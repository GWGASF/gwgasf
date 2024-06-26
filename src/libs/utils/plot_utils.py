import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os
import numpy as np

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

def plot_confusion_matrix(conf_matrix, title, config):
    """Plot confusion matrix using sklearn's ConfusionMatrixDisplay and save the plot."""
    save_path = config['paths']['results_path']
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

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{title}_confusion_matrix.png'), bbox_inches='tight')
    plt.close()

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
