import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
import csv
import os

def save_metrics(conf_matrix, metrics, config, dataset_name):
    """Save precision, recall, F1 score, and support to a CSV file."""
    save_path = config['paths']['results_path']
    csv_file = os.path.join(save_path, f'{dataset_name}_confusion_matrix_metrics.csv')

    # Prepare the header and data
    header = ['Class', 'Precision', 'Recall', 'F1 Score', 'Support']
    data = []
    classes = ['Glitch', 'Signal', 'Background']
    
    for i in range(len(classes)):
        data.append([classes[i], metrics[0][i], metrics[1][i], metrics[2][i], metrics[3][i]])

    # Save to CSV
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def calculate_confusion_matrix(loaded_best_model, dataloader, config, dataset_name, num_classes=3):
    """Calculate confusion matrix and metrics using sklearn."""
    model, device = loaded_best_model
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(torch.argmax(y, 1).cpu().numpy())
    
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes), normalize='true')
    metrics = precision_recall_fscore_support(all_labels, all_preds, labels=range(num_classes))

    save_metrics(conf_matrix, metrics, config, dataset_name)

    return conf_matrix
