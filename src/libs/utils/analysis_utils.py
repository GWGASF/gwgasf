# src/libs/utils/analysis_utils.py

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from libs.utils.s3_helper import create_s3_filesystem
from tqdm import tqdm
import csv
import os
import logging
import tempfile

def save_metrics(metrics, config, dataset_name):
    """Save precision, recall, F1 score, and support to a CSV file on S3."""
    fs = create_s3_filesystem()  # Create the S3 filesystem
    save_path_s3 = os.path.join(config['paths']['results_path'], f'{dataset_name}_confusion_matrix_metrics.csv')

    # Prepare the header and data
    header = ['Class', 'Precision', 'Recall', 'F1 Score', 'Support']
    data = []
    classes = ['Glitch', 'Signal', 'Background']
    
    for i in range(len(classes)):
        data.append([classes[i], metrics[0][i], metrics[1][i], metrics[2][i], metrics[3][i]])

    # Save the metrics to a temporary CSV file locally
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        temp_file_path = tmp_file.name
        logging.info(f"Created temporary file for metrics: {temp_file_path}")
        with open(temp_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

    # Upload the CSV file to S3
    try:
        logging.info(f"Uploading metrics to S3 at {save_path_s3}")
        fs.put(temp_file_path, save_path_s3)  # Upload to S3
        logging.info(f"Successfully uploaded metrics to {save_path_s3}")
    except Exception as e:
        logging.error(f"Failed to upload metrics to S3: {e}")

    # Clean up the local temporary file
    try:
        os.remove(temp_file_path)
        logging.info(f"Deleted temporary file for metrics {temp_file_path}")
    except Exception as e:
        logging.error(f"Failed to delete temporary file {temp_file_path}: {e}")

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

    save_metrics(metrics, config, dataset_name)

    return conf_matrix