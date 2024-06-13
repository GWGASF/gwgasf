# src/libs/utils/analysis_utils.py

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_metrics(conf_matrix):
    """Calculate precision, recall, and F1 score."""
    TP = np.sum(np.diag(conf_matrix))  # True Positives
    FP = np.sum(np.sum(conf_matrix, axis=0) - np.diag(conf_matrix))  # False Positives
    FN = np.sum(np.sum(conf_matrix, axis=1) - np.diag(conf_matrix))  # False Negatives

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

def calculate_confusion_matrix(model, dataloader, device):
    """Calculate confusion matrix."""
    num_classes = 3
    conf_matrix = torch.zeros([num_classes, num_classes]).to(device)
    num_count = torch.zeros([num_classes]).to(device)

    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            max_class = torch.argmax(outputs, axis=1)
            preds = F.one_hot(max_class, num_classes=outputs.shape[1])
            conf_matrix += torch.matmul(preds.T.type(torch.float32), y.type(torch.float32))
            num_count += y.sum(axis=0)

    num_count = num_count.cpu().numpy().astype('float64')
    conf_matrix = conf_matrix.cpu().numpy().astype('float64')

    # Normalize the confusion matrix.
    conf_matrix /= num_count

    return conf_matrix
