import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# torch tensor device check
def get_dtype(device):
    if device.type == 'cpu':
        return torch.FloatTensor
    elif device.type == 'cuda':
        return torch.cuda.FloatTensor

def calculate_metrics(conf_matrix):
    # Calculate precision, recall, and F1 score for the entire model
    TP = np.sum(np.diag(conf_matrix))  # True Positives
    FP = np.sum(np.sum(conf_matrix, axis=0) - np.diag(conf_matrix))  # False Positives
    FN = np.sum(np.sum(conf_matrix, axis=1) - np.diag(conf_matrix))  # False Negatives

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

def plot_confusion_matrix(data, title, model, device):
    num_classes = 3
    conf_matrix = torch.zeros([num_classes, num_classes]).to(device)
    num_count = torch.zeros([num_classes]).to(device)

    dtype = get_dtype(device)

    with torch.no_grad():
        for num, (x, y) in enumerate(tqdm(data)):
            Real_label = y.to(device)

            # Convert softmax output to onehot
            max_class = torch.Tensor.argmax(model(x.to(device)), axis=1)
            pred = F.one_hot(max_class)

            # Accumulating statistical value
            conf_matrix += torch.matmul(pred.T.type(dtype), Real_label.type(dtype))
            num_count += Real_label.sum(axis=0)

        num_count = num_count.detach().cpu().numpy().astype('float64')
        conf_matrix = conf_matrix.detach().cpu().numpy().astype('float64')

    # Normalize the confusion matrix
    conf_matrix /= num_count

    precision, recall, f1_score = calculate_metrics(conf_matrix)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1_score}')

    # Labels for a 3x3 confusion matrix
    lab = np.array([['True Positive', 'False Positive', 'False Positive'], 
                    ['False Negative', 'True Negative', 'False Positive'],
                    ['False Negative', 'False Negative', 'True Negative']])

    plt.figure()
    color = plt.pcolormesh([conf_matrix[2], conf_matrix[1], conf_matrix[0]], cmap='Wistia', vmin=0, vmax=1)

    for i in range(3):
        for j in range(3):
            plt.text(i + .5, j + .5, 
                     f'{lab[2-j, i]}\n{round(conf_matrix[2-j, i], 3)}', 
                     ha='center', 
                     va='center')

    plt.xticks([.5, 1.5, 2.5], ['Glitch', 'Signal', 'Background'])
    plt.yticks([.5, 1.5, 2.5], ['Background', 'Signal', 'Glitch'], rotation=45)
    plt.xlabel('Predicted Value')
    plt.ylabel('Actual Value')
    plt.title(f'3x3 {title} Confusion Matrix')
    plt.colorbar(color)
    plt.savefig(f'/home/dfredin/gwgasf/results/figures/{title}_confuMatrix.png', bbox_inches='tight')
    plt.show()
    plt.close()

def confusion_matrix_analysis(training_data, validation_data, testing_data, model, device):
    plot_confusion_matrix(validation_data, 'Validation', model, device)
    plot_confusion_matrix(training_data, 'Training', model, device)
    plot_confusion_matrix(testing_data, 'Test', model, device)









# ------------
# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # torch tensor device check
# def get_dtype(device):
#     if device.type == 'cpu':
#         return torch.FloatTensor
#     elif device.type == 'cuda':
#         return torch.cuda.FloatTensor

# def calculate_metrics(conf_matrix):
#     # Calculate precision, recall, and F1 score for the entire model
#     TP = np.sum(np.diag(conf_matrix))  # True Positives
#     FP = np.sum(np.sum(conf_matrix, axis=0) - np.diag(conf_matrix))  # False Positives
#     FN = np.sum(np.sum(conf_matrix, axis=1) - np.diag(conf_matrix))  # False Negatives

#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     f1_score = 2 * (precision * recall) / (precision + recall)

#     return precision, recall, f1_score

# def plot_confusion_matrix(data, title, model, device):
#     num_classes = 3
#     conf_matrix = torch.zeros([num_classes, num_classes]).to(device)
#     num_count = torch.zeros([num_classes]).to(device)

#     dtype = get_dtype(device)

#     with torch.no_grad():
#         for num, (x, y) in enumerate(tqdm(data)):
#             Real_label = y.to(device)

#             # Convert softmax output to onehot
#             max_class = torch.Tensor.argmax(model(x.to(device)), axis=1)
#             pred = F.one_hot(max_class, num_classes=num_classes).type(dtype)

#             # Accumulating statistical value
#             conf_matrix += torch.matmul(pred.T, Real_label.type(dtype))
#             num_count += Real_label.sum(axis=0)

#         num_count = num_count.detach().cpu().numpy().astype('float64')
#         conf_matrix = conf_matrix.detach().cpu().numpy().astype('float64')

#     # Normalize the confusion matrix
#     conf_matrix /= num_count

#     precision, recall, f1_score = calculate_metrics(conf_matrix)

#     print(f'Precision: {precision}')
#     print(f'Recall: {recall}')
#     print(f'F1 Score: {f1_score}')

#     # Labels for a 3x3 confusion matrix
#     lab = np.array([['True Positive', 'False Positive', 'False Positive'], 
#                     ['False Negative', 'True Negative', 'False Positive'],
#                     ['False Negative', 'False Negative', 'True Negative']])

#     plt.figure()
#     color = plt.pcolormesh([conf_matrix[2], conf_matrix[1], conf_matrix[0]], cmap='Wistia', vmin=0, vmax=1)

#     for i in range(3):
#         for j in range(3):
#             plt.text(i + .5, j + .5, 
#                      f'{lab[2-j, i]}\n{round(conf_matrix[2-j, i], 3)}', 
#                      ha='center', 
#                      va='center')

#     plt.xticks([.5, 1.5, 2.5], ['Glitch', 'Signal', 'Background'])
#     plt.yticks([.5, 1.5, 2.5], ['Background', 'Signal', 'Glitch'], rotation=45)
#     plt.xlabel('Predicted Value')
#     plt.ylabel('Actual Value')
#     plt.title(f'3x3 {title} Confusion Matrix')
#     plt.colorbar(color)
#     plt.savefig(f'/home/dfredin/gwgasf/results/figures/{title}_confuMatrix.png', bbox_inches='tight')
#     plt.show()
#     plt.close()

# def confusion_matrix_analysis(training_data, validation_data, testing_data, model, device):
#     plot_confusion_matrix(validation_data, 'Validation', model, device)
#     plot_confusion_matrix(training_data, 'Training', model, device)
#     plot_confusion_matrix(testing_data, 'Test', model, device)
