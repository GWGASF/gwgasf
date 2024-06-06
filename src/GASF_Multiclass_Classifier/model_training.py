import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

class CNNModel(nn.Module):
    
    def __init__(self):
        super(CNNModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=6, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=5, stride=3, padding=1)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.fc1 = nn.Linear(in_features=128*8*8, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=3)
        self.dropcnn = nn.Dropout(0.25)
        self.dropfc = nn.Dropout(0.6)

    def forward(self, x):
        out = self.layer1(x)
        out = self.dropcnn(out)
        out = self.layer2(out)
        out = self.dropcnn(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropfc(out)
        out = self.fc2(out)
        out = self.dropfc(out)
        out = self.fc3(out)
        return out

def train_model(training_data, validation_data, device, model_save_path='/home/dfredin/gwgasf/models/gasf_model.pth'):
    GASF_Model = CNNModel().to(device)
    summary(GASF_Model, (2, 194, 194))

    learning_rate = 0.0005
    epochs = 25
    L2_reg = 0.001

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(GASF_Model.parameters(), lr=learning_rate, weight_decay=L2_reg)

    cost_value = np.empty(epochs)
    cost_valid_value = np.empty(epochs)

    for i in range(epochs):
        t_cost = 0
        for j, (x, y) in enumerate(tqdm(training_data, desc=f'Epoch {i+1}')):
            x = x.to(device)
            y = y.to(device)
            p_value = GASF_Model(x)
            p_value = torch.transpose(p_value, 0, 1)
            y = torch.transpose(y, 0, 1)
            y = torch.argmax(y, dim=1)
            cost = loss_func(p_value, y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            t_cost += cost.item()

        cost_value[i] = t_cost / (j + 1)
        print('========')
        print(f'Cost {round(cost_value[i], 2)}')
        print('========', '\n')

        with torch.no_grad():
            v_cost = 0
            for run, (a, b) in enumerate(validation_data):
                a = a.to(device)
                b = b.to(device)
                p_value = GASF_Model(a)
                p_value = torch.transpose(p_value, 0, 1)
                b = torch.transpose(b, 0, 1)
                b = torch.argmax(b, dim=1)
                cost_valid = loss_func(p_value, b)
                v_cost += cost_valid.item()
            cost_valid_value[i] = v_cost / (run + 1)
            print('===================')
            print(f'Validation Cost {round(cost_valid_value[i], 2)}')
            print('===================', '\n')

    total_epoch = np.linspace(1, epochs, epochs)
    plt.title('Loss value to epoch')
    plt.plot(total_epoch, cost_value, label='Training Set')
    plt.plot(total_epoch, cost_valid_value, label='Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.savefig('/home/dfredin/gwgasf/results/figures/train_val_Loss.png', bbox_inches='tight')
    plt.show()
    plt.close()

    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(GASF_Model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
