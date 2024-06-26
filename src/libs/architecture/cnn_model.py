# libs/architecture/cnn_model.py

import torch.nn as nn

class CNNModel(nn.Module):
    
    def __init__(self):
        super(CNNModel, self).__init__()

        # First convolution layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=6, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=5, stride=3, padding=1)
        )
        
        # Second convolution layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Fully connected layers
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




# import torch.nn as nn

# class CNNModel(nn.Module):
    
#     def __init__(self):
#         super(CNNModel, self).__init__()

#         # First convolution layer
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         # Second convolution layer
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         # Third convolution layer
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
        
#         # Fourth convolution layer
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#         # Global average pooling
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(in_features=512, out_features=1024)
#         self.fc2 = nn.Linear(in_features=1024, out_features=512)
#         self.fc3 = nn.Linear(in_features=512, out_features=3)
#         self.dropcnn = nn.Dropout(0.3)
#         self.dropfc = nn.Dropout(0.5)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.dropcnn(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.dropcnn(out)
#         out = self.global_avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.dropfc(out)
#         out = self.fc2(out)
#         out = self.dropfc(out)
#         out = self.fc3(out)
        
#         return out
