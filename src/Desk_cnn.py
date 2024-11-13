import torch
import torch.nn as nn
import torch.nn.functional as F

class Desk_cnn(nn.Module):
    def __init__(self, size: int, num_boxes: int, num_classes: int):
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=size,
                               out_channels=10,
                               kernel_size=(4,4))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=size,
                               out_channels=20,
                               kernel_size=(3,3))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.linear(in_features=600,
                          out_features=300)
        self.relu3 = nn.ReLu()
        
        # Localization Layer
        self.bbox_output = nn.Linear(in_features=300,out_features=8)
        # Classification Layer
        self.class_output = nn.Linear(in_features=300,out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1()
        x = self.pool1(F.relu(self.conv1(x)))

        x = self.conv2(x)
        x = self.relu2()
        x = self.pool2(F.relu(self.conv2(x)))

        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)

        return self.bbox_output(x).view(1,self.num_boxes,4), self.class_output(x)
