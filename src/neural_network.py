"""
    Start of the nerual network, using CNN model as descirbed in 
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# UNCOMMENT BELOW IF YOU HAVE A CUDA-ENABLED NVIDIA GPU, otherwise uses CPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# USE THIS IF YOU HAVE A MAC WITH APPLE SILICON
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

num_epochs = 2
batch_size = 32


train_loader = DataLoader(TextDataset('PATH\TO\train.txt'), batch_size=batch_size, shuffle=True)  # replace with path to data

class Neural_Net_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # this code is adapted from the link above
        self.conv1 = nn.Conv2d(1,6,3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,3)
        # 144 comes from 16 * 3 * 3 (16 outputs on second conv layer and 3x3 kernel size)
        self.fc1 = nn.Linear(144,72)
        
        self.fc2 = nn.Linear(72,24)
        # currently training to be either smudgy or not smudgy so only 2 outputs on last FC layer
        self.fc3 = nn.Linear(24,2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Neural_Net_CNN().to(device) # 

# Loss fucntion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)


# Train
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):

        inputs, labels = data
        # convert inputs to a suitable tensor representation
        inputs = torch.tensor(inputs)  # Convert your text to tensors appropriately
        labels = torch.tensor(labels).to(device)

        # Forward, back and optimization
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print out what epoch number its on and loss rate
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Finished Training")
