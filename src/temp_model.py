import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch import nn, optim

"""
This file is simply to just allow for testing of the functions using the model made in test.ipynb, will be removed when using our own custom neural network
"""

"""UNCOMMENT BELOW IF YOU HAVE A CUDA-ENABLED NVIDIA GPU, otherwise uses CPU"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""USE THIS IF YOU HAVE A MAC WITH APPLE SILICON"""
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Hyperparameters
input_size = 784 # 28x28 pixels flattened (specific to MNIST dataset)
hidden_size = 128
num_classes = 10
num_epochs = 2
batch_size = 64
learning_rate = 0.001

# MNIST dataset
train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

def predict_with_model_test(model: NeuralNet):
    # Get some random testing images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Show images and predictions
    outputs = model(images.reshape(-1, 28*28).to(device))
    _, predicted = torch.max(outputs, 1)

    # Display the first few images, actual labels, and model's predictions (helps us see predicted-actual relationship)
    for i in range(4):
        print(f"Actual: {labels[i].item()}, precited:{predicted[i].item()}")

# testing that correct file paths get here, will eventually be that our model is predicted with this function, will refactor once model is made
def predict_with_model(model,file: str):
    print(file)