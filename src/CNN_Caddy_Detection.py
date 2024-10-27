"""
    Start of the nerual network, using CNN model as descirbed in 
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
from pathlib import Path
from torch.utils.data import DataLoader

# UNCOMMENT BELOW IF YOU HAVE A CUDA-ENABLED NVIDIA GPU, otherwise uses CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# USE THIS IF YOU HAVE A MAC WITH APPLE SILICON
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

num_epochs = 2
batch_size = 32
trans = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(size=(1056,816))
    ])

# # Change datapath to the path to where the data is. The file structure that it comes in is the right file structure.
# train_datapath = Path("submissions/train") 
# test_datapath = Path("submissions/test")
# # Might need to seperate the data into test and training folders, the test and train folder should emulate the clients submissions folder
# train_packet_dataset = data_loader.IndividualIMGDataset(targ_dir=train_datapath,transform=trans,type="caddy") 
# test_packet_dataset = data_loader.IndividualIMGDataset(targ_dir=test_datapath, transform=trans,type="caddy")
# # Datasets turn into dataloaders
# train_loader = DataLoader(train_packet_dataset, batch_size=batch_size, shuffle=True)  
# test_loader = DataLoader(test_packet_dataset, batch_size=batch_size, shuffle=True)

class Caddy_Detection_CNN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, num_boxes: int, num_classes: int):
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        # this code is adapted from the link above
        self.conv1 = nn.Conv2d(
            in_channels = input_shape,
            out_channels = hidden_units,
            kernel_size = 4,
            stride = 1,
            padding = 0)
        self.pool = nn.MaxPool2d(kernel_size = 2,
                                 stride = 2)
        self.conv2 = nn.Conv2d(in_channels = hidden_units,
                               out_channels = hidden_units,
                               kernel_size=2,
                               stride = 1,
                               padding = 0)
        final_cnn_size = hidden_units * 262*201
        
        # Localization Layer
        self.bbox_output = nn.Linear(in_features=final_cnn_size,out_features=self.num_boxes*4)
        # Classifier Layer
        self.class_output = nn.Linear(in_features=final_cnn_size,out_features=self.num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = torch.flatten(x,0)
        # print(x.shape)
        return self.bbox_output(x).view(1,self.num_boxes,4), self.class_output(x)

# model = Caddy_Detection_CNN(3,10,1,1).to(device) # Initialize model and set to the target device (GPU or CPU)

Loss fucntion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)


# # Train
# for epoch in range(num_epochs):
#     for i, data in enumerate(train_loader, 0):
#         img, targets = data
#         model.train()
#         # convert inputs to a suitable tensor representation
#         if isinstance(img, torch.Tensor):
#             img = torch.tensor(img)  # Convert your text to tensors appropriately
#         if isinstance(targets["boxes"], torch.Tensor):
#             targets["boxes"] = torch.tensor(targets["boxes"])

#         # Forward, back and optimization
#         optimizer.zero_grad()
#         outputs = model(img.to(device))
#         loss = criterion(outputs, targets)
        
#         loss.backward()
#         optimizer.step()

#         # Print out what epoch number its on and loss rate
#         if (i + 1) % 100 == 0:
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
#     test_loss = 0
#     model.eval()
#     with torch.inference_mode():
#         for img, targets in test_loader:
#             if isinstance(img,torch.tensor):
#                 img = torch.tensor(img)  # Convert your text to tensors appropriately
#             if isinstance(targets["boxes"],torch.tensor):
#                 targets["boxes"] = torch.tensor(targets["boxes"]) 
#             test_outputs = model(img)
#             test_loss = criterion(test_outputs, targets)
    
#         print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}")

    
        
        

print("Finished Training")
