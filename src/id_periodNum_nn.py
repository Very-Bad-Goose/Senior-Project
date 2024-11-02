"""
    Austin Nolte     
"""



import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import get_individual_data_loader, IndividualIMGDataset, collate_fn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import random_split


# UNCOMMENT BELOW IF YOU HAVE A CUDA-ENABLED NVIDIA GPU, otherwise uses CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# USE THIS IF YOU HAVE A MAC WITH APPLE SILICON
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Transform to apply to data coming in
transform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])


# train_loader = get_individual_data_loader("src/mbrimberry_files/Submissions",transform=transform,batch_size=32,shuffle=True,num_workers=4, type="assignments")



# creating a split dataset of training and testing. First need a whole dataset of all images
packet_dataset = IndividualIMGDataset(targ_dir="src/mbrimberry_files/Submissions",transform=transform,type="packet")

# adjust for percentage of train-to-test split, default to 80-20
train_dataset,test_dataset = random_split(packet_dataset,[0.8,0.2])

# epoch of best loss for checkpoint loading
global best_loss_epoch
best_loss_epoch = None


# need dataloaders for both test and training data

train_loader = DataLoader(
    dataset= train_dataset,
    batch_size = 8,
    shuffle = True,
    collate_fn=collate_fn,
    num_workers= 0
)

# don't want to shuffle test
test_loader = DataLoader(
    dataset= test_dataset,
    batch_size = 8,
    shuffle = False,
    collate_fn=collate_fn,
    num_workers= 0
)

def create_model(num_objects_to_predict:int) -> fasterrcnn_resnet50_fpn:
    "Creates a model for num_objects_to_predict"
    # as there is not a lot of data, using a pre trained model is best for a starting point, using faster rcnn for this purpose
    model = fasterrcnn_resnet50_fpn(pretrained=True,weights="DEFAULT")
    
    # num_objects_to_predict+1 because background of image is considered object
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_objects_to_predict+1)
    
    return model

def train_model(model:fasterrcnn_resnet50_fpn, num_epochs: int):
    "Trains model passed in using train_loader for num_epochs"
    model.to(device)
    
    # parameters for early stopping
    best_loss = None
    patience = 5 # patience is number of epochs it waits to see if loss gets better before stopping
    patience_counter = 0 # counts how many epochs has elapsed
    
    # parameter for checkpointing
    global best_loss_epoch
    best_loss_epoch = None

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)


    model.train()

    for epoch in range(num_epochs):
        print(f"--------------- TRAINING EPOCH: {epoch + 1} ---------------")
        loss = 0
        for images,targets in train_loader:
            # putting images on device, necessary for gpu training
            images = [image.to(device) for image in images]
            # putting bouding boxes and labels on device, necessary for gpu training
            
            # rcnn needs list of batches. Need to convert batch to list of len batch_size
            boxes_list = list(targets['boxes'])
            labels_list = list(targets['labels'])
            
            target_list = []
            for i in range(len(images)):
                d = {}
                temp = boxes_list[i]
                # converting bounding boxes from x_center,y_center,width,height to x_min,y_min,x_max,y_max absolute coardinates for faster rcnn for each label
                for j in range(len(boxes_list[i])):
                    # images are 816x1056
                    x_center_unnormalized = temp[j][0] * 816 
                    y_center_unnormalized = temp[j][1] * 1056
                    width_unnormalized = temp[j][2] * 816
                    height_unnormalized = temp[j][3] * 1056
                    
                    
                    
                    x_min = ((x_center_unnormalized - width_unnormalized)/2.0)
                    x_max = ((x_center_unnormalized + width_unnormalized)/2.0)
                    y_min = ((x_center_unnormalized - height_unnormalized)/2.0)
                    y_max = ((x_center_unnormalized + height_unnormalized)/2.0)
                    
                    if (x_min + x_max + y_min + y_max) != 0:
                        new_tensor = [x_min,y_min,x_max,y_max]
                        boxes_list[i][j] = torch.tensor(new_tensor)
                    else:
                        boxes_list[i] = torch.zeros((0,4), dtype=torch.float32)
                        
                    
                    
                    
                d['boxes'] = boxes_list[i]
                d['labels'] = labels_list[i]
                target_list.append(d)
                
                # sending bounding boxes and labels to device
                target_list = [{key:value.to(device) for key,value in target.items()} for target in target_list]
                
            
            
            # zero gradients out
            optimizer.zero_grad()
            
            # forward pass
            loss = model(images,target_list)
            
            # Compute total loss
            losses = sum(value for value in loss.values())
            

            # Backward pass and optimize
            losses.backward()
            optimizer.step()
            
            
        # early stopping setup and checking
        if best_loss is None:
            best_loss = losses.item()
        elif losses.item() > best_loss:
            patience_counter +=1
            if patience_counter > patience:
                print(f"Early stopping at epoch{epoch+1}, Loss: {losses.item()}")
                break
        
        # saving checkpoint only if current epoch was better in terms of loss than last
        if best_loss > losses.item():
            best_loss = losses.item()
            patience_counter = 0
            best_loss_epoch = epoch+1
            save_checkpoint(model,f"./models/checkpoints/checkpoint_epoch_{epoch+1}.pth")
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {losses.item()}")
        
    print("Done Training")

def test_model(model: fasterrcnn_resnet50_fpn):
    "Tests model using test loader"
    model.eval()
    
    with torch.no_grad():
        for images,targets in test_loader:
            # putting images on device, necessary for evaluation
            images = [image.to(device) for image in images]
            predictions = model(images)
            
    for prediction in predictions:
        # checking that boudning boxes is of type tensor
        boxes = prediction['boxes']
        labels = prediction['labels']
        scores = prediction['scores']
    
        print(f"bounding box coardinates[{boxes}]")
        print(f"label[{labels}]")
        print(f"confidence scores [{scores}]")
        
        assert(type(boxes) is torch.Tensor)    
        assert(type(labels) is torch.Tensor)    
        assert(type(scores) is torch.Tensor)
        
def save_model(model: fasterrcnn_resnet50_fpn, path:str):
    "Saves model to given path, path must be str and model must be fasterrcnn_resnet50_fpn"
    try:
        torch.save(model, path)
    except IOError:
        print("Error saving Model")
        
def save_checkpoint(model:fasterrcnn_resnet50_fpn, path:str):
    "saves a checkpoint of the model during training"
    torch.save(model.state_dict(),path)
    
def load_checkpoint(model:fasterrcnn_resnet50_fpn,path:str):
    "loads a checkpoint of the model at that checkpoint"
    model.load_state_dict(torch.load(path))


# Creating model
model = create_model(3)
# training
train_model(model,50)
load_checkpoint(model,f"./models/checkpoints/checkpoint_epoch_{best_loss_epoch}.pth")
save_model(model,"./models/id_periodNum_model.pt")
# testing
test_model(model)

