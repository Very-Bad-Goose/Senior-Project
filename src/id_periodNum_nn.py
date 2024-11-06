"""
    Austin Nolte     
"""



from typing import Tuple
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import get_individual_data_loader, IndividualIMGDataset, collate_fn
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import random_split
from PIL import Image

import pathlib
from pathlib import Path

# UNCOMMENT BELOW IF YOU HAVE A CUDA-ENABLED NVIDIA GPU, otherwise uses CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# USE THIS IF YOU HAVE A MAC WITH APPLE SILICON
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Transform to apply to data coming in
image_width  = 816
image_height = 1056

transform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.Grayscale(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(size=(image_width,image_height))
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
    best_loss_epoch = 1

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
            for label in labels_list:
                label +=1
            
            target_list = []
            for i in range(len(images)):
                d = {}
                temp = boxes_list[i]
                # converting bounding boxes from x_center,y_center,width,height to x_min,y_min,x_max,y_max absolute coardinates for faster rcnn for each label
                for j in range(len(boxes_list[i])):
                    
                    x_center_unnormalized = temp[j][0] * image_width 
                    y_center_unnormalized = temp[j][1] * image_height
                    width_unnormalized = temp[j][2] * image_width
                    width_unnormalized /= 2
                    height_unnormalized = temp[j][3] * image_height
                    height_unnormalized /= 2
                    
                    
                    x_min = ((x_center_unnormalized - width_unnormalized))
                    x_max = ((x_center_unnormalized + width_unnormalized))
                    y_min = ((y_center_unnormalized - height_unnormalized))
                    y_max = ((y_center_unnormalized + height_unnormalized))
                    
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
            save_checkpoint(model,f"./models/checkpoints/checkpoint_epoch_{epoch+1}.pth")
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
    
        # print(f"bounding box coardinates:[{boxes}]\n\n")
        # print(f"label:[{labels}]\n\n")
        # print(f"confidence scores:[{scores}]\n\n")
        
        assert(type(boxes) is torch.Tensor)    
        assert(type(labels) is torch.Tensor)    
        assert(type(scores) is torch.Tensor)
    print('Done Testing after Train')
        
def save_model(model: fasterrcnn_resnet50_fpn, path:str):
    "Saves model to given path, path must be str and model must be fasterrcnn_resnet50_fpn"
    try:
        torch.save(model.state_dict(), path)
    except IOError:
        print("Error saving Model")
        
def save_checkpoint(model:fasterrcnn_resnet50_fpn, path:str):
    "saves a checkpoint of the model during training"
    torch.save(model.state_dict(),path)
    
def load_checkpoint(model:fasterrcnn_resnet50_fpn,path:str):
    "loads a checkpoint of the model at that checkpoint"
    model.load_state_dict(torch.load(path))

def create_and_train_model(num_epochs:int):
    "This function creates and trains a model based on the dataloaders already coded into the file. Will save to ./models/id_periodNum_model.pt, num_epochs must be int"
    # Creating model
    model = create_model(2)
    # training
    train_model(model,num_epochs)
    load_checkpoint(model,f"./models/checkpoints/checkpoint_epoch_{best_loss_epoch}.pth")
    save_model(model,"./models/id_periodNum_model.pt")
    # testing
    test_model(model)

def predict_with_id_model(image) -> Tuple:
    """
    uses id_periodNum_model, will use cuda or mac, will try and use cuda first then mac then cpu, image must be string or tensor 
    
    Returns:
        A tuple of(bounding_box_id,bounding_box_period_num,confidence_id_box,confidence_label_box,cropped id image, cropped periodNum Image).
    
    """
    
    if torch.cuda.is_available():
        torch.device('cuda')
    elif torch.backends.mps.is_available():
        torch.device('mps')
    else:
        torch.device('cpu')
    
    model = create_model(2)
    model.load_state_dict(torch.load("./models/id_periodNum_model.pt", weights_only=True))
    model.to(device)
    
    model.eval()
    
    if type(image) is str:
        image = Image.open(image).convert("L")
    
    image_to_crop = image
    image = transform(image)
    
    with torch.no_grad():
        images = image.to(device)
        predictions = model([images])
        
        for prediction in predictions:
            # checking that boudning boxes is of type tensor
            boxes = prediction['boxes']
            labels = prediction['labels']
            scores = prediction['scores']
            
            # print(f"bounding box coardinates\n[{boxes}]")
            # print(f"label\n[{labels}]")
            # print(f"confidence scores\n[{scores}]")

            assert(type(boxes) is torch.Tensor)    
            assert(type(labels) is torch.Tensor)    
            assert(type(scores) is torch.Tensor)
        
        confidence_score_threshold = 0.50
        
        highest_confidence_1 = 0
        highest_confidence_1_index = 0
        
        highest_confidence_2 = 0
        highest_confidence_2_index = 0
        
        prediction_box_id = ()
        prediction_box_period_num = ()
        
        prediction_label_score_id = ()
        prediction_label_score_period_num = ()
        
        id_image = None
        period_num_image = None
        
        
        for i in range(len(scores)):
            if scores[i].item() > confidence_score_threshold:
                if labels[i].item() == 1 and scores[i].item() > highest_confidence_1:
                    highest_confidence_1 = labels[i].item()
                    highest_confidence_1_index = i
                elif labels[i].item() == 2 and scores[i].item() > highest_confidence_2:
                    highest_confidence_2 = labels[i].item()
                    highest_confidence_2_index = i
                
        x_min_1 = int(boxes[highest_confidence_1_index][0].item())
        y_min_1 = int(boxes[highest_confidence_1_index][1].item())
        x_max_1 = int(boxes[highest_confidence_1_index][2].item())
        y_max_1 = int(boxes[highest_confidence_1_index][3].item())
        
        x_min_2 = int(boxes[highest_confidence_2_index][0].item())
        y_min_2 = int(boxes[highest_confidence_2_index][1].item())
        x_max_2 = int(boxes[highest_confidence_2_index][2].item())
        y_max_2 = int(boxes[highest_confidence_2_index][3].item())
        
        width_1 = x_max_1 - x_min_1
        height_1 = y_max_1 - y_min_1
        
        width_2 = x_max_2 - x_min_2
        height_2 = y_max_2 - y_min_2
    
        # prediction_box is in format (x_min,y_min,x_max,y_max)
        prediction_box_id = (x_min_1,y_min_1,x_max_1,y_max_1)
        prediction_box_period_num = (x_min_2,y_min_2,x_max_2,y_max_2)
    
        # prediction_label_score is in format (label,confidence socre of label)
        prediction_label_score_id = (labels[highest_confidence_1].item(),scores[highest_confidence_1].item())
        prediction_label_score_period_num = (labels[highest_confidence_2].item(),scores[highest_confidence_2].item())
        id_image = F.crop(image_to_crop,y_min_1,x_min_1,height_1,width_1)
        period_num_image = F.crop(image_to_crop,y_min_2,x_min_2,height_2,width_2)
        
        return prediction_box_id,prediction_box_period_num,prediction_label_score_id,prediction_label_score_period_num,id_image,period_num_image
