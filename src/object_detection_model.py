"""
    Austin Nolte     
"""



from typing import Tuple
import PIL.Image
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import IndividualIMGDataset, collate_fn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torch.utils.data import random_split
from PIL import Image
import PIL
import os

import pathlib
from pathlib import Path
from tqdm.auto import tqdm


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Data used by functions
image_width  = 816
image_height = 1056
model_path = Path("models/")
checkpoint_path = Path("models/checkpoints")
test_loader:DataLoader
train_loader:DataLoader
transform:transforms.Compose
# epoch of best loss for checkpoint loading
global best_loss_epoch
best_loss_epoch = None
num_classes = {"desk": 2,"caddy": 1,"packet": 2}

# train_loader = get_individual_data_loader("src/mbrimberry_files/Submissions",transform=transform,batch_size=32,shuffle=True,num_workers=4, type="assignments")
def create_dataloaders(targ_dir, type):
    global test_loader
    global train_loader
    create_transforms()
    # creating a split dataset of training and testing. First need a whole dataset of all images
    full_dataset = IndividualIMGDataset(targ_dir=targ_dir,transform=transform,type=type)

    # adjust for percentage of train-to-test split, default to 80-20
    train_dataset,test_dataset = random_split(full_dataset,[0.8,0.2])
    
    # need dataloaders for both test and training data
    train_loader = DataLoader(
        dataset= train_dataset,
        batch_size = 8,
        shuffle = True,
        collate_fn=collate_fn,
        num_workers= 0
    )
    print(train_loader)
    # don't want to shuffle test
    test_loader = DataLoader(
        dataset= test_dataset,
        batch_size = 8,
        shuffle = False,
        collate_fn=collate_fn,
        num_workers= 0
    )
def create_transforms():
    global transform
    transform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.Grayscale(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(size=(image_width,image_height))
    ])
def create_model(num_objects_to_predict:int,type="None") -> FasterRCNN:
    "Creates a model for num_objects_to_predict"
    
    # Failure cases
    
    if not isinstance(num_objects_to_predict,(int)):
        raise TypeError("num_objects_to_predict must be int")
    
    if num_objects_to_predict <= 0:
        raise ValueError("num_objects_to_predict must be greater than 0")
    
    if not isinstance(type,(str)):
        raise TypeError("type must be string")
    if type != "None":
        targ_dir = Path("src/mbrimberry_files/Submissions")
        create_dataloaders(targ_dir, type)
    else:
        create_transforms()
    # as there is not a lot of data, using a pre trained model is best for a starting point, using faster rcnn for this purpose
    model = fasterrcnn_resnet50_fpn(pretrained=True,weights="DEFAULT")
    
    # num_objects_to_predict+1 because background of image is considered object
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_objects_to_predict+1)
    
    return model

def train_model(model:FasterRCNN, num_epochs: int):
    "Trains model passed in using train_loader for num_epochs"
    
    # Failure cases
    
    if not isinstance(model, FasterRCNN):
        raise TypeError("model must be type torchvision.models.detection.FasterRCNN")
    
    if not isinstance(num_epochs,(int)):
        raise TypeError("num_epochs must be type int")
    
    if num_epochs <= 0:
        raise ValueError("num_epochs must be greater than 0")
    
    global train_loader
    global test_loader
    
    if train_loader is None:
       raise TypeError("train_loader is None")
    
    # checking for device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    
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
        for images,targets in tqdm(train_loader):
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
            save_checkpoint(model, checkpoint_path / f"checkpoint_epoch_{epoch+1}.pth")
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
            save_checkpoint(model, checkpoint_path / f"checkpoint_epoch_{epoch+1}.pth")
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {losses.item()}")
        
    print("Done Training")

def test_model(model: FasterRCNN):
    "Tests model using test loader"
    
    # Failure cases
    
    if not isinstance(model, FasterRCNN):
        raise TypeError("model must be type torchvision.models.detection.FasterRCNN")
    
    
    # checking for device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    
    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for images,targets in tqdm(test_loader):
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
        
def save_model(model: FasterRCNN, path:str):
    "Saves model to given path, path must be str and model must be FasterRCNN"
    
    # Failure Cases
    if not isinstance(model, FasterRCNN):
        raise TypeError("model must be type torchvision.models.detection.FasterRCNN")
    
    if not isinstance(path, str):
        raise TypeError("path must be type str")
    
    
    try:
        torch.save(model.state_dict(), path)
    except IOError:
        print("Error saving Model")
        
def save_checkpoint(model:FasterRCNN, path:str):
    "saves a checkpoint of the model during training"
    
    # Failure Cases
    if not isinstance(model, FasterRCNN):
        raise TypeError("model must be type torchvision.models.detection.FasterRCNN")
    
    if not isinstance(path, str) and not isinstance(path, Path):
        raise TypeError("path must be type str or Path")
    
    #if not os.path.exists(path):
    #    raise FileNotFoundError("path does not exist")
    
    try:
        torch.save(model.state_dict(), path)
    except IOError:
        print("Error saving Model")
    
def load_checkpoint(model:FasterRCNN,path:str):
    "loads a checkpoint of the model at that checkpoint"
    
    # Failure Cases
    if not isinstance(model, FasterRCNN):
        raise TypeError("model must be type torchvision.models.detection.FasterRCNN")
    
    if not isinstance(path, str):
        raise TypeError("path must be type str")
    
    if not os.path.exists(path):
        raise FileNotFoundError("path does not exist")
    
    
    model.load_state_dict(torch.load(path, weights_only=True))

def create_and_train_model(num_epochs:int,num_objects_to_predict:int, model_path: str, type:str):
    """This function creates and trains a model based on the dataloaders already coded into the file. 
    Will save to model_path, num_epochs must be int checkpoint_path is where checkpoints will be saved, else will be saved to ./models/checkpoints"""
    
    # Failure Cases
    if not isinstance(num_epochs,(int)):
        raise TypeError("num_epochs must be type int")
    
    if num_epochs <= 0:
        raise ValueError("num_epochs must be greater than 0")
    
    if not isinstance(model_path,(str)):
        raise TypeError("model_path must be type str")
    
    if model_path == "":
        raise FileNotFoundError("not valid path")
    
    
    # Creating model
    model = create_model(num_objects_to_predict=num_objects_to_predict,type=type)
    # training
    train_model(model,num_epochs)
    # if checkpoint_path:
    #     load_checkpoint(model,checkpoint_path)
    # elif best_loss_epoch is not None:
    load_checkpoint(model,f"./models/checkpoints/checkpoint_epoch_{best_loss_epoch}.pth")
    save_model(model,model_path)
    # testing
    test_model(model)

def predict_with_model(image, model:FasterRCNN, type:str):
    """
    uses id_periodNum_model, will use cuda or mac, will try and use cuda first then mac then cpu, image must be string or tensor 
    loads model from path, which must be a str
    image must be path to a image or a PIL.Image
    Returns:
        A tuple of(bounding_box_id,bounding_box_period_num,confidence_id_box,confidence_label_box,cropped id image, cropped periodNum Image).
    """
    if not isinstance(image, (str,PIL.Image.Image)):
        raise TypeError("image must be type str or PIL.Image")
    
    
    if isinstance(image,str) and not os.path.exists(image):
        raise FileNotFoundError("image path does not exist")
    
    if not isinstance(model,FasterRCNN):
        raise TypeError("model must be of type FasterRCNN")
    
    if not isinstance(type,str) or type is None:
        raise TypeError("type must be a str of either packet,desk, or caddy")
    
    # make sure type is in dictionary
    type = type.lower()
    if type not in num_classes:
        raise KeyError(type + " does not exist in dictionary of known class types")
    
    # checking for device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    
    model.eval()
    
    #convert to grayscale image
    if isinstance(image, str):
        image = Image.open(image).convert("L")
    
    image_to_crop = image
    image = transform(image)
    
    with torch.inference_mode():
        images = image.to(device)
        predictions = model([images])
        
        for prediction in predictions:
            # checking that bounding boxes is of type tensor
            boxes = prediction['boxes']
            labels = prediction['labels']
            scores = prediction['scores']
            
            # print(f"bounding box coordinates\n[{boxes}]")
            # print(f"label\n[{labels}]")
            # print(f"confidence scores\n[{scores}]")

            assert(isinstance(boxes, torch.Tensor))
            assert(isinstance(labels, torch.Tensor))
            assert(isinstance(scores, torch.Tensor))
        
        confidence_score_threshold = 0.10
        
        #iterate through all the classes provided from dataset starting from index 1 to the length of labels list
        #using lowercase "L" as label index to not get mixed up with other indices
        for l in range(0,num_classes.get(type)):
            #initialize data for each iteration
            highest_confidence = 0
            highest_confidence_index = 0
            prediction_box = ()
            prediction_label_score = ()
            new_image = None
            return_tuple = None
            for i in range(len(scores)):
                if scores[i].item() > confidence_score_threshold:
                    if labels[i].item() == l+1 and scores[i].item() > highest_confidence:
                        highest_confidence = labels[i].item()
                        highest_confidence_index = i
            
            if len(boxes) != 0:     
                x_min = int(boxes[highest_confidence_index][0].item())
                y_min = int(boxes[highest_confidence_index][1].item())
                x_max = int(boxes[highest_confidence_index][2].item())
                y_max = int(boxes[highest_confidence_index][3].item())
            
                width = x_max - x_min
                height = y_max - y_min

                # prediction_box is in format (x_min,y_min,x_max,y_max)
                prediction_box = (x_min,y_min,x_max,y_max)
            
                # prediction_label_score is in format (label,confidence socre of label)
                prediction_label_score = (labels[highest_confidence_index].item(),scores[highest_confidence_index].item())
                new_image = F.crop(image_to_crop,y_min,x_min,height,width)
                return_tuple = (prediction_box,prediction_label_score,new_image, l)
                #return tuples using yield so the state of the loop can be saved and iterated to find multiple boxes.
                yield return_tuple

if __name__ == "__main__":
    #create and train model for packet
    create_and_train_model(num_epochs=100,num_objects_to_predict=num_classes.get("packet"),model_path="./models/packetmodel.pt",type="packet")
    #train model for desk caddy
    create_and_train_model(num_epochs=100,num_objects_to_predict=num_classes.get("caddy"),model_path="./models/caddy_model.pt",type="caddy")
    #train model for desk
    create_and_train_model(num_epochs=100,num_objects_to_predict=num_classes.get("desk"),model_path="./models/desk_model.pt",type="desk")
    #test trained model with sample image. will iterate through generator function outputting tuples of boxes
    image_generator = predict_with_model(image="src/mbrimberry_files/Submissions/03 14 2024/Activity  478411 - 03 14 2024/Desk Images/desk_1.png",type="desk",model_path="models/deskmodel.pt")
    for i in image_generator:
        print(i)
