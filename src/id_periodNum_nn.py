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
from data_loader import get_individual_data_loader, IndividualIMGDataset, collate_fn
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


# UNCOMMENT BELOW IF YOU HAVE A CUDA-ENABLED NVIDIA GPU, otherwise uses CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# USE THIS IF YOU HAVE A MAC WITH APPLE SILICON
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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

# train_loader = get_individual_data_loader("src/mbrimberry_files/Submissions",transform=transform,batch_size=32,shuffle=True,num_workers=4, type="assignments")
def create_dataloaders(targ_dir, type):
    global test_loader
    global train_loader
    create_transforms()
    # creating a split dataset of training and testing. First need a whole dataset of all images
    packet_dataset = IndividualIMGDataset(targ_dir=targ_dir,transform=transform,type=type)

    # adjust for percentage of train-to-test split, default to 80-20
    train_dataset,test_dataset = random_split(packet_dataset,[0.8,0.2])
    
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
    
    if train_loader is None:
        raise TypeError("train_loader is None")
    
    
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
    
    if not os.path.exists(path):
        raise FileNotFoundError("path does not exist")
    
    torch.save(model.state_dict(),path)
    
def load_checkpoint(model:FasterRCNN,path:str):
    "loads a checkpoint of the model at that checkpoint"
    
    # Failure Cases
    if not isinstance(model, FasterRCNN):
        raise TypeError("model must be type torchvision.models.detection.FasterRCNN")
    
    if not isinstance(path, str):
        raise TypeError("path must be type str")
    
    if not os.path.exists(path):
        raise FileNotFoundError("path does not exist")
    
    
    model.load_state_dict(torch.load(path))

def create_and_train_model(num_epochs:int,num_objects_to_predict:int, model_path: str, type:str):
    "This function creates and trains a model based on the dataloaders already coded into the file. Will save to model_path, num_epochs must be int checkpoint_path is where checkpoints will be saved, else will be saved to ./models/checkpoints"
    
    # Failure Cases
    if not isinstance(num_epochs,(int)):
        raise TypeError("num_epochs must be type int")
    
    if num_epochs <= 0:
        raise ValueError("num_epochs must be greater than 0")
    
    if not isinstance(model_path,(str)):
        raise TypeError("model_path must be type str")
    # Shouldn't we be able to make a new model?
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError("model_path does not exist")
    
    # if not isinstance(checkpoint_path,(str)) and checkpoint_path is not None:
    #     raise TypeError("checkpoint_path must be type str or None")
    
    # if checkpoint_path is not None and not os.path.exists(checkpoint_path):
    #     raise FileNotFoundError("checkpoint_path does not exist")
    
    
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

def predict_with_id_model(image, model_path:str) -> Tuple:
    """
    uses id_periodNum_model, will use cuda or mac, will try and use cuda first then mac then cpu, image must be string or tensor 
    loads model from path, which must be a str
    image must be path to a image or a PIL.Image
    Returns:
        A tuple of(bounding_box_id,bounding_box_period_num,confidence_id_box,confidence_label_box,cropped id image, cropped periodNum Image).
    
    """
    
    if not isinstance(image, (str,PIL.Image.Image)):
        raise TypeError("image must be type str or PIL.Image")
    
    if not isinstance(model_path,str):
        raise TypeError("model_path must be type str")
    
    if isinstance(image,str) and not os.path.exists(image):
        raise FileNotFoundError("image path does not exist")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("model_path does not exist")
    
    if torch.cuda.is_available():
        torch.device('cuda')
    elif torch.backends.mps.is_available():
        torch.device('mps')
    else:
        torch.device('cpu')
    
    model = create_model(2)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    
    model.eval()
    
    if type(image) is str:
        image = Image.open(image).convert("L")
    
    image_to_crop = image
    image = transform(image)
    
    with torch.inference_mode():
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
        
        return_tuple = None
        
        for i in range(len(scores)):
            if scores[i].item() > confidence_score_threshold:
                if labels[i].item() == 1 and scores[i].item() > highest_confidence_1:
                    highest_confidence_1 = labels[i].item()
                    highest_confidence_1_index = i
                elif labels[i].item() == 2 and scores[i].item() > highest_confidence_2:
                    highest_confidence_2 = labels[i].item()
                    highest_confidence_2_index = i
        
        if len(boxes) != 0:     
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
            return_tuple = (prediction_box_id,prediction_box_period_num,prediction_label_score_id,prediction_label_score_period_num,id_image,period_num_image)
        
        
        if return_tuple:
            return return_tuple
        else:
            return () 
def predict_with_caddy_model(image, model_path:str) -> Tuple:
    """
    uses caddy_model, will use cuda or mac, will try and use cuda first then mac then cpu, image must be string or tensor 
    
    Returns:
        A tuple of(prediction_box_cnum,prediction_label_score_cnum,cnum_image).
    
    """
    
    if torch.cuda.is_available():
        torch.device('cuda')
    elif torch.backends.mps.is_available():
        torch.device('mps')
    else:
        torch.device('cpu')
    
    model = create_model(1)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    
    model.eval()
    
    if type(image) is str:
        image = Image.open(image).convert("L")
    
    image_to_crop = image
    image = transform(image)
    
    with torch.inference_mode():
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
        
        prediction_box_cnum = ()

        
        prediction_label_score_cnum = ()
        
        cnum_image = None
        
        
        for i in range(len(scores)):
            if scores[i].item() > confidence_score_threshold:
                if labels[i].item() == 1 and scores[i].item() > highest_confidence_1:
                    highest_confidence_1 = labels[i].item()
                    highest_confidence_1_index = i
        if len(boxes) != 0:           
            x_min = int(boxes[highest_confidence_1_index][0].item())
            y_min = int(boxes[highest_confidence_1_index][1].item())
            x_max = int(boxes[highest_confidence_1_index][2].item())
            y_max = int(boxes[highest_confidence_1_index][3].item())
            
            width = x_max - x_min
            height = y_max - y_min
    
        # prediction_box is in format (x_min,y_min,x_max,y_max)
            print(prediction_box_cnum)
            prediction_box_cnum = (x_min,y_min,x_max,y_max)
            print(prediction_box_cnum)
            # prediction_label_score is in format (label,confidence socre of label)
            prediction_label_score_cnum = (labels[highest_confidence_1].item(),scores[highest_confidence_1].item())
            cnum_image = F.crop(image_to_crop,y_min,x_min,height,width)
        
        return prediction_box_cnum,prediction_label_score_cnum,cnum_image  
# id_box, period_num_box, label_score, id_score, id_image, period_num_image = predict_with_id_model("./src/mbrimberry_files/Submissions/03 13 2024/Activity  474756 - 03 13 2024/Activity Packet/activity_1.png", model_path="./models/id_periodNum_model.pt")

# id_box, period_num_box, label_score, id_score, id_image, period_num_image = predict_with_id_model(image_path="./src/test_files/obj_detect_test/test_image.png",model_path="./models/id_periodNum_model.pt")
# if period_num_image:
    # period_num_image.show()
# if id_image:
    # id_image.show()
    
create_and_train_model(num_epochs=10,num_objects_to_predict=2,model_path="./models/test_model.pt",type="packet")