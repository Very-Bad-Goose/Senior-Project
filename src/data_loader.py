#change this line to match our formatting if needed
'''features = torch.tensor([sample['Data1']... etc'''

''' Add to trainer script
from data_loader import get_data_loader

train_loader = get_data_loader('train.json', batch_size=32)
test_loader = get_data_loader('test.json', batch_size=32)
'''

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import json
import pathlib
from pathlib import Path
import os
from typing import Dict, List, Tuple
import re
from PIL import Image
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

class JSONDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as file:
            self.data = json.load(file)  # Load the JSON file into a list of dictionaries
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract the features and label from the dictionary
        sample = self.data[idx]
        features = torch.tensor([sample['Data1'], sample['Data2'], sample['Data3']], dtype=torch.float32)
        label = torch.tensor(sample['dataLabel'], dtype=torch.long)  # Adjust the key as per your JSON format

        sample = {'features': features, 'label': label}

        # Apply any transformations if specified
        if self.transform:
            sample = self.transform(sample)

        return sample

class IndividualIMGDataset(Dataset):
    # Initializes the dataset
    def __init__(self,targ_dir: str,transform=None,type = "desk"):
        # This grabs all of the paths to the desk_1 images and puts them into a sorted list
        if type == "desk":
            img_paths = list(sorted(Path(targ_dir).glob("*/*/Desk Images/desk_1.png")))
        if type == "caddy":
            img_paths = list(sorted(Path(targ_dir).glob("*/*/Desk Images/desk_2.png")))  
        else:
            img_paths = list(sorted(Path(targ_dir).glob("*/*/Activity Packet/activity*.png")))
        #  
        # This searches for the associated txt file for the image file
        self.paths = []
        for img in img_paths:
            img_name = img.stem + '.txt'
            img_dir = img.parent
            txt_file = None
            for f in img_dir.iterdir():
                if f.name == img_name:
                    txt_file = f
                    break
            self.paths.append((img,txt_file)) 
        # We will apply this transform to the data (a transform can be a list of multiple other transforms)
        self.transform = transform
        # This is the set of classes defined by classes.txt, it also features a dictionary that has class_to_idx
        if type == "desk" or type == "caddy":
            self.classes, self.class_to_idx = find_classes(targ_dir,"Desk Images")
        else: 
            self.classes, self.class_to_idx = find_classes(targ_dir,"Activity Packet")
        
    # Helper Function for loading images that __getitem__ will use
    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index][0]
        return Image.open(image_path).convert("RGB")
    
    # Helper Function for loading bounding boxes that __getitme will use
    def load_bbox(self, index: int) -> Tuple[int,torch.Tensor]:
        boundingbox_path = self.paths[index][1]
        # passing boundingbox file into the txtParse function to return the tensor and label
        return bounding_box_txt_parse(boundingbox_path, len(self.classes))
    
    # function that returns the length of dataset
    def __len__(self) -> int:
        return len(self.paths)
    
    # Function to get specified item, used by the data loader to obtain data
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img = self.load_image(index)
        target = {}
        target["labels"], target['boxes'] = self.load_bbox(index)
        if self.transform:
            img, target['boxes'] = self.transform(img, target['boxes'])
        return img, target

# This parses the txt file and gets the bounding box and their classes
def bounding_box_txt_parse(txt_file, num_of_classes) -> Tuple[int,torch.tensor]:
    "Parses the txt file get the bounding boxes"
    coords = []
    filled_class_idx = []
    class_idx = list(range(num_of_classes))
    if txt_file:
        file = open(txt_file)
        for line in file:
            box_data = line.split()
            idx = int(box_data.pop(0))
            filled_class_idx.append(idx)
            class_idx.remove(idx)
            coord = [float(word) for word in box_data]
            coords.append(coord)
        bbox = torch.tensor(coords, dtype=torch.float32)
    else:
        return class_idx, torch.zeros(size=(num_of_classes,4), dtype=torch.float32)
    # Fills tensor with zeros for remaining classes
    for num in class_idx:
        filled_class_idx.append(num)
        no_bbox = torch.zeros(size=(1,4), dtype=torch.float32)
        bbox = torch.cat((bbox, no_bbox), dim=0)
                
        
    return filled_class_idx, bbox

        
    
    
def find_classes(targ_dir: str, pattern: str) -> Tuple[List[str], Dict[int, str]]:
    "Find Classes from classes.txt files and returns a list of classes and the dictionary including the indexes to those classes"
    #Handle specification of file
    x = re.findall("classes.txt$", pattern)
    if x:
        print(pattern)
    else:
        pattern = pattern + "\classes.txt"
        print(pattern)
        
    #Return classes and idx those classses
    for file in pathlib.Path(targ_dir).rglob(pattern):
        txt = open(file)
        classes = [str.strip(line) for line in txt]
        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
        return classes, class_to_idx
    else:
        print("Class File Not Found.")  
        return

# Returns image with bounding boxes drawn on   
def DrawBox(img,box,classes):
    bbox = box_convert(box,'cxcywh','xyxy')
    for i in range(bbox.shape[0]):
        bbox[i][0] = bbox[i][0] * img.shape[2]
        bbox[i][1] = bbox[i][1] * img.shape[1]
        bbox[i][2] = bbox[i][2] * img.shape[2]
        bbox[i][3] = bbox[i][3] * img.shape[1]
    colors = []
    class_to_colors = {
            0 : "blue",
            1 : "yellow"
    }
    for i in classes:
        if isinstance(i, torch.Tensor):
            colors.append(class_to_colors[i.item()])
        else:
            colors.append(class_to_colors[i])
        
    bimg = draw_bounding_boxes(img, bbox, colors=colors, width=5)
    return bimg

def collate_fn(batch):
    images = []
    bboxes = []
    labels = []
    for item in batch:
        images.append(item[0])
        bboxes.append(item[1]["boxes"])
        labels.append(torch.tensor(item[1]["labels"]))
    images = torch.stack(images,dim=0)
    bboxes = torch.stack(bboxes,dim=0)
    labels = torch.stack(labels,dim=0)
    targets = {"boxes" : bboxes,
               "labels" : labels}
    return images, targets

# Create the DataLoader
def get_packet_data_loader(json_file, batch_size=32, shuffle=True, num_workers=0):
    dataset = JSONDataset(json_file=json_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_individual_data_loader(targ_dir, transform, batch_size=32, shuffle=False, num_workers=0, type = "desk"):
    dataset = IndividualIMGDataset(targ_dir = targ_dir, transform=transform, type = type)
    return DataLoader(dataset,
                      batch_size=batch_size, 
                      shuffle=shuffle, 
                      collate_fn=collate_fn, 
                      num_workers=num_workers )

# Test the DataLoader
if __name__ == '__main__':
    json_path = 'path\to\file.json'  # Replace with our json file path
    dataloader = get_packet_data_loader(json_path, batch_size=4)

    # Iterate through the DataLoader
    for i, batch in enumerate(dataloader):
        print(f'Batch {i}:')
        print('Features:', batch['features'])
        print('Labels:', batch['label'])
