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

filestructure = "*/Activity"
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

class DeskTopDataset(Dataset):
    # Initializes the dataset
    def __init__(self,targ_dir: str,transform=None):
        # TODO check if these two (img_paths and bbox_paths) get the same index
        # This grabs all of the paths to the desk_1 images and puts them into a sorted list
        img_paths = list(sorted(Path(targ_dir).glob("*/*/Desk Images/desk_1.png")))  
        #img_paths = list(sorted(Path(targ_dir).glob("*/*/Activity Packet/activity*.png")))  
        # This grabs the paths for all the txt files that has the bounding boxes and puts them into a sorted list
        # self.bbox_paths = list(sorted(Path(targ_dir).glob("*/*/Desk Images/desk_1.txt")))
        # self.bbox_paths = list(sorted(Path(targ_dir).glob("*/*/Activity Packet/activity*.txt")))
        # The "globbed too hard" for loop
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
        self.classes, self.class_to_idx = find_classes(targ_dir,"Desk Images")
        
    # Helper Function for loading images that __getitem__ will use
    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index][0]
        return Image.open(image_path)
    
    # Helper Function for loading bounding boxes that __getitme will use
    def load_bbox(self, index: int) -> Tuple[int,torch.Tensor]:
        if self.paths[index][1] == None:
            return None,None
        boundingbox_path = self.paths[index][1]
        # passing boundingbox file into the txtParse function to return the tensor and label
        return txtParse(boundingbox_path)
    
    # function that returns the length of dataset
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img = self.load_image(index)
        class_idx, bbox = self.load_bbox(index)
        # TODO Handle Transforms(have to apply them to coordinates also maybe)
        if self.transform:
            return self.transform(img, bbox, class_idx)
        else:
            return img, bbox, class_idx

def txtParse(txt_file) -> Tuple[int,torch.tensor]:
    "Parses the JSON file get the bounding boxes"
    if txt_file:
        print(str(txt_file))
        file = open(txt_file)
        coords = []
        class_idx = []
        for line in file:
            box_data = line.split()
            class_idx.append(int(box_data.pop(0)))
            coord = [float(word) for word in box_data]
            coords.append(coord)
            
        bbox = torch.tensor(coords)
        return class_idx, bbox
    
    
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
        print(f"Class File Found: {file}")
        txt = open(file)
        classes = [str.strip(line) for line in txt]
        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
        return classes, class_to_idx
    else:
        print("Class File Not Found.")  
        return


# Create the DataLoader
def get_packet_data_loader(json_file, batch_size=32, shuffle=True, num_workers=0):
    dataset = JSONDataset(json_file=json_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_desk_data_loader(targ_dir,txt_file, batch_size=32, shuffle=True, num_workers=0):
    dataset = DeskTopDataset(txt_file = txt_file, targ_dir = targ_dir)
    #TODO add transforms
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers )
    

# # Test the DataLoader
# if __name__ == '__main__':
#     json_path = 'path\to\file.json'  # Replace with our json file path
#     dataloader = get_packet_data_loader(json_path, batch_size=4)

#     # Iterate through the DataLoader
#     for i, batch in enumerate(dataloader):
#         print(f'Batch {i}:')
#         print('Features:', batch['features'])
#         print('Labels:', batch['label'])
