"""
    Austin Nolte
    
    Unit tests for data_loader
    
"""


import pytest
import torch
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
from torch.utils.data import Dataset, DataLoader
import data_loader as dl

"""
Helper functions to access each of the data loader functions
"""
jsonDS = None
imgDS = None

def makeJSONDataset(json_file,transform=None):
    global jsonDS
    jsonDS = dl.JSONDataset(json_file=json_file,transform=transform)

def makeIndividualImgDataset(targ_dir: str,transform=None,type = "desk"):
    global imgDS
    imgDS = dl.IndividualIMGDataset(targ_dir=targ_dir,transform=transform,type=type)

def bounding_box_txt_parse(txt_file, num_of_classes) -> Tuple[int,torch.tensor]:
    dl.bounding_box_txt_parse(txt_file=txt_file,num_of_classes=num_of_classes)
    
def find_classes(targ_dir: str, pattern: str) -> Tuple[List[str], Dict[int, str]]:
    dl.find_classes(targ_dir=targ_dir,pattern=pattern)
    
def DrawBox(img,box,classes):
    dl.DrawBox(img=img,box=box,classes=classes)
    
def collate_fn(batch):
    dl.collate_fn(batch=batch)

def get_packet_data_loader(json_file, batch_size=32, shuffle=True, num_workers=0):
    dl.get_packet_data_loader(json_file=json_file,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    
def get_individual_data_loader(targ_dir, transform, batch_size=32, shuffle=False, num_workers=0, type = "desk"):
    dl.get_individual_data_loader(targ_dir=targ_dir,transform=transform,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,type=type)

    
"""
pytest cases
"""

def test_makeIndividualImgDataset():
    # Test case 1: targ_dir incorrect type
    with pytest.raises(TypeError, match= "targ_dir must be a string"):
        makeIndividualImgDataset(targ_dir=None)
    
    # Test case 2: targ_dir is invalid
    with pytest.raises(FileNotFoundError):
        makeIndividualImgDataset(targ_dir="badPath")

    # Test case 3: type is invalid type
    with pytest.raises(ValueError, match =" 'type' must be a string of either 'desk' for desk images or 'assignments' for assignments"):
        makeIndividualImgDataset(targ_dir="./src/mbrimberry_files/Submissions",type = 3)
    
    # Test case 4: transform is invalid type
    with pytest.raises(TypeError, match= " 'transform' must be of type torchvision.transforms.V2 or a list of torchvision.transforms.V2"):
        makeIndividualImgDataset(targ_dir="./src/mbrimberry_files/Submissions",transform='error')
        
    # Test case 5: transform is a list but not of transforms
    with pytest.raises(TypeError,match="'transform' must be of type torchvision.transforms.V2 or a list of torchvision.transforms.V2"):
        makeIndividualImgDataset(targ_dir="./src/mbrimberry_files/Submissions",transform=['error',5,'this should not work'])

def test_make_jsonDataset():
    # Test case 1: json_file incorrect type
    with pytest.raises(TypeError, match= "json_file should be a path to a jsonfile"):
        makeJSONDataset(json_file=None)
    
    # Test case 2: json_file is invalid
    with pytest.raises(FileNotFoundError):
        makeJSONDataset(json_file="badPath")
        
    # Test case 3: transform is invalid type
    with pytest.raises(TypeError, match= " 'transform' must be of type torchvision.transforms.V2 or a list of torchvision.transforms.V2"):
        makeJSONDataset(json_file="./src/test_files/test.json",transform='error')
        
    # Test case 4: transform is a list but not of transforms
    with pytest.raises(TypeError,match="'transform' must be of type torchvision.transforms.V2 or a list of torchvision.transforms.V2"):
        makeIndividualImgDataset(targ_dir="./src/test_files/test.json",transform=['error',5,'this should not work'])

def test_bounding_box_txt_parse():
    
    # Test case 1: txt_file is invalid type
    with pytest.raises(TypeError, match="txt_file needs to be a path to a file"):
        bounding_box_txt_parse(txt_file=5,num_of_classes=4)
    
    # Test case 2: txt_file is a bad path
    with pytest.raises(FileNotFoundError):
        bounding_box_txt_parse(txt_file="badPath",num_of_classes=4)
        
    # Test case 3: num_of_classes is invalid type
    with pytest.raises(TypeError, match="num_of_classes must be of type int"):
        bounding_box_txt_parse(txt_file="./src/test_files/activity_1.txt",num_of_classes='should not work')
        
    # Test case 4: num_of_classes is too large of a number
    with pytest.raises(ValueError, match="num_of_classes too large"):
        bounding_box_txt_parse(txt_file="./src/test_files/activity_1.txt",num_of_classes=3)
        
    # Test case 5: assert that return type is of Tuple(int,torch.tensor)
    result = bounding_box_txt_parse(txt_file="./src/test_files/activity_1.txt",num_of_classes=2)
    
    assert isinstance(result[0],int)
    assert isinstance(result[1],torch.tensor)
    
    # Test case 6: assert that the len of return is 2
    assert(len(result) == 2)

def test_find_classes():
    # Test case 1: targ_dir is invalid type
    with pytest.raises(TypeError, match="targ_dir must be of type str"):
        find_classes(targ_dir=5,pattern="testPattern")
    
    # Test case 2: targ_dir is not a valid directory
    with pytest.raises(FileNotFoundError):
        find_classes(targ_dir="badPath",pattern="testPattern")
        
    # Test Case 3: pattern is invalid type
    with pytest.raises(TypeError, match= "pattern must be of type str"):
        find_classes(targ_dir="./src/test_files/find_classes_test_dir",pattern=5)

    # Test Case 4: assert return type is None when bad pattern
    result = find_classes(targ_dir="./src/test_files/find_classes_test_dir",pattern="badPattern")
    
    assert isinstance(result,None)
    
    # Test case 5: assert that return type is correct being Tuple(list[str], dict[int,str])
    result = find_classes(targ_dir="./src/test_files/find_classes_test_dir",pattern="Desk Images")
    
    assert isinstance(result[0],list)
    assert isinstance(result[1], dict)
    
    # asserting that list is of string and dictionary is correct as well
    assert isinstance((result[0])[0],str)
    assert isinstance((result[1]).values(),str)
    
def test_draw_box():
    # bad tensor 
    bad_tensor_test = torch.tensor([[1., -1.], [1., -1.]])
    
    # good tensor for rgb image
    good_tensor_test =  torch.tensor([
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    
    # good tensor for box
    good_tensor_box = torch.tensor([10,110,50,90])
    
    # good classes list of tensors
    good_classes_list = [torch.tensor('test1'),torch.tensor('test2')]
    
    # Test case 1: img is of invalid type
    with pytest.raises(TypeError, match="img must be of type torch.tensor"):
        DrawBox(img=None,box=good_tensor_box,classes=good_classes_list)
        
    # Test Case 2: box is of invalid type
    with pytest.raises(TypeError, match="box must be of type torch.tensor"):
        DrawBox(img=good_tensor_test,box=None,classes=good_classes_list)
    
    # Test Case 3: classes is of invalid type
    with pytest.raises(TypeError, match="box must be of type torch.tensor"):
        DrawBox(img=good_tensor_test,box=good_tensor_box,classes=None)
    
    # Test Case 4: assert that return is of type torch.tensor
    result = DrawBox(img=good_tensor_test,box=good_tensor_box,classes=good_classes_list)
    
    assert isinstance(result,torch.tensor)
    
def test_collate_fn():
    
    #Test case 1: batch is of invalid type
    with pytest.raises(TypeError,match= "batch must be of type list[torch.tensor,torch.tensor,dict[int,str]]"):
        collate_fn('badType')
        
def test_get_json_dataloader():
    # Test case 1: json_file incorrect type
    with pytest.raises(TypeError, match= "json_file should be a path to a jsonfile"):
        get_packet_data_loader(json_file=None)
        
    # Test case 2: json_file is invalid
    with pytest.raises(FileNotFoundError):
        get_packet_data_loader(json_file="badPath")
    
    # Test case 3: batch_size is invalid type
    with pytest.raises(TypeError, match= "batch_size should be of type int"):
        get_packet_data_loader(json_file="./src/test_files/test.json",batch_size='none')
        
    # Test case 3: shuffle is invalid type
    with pytest.raises(TypeError, match= "shuffle should be of type boolean"):
        get_packet_data_loader(json_file="./src/test_files/test.json",shuffle='none')
    
    # Test case 4: num_workers is invalid type
    with pytest.raises(TypeError, match= "num_workers should be of type int"):
        get_packet_data_loader(json_file="./src/test_files/test.json",num_workers='none')
        
    # Test case 5: should return type torch.utils.data.DataLoader
    result = get_packet_data_loader(json_file="./src/test_files/test.json")
    
    assert isinstance(result,torch.utils.data.DataLoader)

def test_get_individual_data_loader():
    # Test case 1: json_file incorrect type
    with pytest.raises(TypeError, match= "targ_dir should be a path to a directory"):
        get_packet_data_loader(targ_dir=None)
        
    # Test case 2: json_file is invalid
    with pytest.raises(FileNotFoundError):
        get_packet_data_loader(targ_dir="badPath")
    
    # Test case 3: batch_size is invalid type
    with pytest.raises(TypeError, match= "batch_size should be of type int"):
        get_packet_data_loader(targ_dir="./src/mbrimberry_files/Submissions",batch_size='none')
        
    # Test case 3: shuffle is invalid type
    with pytest.raises(TypeError, match= "shuffle should be of type boolean"):
        get_packet_data_loader(targ_dir="./src/mbrimberry_files/Submissions",shuffle='none')
    
    # Test case 4: num_workers is invalid type
    with pytest.raises(TypeError, match= "num_workers should be of type int"):
        get_packet_data_loader(json_file="./src/mbrimberry_files/Submissions",num_workers='none')
        
    # Test case 5: should return type torch.utils.data.DataLoader
    result = get_packet_data_loader(targ_dir="./src/mbrimberry_files/Submissions")
    
    assert isinstance(result,torch.utils.data.DataLoader)