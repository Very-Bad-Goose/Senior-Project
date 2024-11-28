""" 
Testing id_periodNum_nn.py and its functions
Austin Nolte
"""

import pytest
import torch
import torch.torch_version
import torchvision.transforms.v2 as transforms
import json
import pathlib
from pathlib import Path
import os
from typing import Dict, List, Tuple
import re
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN

# from id_periodNum_nn import create_model, create_and_train_model, train_model, test_model, save_checkpoint, save_model, load_checkpoint, predict_with_id_model
import object_detection_model



"""
Pytest unit test cases
"""

def test_create_model():
    
    # Test case 1: incorrect type
    with pytest.raises(TypeError, match= "num_objects_to_predict must be int"):
        object_detection_model.create_model(num_objects_to_predict="")
    
    # Test case 2: none object passed
    with pytest.raises(TypeError, match= "num_objects_to_predict must be int"):
        object_detection_model.create_model(num_objects_to_predict=None)
        
    # Test case 3: num_objects_to_predict must be > 0
    with pytest.raises(ValueError, match= "num_objects_to_predict must be greater than 0"):
        object_detection_model.create_model(num_objects_to_predict=0)
    
    # Test case 4: num_objects_to_predict must be > 0
    with pytest.raises(ValueError, match= "num_objects_to_predict must be greater than 0"):
        object_detection_model.create_model(num_objects_to_predict=-1)
        
    # Test case 5: assert that retrun value is of type fasterrcnn_resnet50_fpn 
    model = object_detection_model.create_model(num_objects_to_predict=2)
    assert isinstance(model,FasterRCNN)

def test_train_model():
    
    # Test case 6: incorrect type for model
    with pytest.raises(TypeError, match= "model must be type torchvision.models.detection.FasterRCNN"):
        object_detection_model.train_model(model="", num_epochs=10)
        
    # Test case 7: none type for model
    with pytest.raises(TypeError, match= "model must be type torchvision.models.detection.FasterRCNN"):
        object_detection_model.train_model(model=None, num_epochs=10)

    model = object_detection_model.create_model(2,"packet")
    
    
    # Test case 8: incorrect type for num_epochs
    with pytest.raises(TypeError, match= "num_epochs must be type int"):
        object_detection_model.train_model(model=model, num_epochs="")
        
    # Test case 9: none type for num_epochs
    with pytest.raises(TypeError, match= "num_epochs must be type int"):
        object_detection_model.train_model(model=model, num_epochs=None)
        
    # Test case 10: num_epochs must be > 0
    with pytest.raises(ValueError, match= "num_epochs must be greater than 0"):
        object_detection_model.train_model(model=model, num_epochs=0)
        
    # Test case 11: num_epochs must be > 0
    with pytest.raises(ValueError, match= "num_epochs must be greater than 0"):
        object_detection_model.train_model(model=model, num_epochs=-1)
        
    # Test case 12: assert that after training, model is still of type fasterrcnn_resnet50_fpn. It trains the model passed in so no retrun value to test
    object_detection_model.train_model(model=model, num_epochs=1)
    assert isinstance(model,FasterRCNN)
    
def test_test_model():
    # Test case 13: incorrect type for model
    with pytest.raises(TypeError, match= "model must be type torchvision.models.detection.FasterRCNN"):
        object_detection_model.test_model(model="")
    
    # Test case 14: incorrect type for model
    with pytest.raises(TypeError, match= "model must be type torchvision.models.detection.FasterRCNN"):
        object_detection_model.test_model(model=None)
    
    # Test case 15: assert that after testing, model is still of type FasterRCNN. It trains the model passed in so no retrun value to test
    model = object_detection_model.create_model(2, "packet")
    object_detection_model.train_model(model=model, num_epochs=1)
    
    object_detection_model.test_model(model=model)
    assert isinstance(model,FasterRCNN)
        
def test_save_model():
    
    path = "./src/test_files/obj_detect_test/test.pt"
    
    # Test case 16: incorrect type for model
    with pytest.raises(TypeError, match= "model must be type torchvision.models.detection.FasterRCNN"):
        object_detection_model.save_model(model="", path=path)
    
    # Test case 17: incorrect type for model
    with pytest.raises(TypeError, match= "model must be type torchvision.models.detection.FasterRCNN"):
        object_detection_model.save_model(model=None, path=path)
    
    model = object_detection_model.create_model(2,"packet")
    
    # Test case 18: incorrect type for path
    with pytest.raises(TypeError, match= "path must be type str"):
        object_detection_model.save_model(model=model, path=5)
    
    # Test case 19: incorrect type for path
    with pytest.raises(TypeError, match= "path must be type str"):
        object_detection_model.save_model(model=model, path=None)
        
    # Test case 20: assert test model exists after saving
    object_detection_model.save_model(model=model,path=path)
    assert os.path.exists(path)
    
def test_save_checkpoint():
    path = "./src/test_files/obj_detect_test/checkpoints/test.pth"
    
    # Test case 21: incorrect type for model
    with pytest.raises(TypeError, match= "model must be type torchvision.models.detection.FasterRCNN"):
        object_detection_model.save_checkpoint(model="", path=path)
    
    # Test case 22: incorrect type for model
    with pytest.raises(TypeError, match= "model must be type torchvision.models.detection.FasterRCNN"):
        object_detection_model.save_checkpoint(model=None, path=path)
    
    model = object_detection_model.create_model(2)
    
    # Test case 23: incorrect type for path
    with pytest.raises(TypeError, match= "path must be type str"):
        object_detection_model.save_checkpoint(model=model, path=5)
    
    # Test case 24: incorrect type for path
    with pytest.raises(TypeError, match= "path must be type str"):
        object_detection_model.save_checkpoint(model=model, path=None)
        
    # Test case 24.1 bad path for path
    with pytest.raises(FileNotFoundError,match="path does not exist"):
        object_detection_model.load_checkpoint(model=model, path="./badPath/folders/do/not/exist")
        
    # Test case 25: assert test checkpoint exists after saving
    object_detection_model.save_checkpoint(model=model,path=path)
    assert os.path.exists(path)
    
def test_load_checkpoint():
    path = "./src/test_files/obj_detect_test/checkpoints/test.pth"
    
    # Test case 26: incorrect type for model
    with pytest.raises(TypeError, match= "model must be type torchvision.models.detection.FasterRCNN"):
        object_detection_model.load_checkpoint(model="", path=path)
    
    # Test case 27: incorrect type for model
    with pytest.raises(TypeError, match= "model must be type torchvision.models.detection.FasterRCNN"):
        object_detection_model.load_checkpoint(model=None, path=path)
    
    model = object_detection_model.create_model(2)
    
    # Test case 28: incorrect type for path
    with pytest.raises(TypeError, match= "path must be type str"):
        object_detection_model.load_checkpoint(model=model, path=5)
    
    # Test case 29: incorrect type for path
    with pytest.raises(TypeError, match= "path must be type str"):
        object_detection_model.load_checkpoint(model=model, path=None)
    
    # Test case 29.1 bad path for path
    with pytest.raises(FileNotFoundError,match="path does not exist"):
        object_detection_model.load_checkpoint(model=model, path="")
        
    # Test case 30: assert model is of correct type
    object_detection_model.load_checkpoint(model=model,path=path)
    assert isinstance(model,FasterRCNN)
    
def test_create_and_train_model():
    model_path = "./src/test_files/obj_detect_test/test.pt"
    num_objects_predict = 4
    type = "packet"
    # Test case 31: incorrect type for num_epochs
    with pytest.raises(TypeError, match= "num_epochs must be type int"):
        object_detection_model.create_and_train_model(num_epochs="", model_path=model_path, num_objects_to_predict=num_objects_predict, type=type)
        
    # Test case 32:  none type for num_epochs
    with pytest.raises(TypeError, match= "num_epochs must be type int"):
        object_detection_model.create_and_train_model(num_epochs=None, model_path=model_path, num_objects_to_predict=num_objects_predict, type=type)
    
    # Test case 33: num_epochs must be > 0
    with pytest.raises(ValueError, match= "num_epochs must be greater than 0"):
        object_detection_model.create_and_train_model(num_epochs=0, model_path=model_path, num_objects_to_predict=num_objects_predict, type=type)
        
    # Test case 34: num_epochs must be > 0
    with pytest.raises(ValueError, match= "num_epochs must be greater than 0"):
        object_detection_model.create_and_train_model(num_epochs=-1, model_path=model_path, num_objects_to_predict=num_objects_predict, type=type)

    # Test case 35: incorrect type for model_path
    with pytest.raises(TypeError, match= "model_path must be type str"):
        object_detection_model.create_and_train_model(num_epochs=1, model_path=1, num_objects_to_predict=num_objects_predict, type=type)
        
    # Test case 36: incorrect type for model_path
    with pytest.raises(TypeError, match= "model_path must be type str"):
        object_detection_model.create_and_train_model(num_epochs=1, model_path=None, num_objects_to_predict=num_objects_predict, type=type)

    # Test case 37: model_path does not exist
    with pytest.raises(FileNotFoundError, match="not valid path"):
        object_detection_model.create_and_train_model(num_epochs=1, model_path="", num_objects_to_predict=num_objects_predict, type=type) 
    
    # Test case 38: assert that model and checkpoint are saved
    object_detection_model.create_and_train_model(1,model_path=model_path, num_objects_to_predict=num_objects_predict, type=type)
    assert os.path.exists(model_path)
    
    
def test_predict_with_model():
    
    from main import get_caddy_model,get_desk_model,get_packet_model
    model = get_packet_model()
    image_path = "./src/test_files/obj_detect_test/test_image.png"
    bad_image_path = "./src/test_files/obj_detect_test/bad_image.png"
    does_not_exist_image = ""
    
    # Test case 39: incorrect type for image
    with pytest.raises(TypeError, match= "image must be type str or PIL.Image"):
        gen = object_detection_model.predict_with_model(image=None,model=model,type="packet")
        next(gen)
    
    # Test case 40: incorrect type for image
    with pytest.raises(TypeError, match= "image must be type str or PIL.Image"):
        gen = object_detection_model.predict_with_model(image=None,model=model,type="packet")
        next(gen)
        
    # Test case 40.1 image does not exist
    with pytest.raises(FileNotFoundError,match= "image path does not exist"):
        gen = object_detection_model.predict_with_model(image=does_not_exist_image,model=model,type="packet")
        next(gen)
        
    # Test case 41: incorrect type for model_path
    with pytest.raises(TypeError, match= "model must be of type FasterRCNN"):
        gen = object_detection_model.predict_with_model(image=image_path,model=4,type="packet")
        next(gen)
    
    # Test case 42: incorrect type for model_path
    with pytest.raises(TypeError, match= "model must be of type FasterRCNN"):
        gen = object_detection_model.predict_with_model(image=image_path,model=4,type="packet")
        next(gen)
        
    # Test case 42.2 type is wrong type does not exist
    with pytest.raises(TypeError,match= "type must be a str of either packet,desk, or caddy"):
        gen = object_detection_model.predict_with_model(image=image_path,model=model,type=None)
        next(gen)
    
    type = "hello"
    # Test case 42.3 type is not a existing class type of packet,desk, or caddy
    with pytest.raises(KeyError,match= type + " does not exist in dictionary of known class types"):
        gen = object_detection_model.predict_with_model(image=image_path,model=model,type=type)
        next(gen)
        
    
    type = "packet"
    
if __name__ == '__main__':
    test_train_model()