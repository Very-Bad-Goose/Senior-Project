
"""
Austin Nolte

unit test cases for model_api
"""
import pytest
import torch
import json
import pathlib
from pathlib import Path
import os
from typing import Tuple
import re
from PIL import Image
from torchvision.models.detection import FasterRCNN
import model_api

def test_load_model():
    
    # checking for device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # Test case 1: model_path is not str or pathlib.Path
    model_path = 4
    type = "packet"
    with pytest.raises(TypeError, match= "model_path must be type str or Path"):
        model_api.load_model(model_path=model_path,model_type=type)
    
    # Test case 2: model_path is none
    model_path = None
    type = "packet"
    with pytest.raises(TypeError, match= "model_path must be type str or Path"):
        model_api.load_model(model_path=model_path,model_type=type)
    
    # Test case 3: model_path is of type str, should return a model of FasterRCNN
    model_path = "./models/id_periodNum_model.pt"
    type ="packet"
    model = model_api.load_model(model_path=model_path,model_type=type)
    assert(isinstance(model,FasterRCNN))
    
    
    # Test case 4: asserting model was loaded onto correct device after load
    assert next(model.parameters()).device == device 
    
    # Test case 4: model_path is of type pathLib.Path, should return a model of FasterRCNN
    model_path = pathlib.Path("./models/id_periodNum_model.pt")
    type ="packet"
    model = model_api.load_model(model_path=model_path,model_type=type)
    assert(isinstance(model,FasterRCNN))
    
    # Test case 5: asserting model was loaded onto correct device after load
    assert next(model.parameters()).device == device
    
def test_model_predict():
    # Test case 1: models is wrong type
    models = [] # list not tuple
    folder_path = "./src"
    with pytest.raises(TypeError,match=("models must be type tuple")):
        model_api.model_predict(models=models,folder_path=folder_path)
        
    # Test case 2: models is None
    models = None
    folder_path = "./src"
    with pytest.raises(TypeError,match=("models must be type tuple")):
        model_api.model_predict(models=models,folder_path=folder_path)
        
    # Test case 3: folder_path is not type str or pathlib.Path
    
    model1:FasterRCNN = model_api.load_model("./models/caddy_model.pt","caddy")
    model2:FasterRCNN = model_api.load_model("./models/desk_model.pt","desk")
    model3:FasterRCNN = model_api.load_model("./models/id_periodNum_model.pt","packet")
    
    models = (model1,model2,model3)
    folder_path = 4
    with pytest.raises(TypeError,match=("folder path must be type str or Path")):
        model_api.model_predict(models=models,folder_path=folder_path)
        
    # Test case 4: folder_path is None
    folder_path = None
    with pytest.raises(TypeError,match=("folder path must be type str or Path")):
        model_api.model_predict(models=models,folder_path=folder_path)
        
    # Test case 5: models is not of len 3
    models = (model1,model2)
    folder_path = "./src"
    with pytest.raises(ValueError,match=("models must be of len 3")):
        model_api.model_predict(models=models,folder_path=folder_path)
    
    # Test case 6: models does not have FastRCNN models in the tuple
    models = (1,2,4)
    folder_path = "./src"
    with pytest.raises(TypeError,match=("model must be of type FasterRCNN")):
        model_api.model_predict(models=models,folder_path=folder_path)
    
    # Test case 7: folder_path does not exist
    models = (model1,model2,model3)
    folder_path = ""
    with pytest.raises(FileNotFoundError):
        model_api.model_predict(models=models,folder_path=folder_path)
    
    