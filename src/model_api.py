"""
    Austin Nolte
    Funcitons to connect front end and backend
"""


import torch
import threading
from image_blur_detection import detect_image_blur as detect_blur
from object_detection_model import predict_with_model, create_model
from torchvision.models.detection import FasterRCNN
from pathlib import Path
import os
from easyOCR_Number_Recognition import Desk_Number_Recognition
from PIL import Image
from handwriting_recognition import process_image_to_digits

num_classes = {"desk": 2,"caddy": 1,"packet": 2}
predict_flag = False
t1 = None

# Loads model, will be called at startup, edit model_path variable to ensure correct model loaded
def load_model(model_path,model_type):    
    
    if not isinstance(model_path,(str,Path)):
        raise TypeError("model_path must be type str or Path")
    if not os.path.exists(model_path):
        raise FileNotFoundError
    
    if not isinstance(model_type,str):
        raise TypeError("model_type must be a str of either: 'packet', 'desk', or 'caddy'")
    
    model = create_model(num_classes.get(model_type),model_type)
    
    # checking device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.load_state_dict(torch.load(model_path,weights_only=False,map_location=device))
    model.to(device)
    
    print("Model succesfully loaded")
    return model
# stop the model predicting 
def stop_model():
    global predict_flag
    predict_flag = False
    global t1
    # if the thread has started join it back to main thread when stop is pressed
    if t1 is not None:
        t1.join()
        
# this will become the real predict model, other one is just for testing purposes while real model is being made
def model_predict(models:tuple , folder_path, results):
    
    if not isinstance(folder_path,(str,Path)):
        raise TypeError("folder path must be type str or Path")
    if not os.path.exists(folder_path):
        raise FileNotFoundError
    if not isinstance(models,tuple):
        raise TypeError("models must be type tuple")
    if len(models) != 3:
        raise ValueError("models must be of len 3 ")
    for model in models:
        if not isinstance(model,FasterRCNN):
            raise TypeError("model must be of type FasterRCNN")

    
    global predict_flag
    predict_flag = True
    t1 = threading.Thread(target=model_predict_helper,args=(models,folder_path,results))
    t1.start()
    return t1

def model_predict_helper(models:tuple, folder_path, results):
    
    # must be in order of packet,desk,caddy model in tuple
    packet_model,desk_model,caddy_model = models
    
    
    global predict_flag
    # only need to run image blur check once, therefore outside of loop
    # if(predict_flag):
    #     print("detecting image blur")
    #     detect_blur(folder_path)
    #     print("done detecting image blur")
    #     # get image blur results into a string to be able to check
    #     with open("image_blur_results.txt","r") as image_blur_file:
    #         image_blur_results = image_blur_file.read()
    while predict_flag:
        packet_results = []
        desk_results = []
        caddy_results = []
        for path,sub_path,files in os.walk(folder_path):
            # check if file is supported and if it passed blur check
            for file in files:
                if not predict_flag:
                    break
                if file.endswith((".png",".jpeg",".jpg",".heic")):
                    image_path = os.path.join(path,file)
                    check_path = image_path.split('\\')
                    try:
                        blur_check = detect_blur(image_path)
                    except Exception as e:
                        print(f"Error: {e}")    
                    check_path = check_path[-3] + "/" + check_path[-2] + "/" + check_path[-1]
                        # use line below once model has been trained and function has been made 
                    if "Activity Packet" in image_path:
                        if blur_check:
                            print(f"{image_path} is too blury")
                            packet_results.append(None)
                        preds = predict_with_model(image=image_path,model=packet_model,type="packet")
                        for i in preds:
                            pred_box,score,image,label = i
                            if label == 0:
                                stu_box = pred_box
                            elif label == 1:
                                per_box = pred_box
                        if per_box is not None and stu_box is not None:
                            #print(f"{per_box.dtype} is period, {stu_box.dtype} is ID")
                            packet_results.append(process_image_to_digits(image_path, stu_box, per_box))
                        else:
                            print(f"Model could not detect period num and/or student ID for {image_path}")
                            packet_results.append(None)
                        per_box = None
                        stu_box = None
                    elif "Desk Images" in image_path:
                        if "desk_1" in image_path:
                            if blur_check:
                                print(f"{image_path} is too blury")
                                desk_results.append(None)
                            # preds = predict_with_model(image=image_path,model=desk_model,type="desk")
                            desk_results = Desk_Number_Recognition(image_path,confidence_threshhold=0.7, type = "desk")
                        elif "desk_2" in image_path:
                            if blur_check:
                                print(f"{image_path} is too blury")
                                caddy_results.append(None)
                            # preds = predict_with_model(image=image_path,model=caddy_model,type="caddy")
                            caddy_results = Desk_Number_Recognition(image_path,confidence_threshhold=0.7, type = "caddy")
        results.append(packet_results)
        results.append(desk_results)
        results.append(caddy_results)   
        predict_flag = False    
        
                            
