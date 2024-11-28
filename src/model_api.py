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
from PIL import Image


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
def model_predict(models:tuple , folder_path):
    
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
    t1 = threading.Thread(target=model_predict_helper,args=(models,folder_path,))
    t1.start()

def model_predict_helper(models:tuple, folder_path):
    
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
        for path,sub_path,files in os.walk(folder_path):
            for file in files:
                # check if file is supported and if it passed blur check
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
                    if not blur_check:
                        # use line below once model has been trained and function has been made 
                        if "Activity Packet" in image_path:
                            test = predict_with_model(image=image_path,model=packet_model,type="packet")
                            processed_path = check_path.replace('/','_')
                            for i in test:
                                pred_box,score,image,label = i
                                if label == 0:
                                    image.save(f"./src/result_images/id_num{processed_path}")
                                elif label == 1:
                                    image.save(f"./src/result_images/period_num{processed_path}")
                            
                        elif "Desk Images" in image_path:
                            if "desk_1" in image_path:
                                test = predict_with_model(image=image_path,model=desk_model,type="desk")
                                processed_path = check_path.replace('/','_')
                                for i in test:
                                    pred_box,score,image,label = i
                                    if label == 0:
                                        image.save(f"./src/result_images/calculator{processed_path}")
                                    elif label == 1:
                                        image.save(f"./src/result_images/desk_number{processed_path}")
                            elif "desk_2" in image_path:
                                print("here in desk_2")
                                test = predict_with_model(image=image_path,model=caddy_model,type="caddy")
                                
                                processed_path = check_path.replace('/','_')
                                for i in test:
                                    pred_box,score,image,label = i
                                    image.save(f"./src/result_images/caddy{processed_path}")