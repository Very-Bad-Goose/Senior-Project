"""
    Austin Nolte
    Funcitons to connect front end and backend
"""


import torch
import threading
from image_blur_detection import detect_image_blur_helper as detect_blur
from id_periodNum_nn import predict_with_id_model
import os
from PIL import Image

predict_flag = False
t1 = None

# Loads model, will be called at startup, edit model_path variable to ensure correct model loaded
def load_model(model_path: str):    
    if torch.cuda.is_available():
        torch.device('cuda')
        model = torch.load(model_path,weights_only=True,map_location=torch.device('cuda'))
    elif torch.backends.mps.is_available():
        torch.device('mps')
        model = torch.load(model_path,weights_only=True,map_location=torch.device('mps'))
    else:
        torch.device('cpu')
        model = torch.load(model_path,weights_only=True,map_location=torch.device('cpu'))
        
    print("Model succesfully loaded")
    return model

# use the model to make predictions, will image pre process first and not use those images, uses a thread to make predictions so it can be stopped by gui
def predict_model_test(model):
    global predict_flag
    predict_flag = True
    t1 = threading.Thread(target=predict_model_test_helper,args=(model,))
    t1.start()

# use the model to make predictions, will image pre process first and not use those images
def predict_model_test_helper(model):
    global predict_flag
    while predict_flag:
        # for testing purposes this will work with test model which does not use images, look at predict_model and predict_model_helper below for real functions
        pass
    
# stop the model predicting 
def stop_model():
    global predict_flag
    predict_flag = False
    global t1
    # if the thread has started join it back to main thread when stop is pressed
    if t1 is not None:
        t1.join()
        
# this will become the real predict model, other one is just for testing purposes while real model is being made
def predict_model(model, folder_path: str):
    global predict_flag
    predict_flag = True
    t1 = threading.Thread(target=predict_model_helper,args=(model,folder_path,))
    t1.start()

def predict_model_helper(model, folder_path:str):
    global predict_flag
    # only need to run image blur check once, therefore outside of loop
    if(predict_flag):
        print("detecting image blur")
        detect_blur(folder_path)
        print("done detecting image blur")
        # get image blur results into a string to be able to check
        with open("image_blur_results.txt","r") as image_blur_file:
            image_blur_results = image_blur_file.read()
    while predict_flag:
            for path,sub_path,files in os.walk(folder_path):
                for file in files:
                    # check if file is supported and if it passed blur check
                    if not predict_flag:
                        break
                    if file.endswith((".png",".jpeg",".jpeg",".heic")):
                        image_path = os.path.join(path,file)
                        check_path = image_path.split('\\')
                        
                        check_path = check_path[-3] + "/" + check_path[-2] + "/" + check_path[-1]
                        if check_path not in image_blur_results:
                            # use line below once model has been trained and function has been made 
                            if "Activity Packet" in image_path:
                                test = predict_with_id_model(image_path,"./models/id_periodNum_model.pt","packet")
                                processed_path = check_path.replace('/','_')
                                for i in test:
                                    pred_box,score,image,label = i
                                    if label == 0:
                                        image.save(f"./src/result_images/id_num{processed_path}")
                                    elif label == 1:
                                        image.save(f"./src/result_images/period_num{processed_path}")
                                pass
                                
                            elif "Desk Images" in image_path:
                                test = predict_with_id_model(image_path,"./models/deskmodel.pt","desk")
                                processed_path = check_path.replace('/','_')
                                for i in test:
                                    pred_box,score,image,label = i
                                    if label == 0:
                                        image.save(f"./src/result_images/calculator{processed_path}")
                                    elif label == 1:
                                        image.save(f"./src/result_images/desk_number{processed_path}")