"""
    Austin Nolte
    Funcitons to connect front end and backend
"""


import torch
import threading
from image_blur_detection import detect_image_blur_helper as detect_blur
from temp_model import NeuralNet
from temp_model import predict_with_model_test,predict_with_model
import os

predict_flag = False
t1 = None

# Loads model, will be called at startup, edit model_path variable to ensure correct model loaded
def load_model(model_path: str):    
    model = torch.load(model_path,weights_only=False)
    print("Model succesfully loaded")
    return model

# use the model to make predictions, will image pre process first and not use those images, uses a thread to make predictions so it can be stopped by gui
def predict_model_test(model:NeuralNet):
    global predict_flag
    predict_flag = True
    t1 = threading.Thread(target=predict_model_test_helper,args=(model,))
    t1.start()

# use the model to make predictions, will image pre process first and not use those images
def predict_model_test_helper(model: NeuralNet):
    global predict_flag
    while predict_flag:
        # for testing purposes this will work with test model which does not use images, look at predict_model and predict_model_helper below for real functions
        predict_with_model_test(model)

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
                    if file.endswith((".png",".jpeg",".jpeg",".heic")):
                        image_path = os.path.join(path,file)
                        check_path = image_path.split('\\')
                        check_path = check_path[-3] + "\\" + check_path[-2] + "\\" + check_path[-1]
                        if check_path not in image_blur_results:
                            # use line below once model has been trained and function has been made 
                            #predict_with_model(model,image_path)
                            print(image_path)