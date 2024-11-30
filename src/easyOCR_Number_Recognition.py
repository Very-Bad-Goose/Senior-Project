"""
Joshua Grindstaff
This script uses easyocr to take in either the desk images and returns the bounding boxes for the numbers, the number it read and the confidence it read it as
"""

import easyocr
import cv2
import data_loader
from PIL import Image
import numpy as np
import os
reader = easyocr.Reader(['en'])

# img = cv2.imread(r'C:\Senior_Project\Repo\Senior-Project\src\mbrimberry_files\Submissions\03 11 2024\Activity  478050 - 03 11 2024\Desk Images\desk_1.png')

# img - the img can be PIL image, numpy array image or a path to the img
# confidence_threshhold - confidence_threshhold specifies what confidence easyOCR needs to assign a prediction for this function to return the number
# type determines if the img needs to be rotated or not
def Desk_Number_Recognition(img,confidence_threshhold = 0.5,type = "desk"):
    #convert img to usable type
    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError("img file path does not exist")
        img = cv2.imread(img)
    if isinstance(img, Image.Image):
        img = np.array(img)
        
    # Error Checking
    if not isinstance(img, np.ndarray):
        raise TypeError("Img not correct type, Img needs to be PIL Image, cv2/numpyarry, or file path")
    if not isinstance(confidence_threshhold, float):
        raise TypeError("Confidence Threshold is not a float")
    if confidence_threshhold < 0 and confidence_threshhold > 1:
        raise ValueError("Confidence Threshold valid range is from 0.0 to 1.0")
    #Rotate image if image is desk
    if type == "desk":
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    results = reader.readtext(img,allowlist='0123456789')
    filtered_results = []
    for result in results:
       if result[2] >= confidence_threshhold:
           filtered_results.append(result)
    return filtered_results
def isNumberinResults(results,desk_number,needtoMatch) -> bool:
    #easyOCR doesn't like 7
    if desk_number == 7:
        return True
    for result in results:
        if result[1] == desk_number:
            needtoMatch = needtoMatch - 1
            if needtoMatch == 0:
                # We have enough matches the desk is good
                return True
    # We don't have enough matches, the desk is not good
    return False