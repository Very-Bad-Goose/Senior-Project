"""
Austin Nolte

Detection for how blurry an image is using laplacian filter

"""

import os
import cv2
from pathlib import Path as PathLib

# threshold for laplacian variance, adjustable here
threshold = 50 

"""

This is the old way of doing this, refactored to allow for easy stopping and starting in threading to allow gui to keep functioning and close properly 


# Global Boolean to detect if the program has already written output file to append instead of overwriting
global file_made 
file_made = 0

def detect_image_blur(path: str, folder_path: str):
    
    print_path = os.path.relpath(path)

    # Read the image and process as before
    image_file = cv2.imread(path)
    image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(image_file, cv2.CV_64F)
    laplacian_var = laplacian.var()
    global file_made
    if laplacian_var < threshold:
        if file_made == 0:
            file_made = 1
            with open(os.path.join(os.getcwd(), "image_blur_results.txt"), "w") as output_file:
                output_file.write(f"{print_path}\n")
        else:
            with open(os.path.join(os.getcwd(), "image_blur_results.txt"), "a") as output_file:
                output_file.write(f"{print_path}\n")

def detect_image_blur_helper(folder_path: str):
    
    path_exists = os.path.exists(folder_path)

    # printing error if path does not exist and exiting script
    if(not path_exists):
        print("Subsmissions folder not found, please put in a valid folder location.")
        quit()

    # Walk the sub directory and get all images
    global file_made
    file_made = 0
    for path,sub_path,files in os.walk(folder_path):
        for file in files:
                # if the file ends with supported image types check how blurry it is
                if file.endswith((".png",".jpeg",".jpeg",".heic")):
                    image_path = os.path.join(path,file)
                    detect_image_blur(image_path, path)
"""

def detect_image_blur(image_path: str):
    image_file = cv2.imread(image_path)
    image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(image_file, cv2.CV_64F)
    laplacian_var = laplacian.var()
    if laplacian_var < threshold:
        return True
    else:
        return False