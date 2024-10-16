"""
Austin Nolte

Detection for how blurry an image is using laplacian filter

"""

import os
import cv2
from pathlib import Path as PathLib

# threshold for laplacian variance, adjustable here
threshold = 30 

# Global Boolean to detect if the program has already written output file to append instead of overwriting
global file_made 
file_made = 0

def detect_image_blur(path: str, folder_path: str):
    
    # read in image into cv2 format
    image_file = cv2.imread(path)
    
    # convert to grayscale
    image_file = cv2.cvtColor(image_file,cv2.COLOR_BGR2GRAY)
    
    # apply laplacian filter to detect blur
    laplacian = cv2.Laplacian(image_file, cv2.CV_64F)
    
    # making laplacian the variance
    laplacian_var = laplacian.var()
    global file_made
    print_path = path.split('\\')
    print_path = print_path[-3] + "\\" + print_path[-2] + "\\" + print_path[-1]
    if laplacian_var < threshold:
        if file_made == 0:
            file_made = 1
            with open(os.getcwd() + "\image_blur_results.txt", "w") as output_file:
                output_file.write(f"{print_path}\n")
        else:
            with open(os.getcwd() + "\image_blur_results.txt", "a") as output_file:
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