#Jacob Sherer 11/26/2024

import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.ops import box_convert
from PIL import Image
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as nnF
import matplotlib.pyplot as plt

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool2d((12, 12))
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nnF.relu(self.conv1(x))
        x = nnF.max_pool2d(x, 2)
        x = nnF.relu(self.conv2(x))
        x = nnF.max_pool2d(x, 2)
        x = self.pool(x)
        x = x.view(-1, 64 * 12 * 12)
        x = nnF.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Takes in an image image path and two torch tensors. Returns any recognised digits in those two locations
def process_image_to_digits(img_path, box_stu, box_per):
    img = Image.open(img_path)
    results = []

    #Define model and load weights
    global model
    model = MNISTModel()
    model.load_state_dict(torch.load('./models/handwriting_recognition_model.pt'))
    model.eval()

    #Define transformation for predictions
    global transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    #student number
    cropped_img = crop_img(img, box_stu)
    gray_img = preprocess_image(cropped_img)
    deskewed_img = deskew_image(gray_img)
    binary_img = convert_to_binary(deskewed_img)
    cropped_img_bottom = crop_to_bottom_half(binary_img)
    cropped_img_bottom_np = np.array(cropped_img_bottom.convert("L"))
    digit_images = segment_digits(cropped_img_bottom_np)
    predictions = predict_digits(digit_images)
    result_string = ''.join(str(pred[0]) for pred in predictions[:6])
    results.append(result_string)

    #period number
    cropped_img = crop_img(img, box_per)
    gray_img = preprocess_image(cropped_img)
    deskewed_img = deskew_image(gray_img)
    binary_img = convert_to_binary(deskewed_img)
    cropped_img_bottom = crop_to_bottom_half(binary_img)
    cropped_img_bottom_np = np.array(cropped_img_bottom.convert("L"))
    digit_images = segment_digits(cropped_img_bottom_np)
    predictions = predict_digits(digit_images)
    result_string = ''.join(str(pred[0]) for pred in predictions[:1])
    results.append(result_string)

    return results

#Predict each segmented digit
def predict_digits(digit_images):
    predictions = []
    
    for i, digit_img in enumerate(digit_images):
        digit_tensor = transform(digit_img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(digit_tensor)
            probabilities = nnF.softmax(output, dim=1)
            predicted_digit = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_digit].item()
        
        predictions.append((predicted_digit, confidence))

        #debug display
        #plt.figure()
        #plt.imshow(digit_img, cmap='gray')
        #plt.title(f"Predicted: {predicted_digit}, Confidence: {confidence:.2f}")
        #plt.axis("off")
        #plt.show()
    
    return predictions

#Crops the image to the dimensions specified in the tensor
def crop_img(img, box):
    # bbox = box_convert(box, 'cxcywh', 'xywh')
    bbox = box

    width = int(bbox[2]) - int(bbox[0])
    height = int(bbox[3]) - int(bbox[1])
    # width, height = img.size
    # left, top, crop_width, crop_height = (int(bbox[0][i] * (width if i % 2 == 0 else height)) for i in range(4))
    cropped = F.crop(img, int(bbox[1]), int(bbox[0]), width, height)

    #debug display
    #plt.imshow(cropped)
    #plt.title("Cropped Image")
    #plt.show()

    return cropped

def preprocess_image(img):
    gray_img = img.convert("L")
    gray_img_np = np.array(gray_img)
    
    #debug display
    #plt.imshow(gray_img, cmap='gray')
    #plt.title("Grayscale Image")
    #plt.show()
    
    return gray_img_np

def deskew_image(gray_img_np):
    #Detect edges using Canny edge detection
    edges = cv2.Canny(gray_img_np, 50, 150, apertureSize=3)
    
    #Use Hough Line Transform to detect lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
        
        #Calculate the median angle
        median_angle = np.median(angles)

        #Rotate the image to deskew
        (h, w) = gray_img_np.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        deskewed_img = cv2.warpAffine(gray_img_np, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        #debug display
        #plt.imshow(deskewed_img, cmap='gray')
        #plt.title("Deskewed Image")
        #plt.show()
        
        return deskewed_img
    else:
        return gray_img_np  # Return original if no lines are detected
    
# Convert to binary
def convert_to_binary(deskewed_img):
    _, binary_img = cv2.threshold(deskewed_img, 170, 255, cv2.THRESH_BINARY_INV)
    
    #debug display
    #plt.imshow(binary_img, cmap='gray')
    #plt.title("Binary Image")
    #plt.show()
    
    return binary_img

#Crop off top half of image
def crop_to_bottom_half(binary_img):
    img = Image.fromarray(binary_img)
    width, height = img.size
    cropped_half = img.crop((0, height // 2, width, height))
    
    #debug display
    #plt.imshow(cropped_half, cmap='gray')
    #plt.title("cropped_half Image")
    #plt.show()

    return cropped_half

def segment_digits(cropped_half):
    #Find contours
    contours, _ = cv2.findContours(cropped_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    digit_images = []

    #Sort contours left-to-right
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    #Set expected contour size
    height, width = cropped_half.shape
    min_digit_height = height * 0.2
    max_digit_height = height * 0.9
    min_digit_width = width * 0.02
    max_digit_width = width * 0.3

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (min_digit_width <= w <= max_digit_width and min_digit_height <= h <= max_digit_height):
            continue

        digit = cropped_half[y:y+h, x:x+w]
        digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        digit_images.append(digit_resized)

    #Convert to PIL format
    digit_images = [Image.fromarray(digit) for digit in digit_images]
    return digit_images