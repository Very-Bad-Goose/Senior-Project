
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.ops import box_convert
from PIL import Image
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as nnF

def process_image_to_digits(img_path, box):
    img = Image.open(img_path)

    def crop_img(img, box):
        # Convert box format and scale to image dimensions
        bbox = box_convert(box, 'cxcywh', 'xywh')
        width, height = img.size
        left, top, crop_width, crop_height = (int(bbox[0][i] * (width if i % 2 == 0 else height)) for i in range(4))
        # Crop and return image section
        return F.crop(img, top, left, crop_height, crop_width)

    def crop_to_bottom_half(img):
        # Keep only the bottom half of the image
        width, height = img.size
        return img.crop((0, height // 2, width, height))

    def preprocess_image(img):
        # Convert to grayscale and apply thresholding
        gray_img = img.convert("L")
        gray_img_np = np.array(gray_img)
        _, binary_img = cv2.threshold(gray_img_np, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary_img

    def segment_digits(binary_img):
        # Find contours in binary image
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        digit_images = []
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])  # Sort left-to-right
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Extract and resize each digit
            digit = binary_img[y:y+h, x:x+w]
            digit_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
            digit_images.append(Image.fromarray(digit_resized))
        
        return digit_images

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

    # Crop and preprocess image
    cropped_img = crop_img(img, box_t)
    cropped_img_bottom = crop_to_bottom_half(cropped_img)
    binary_img = preprocess_image(cropped_img_bottom)
    digit_images = segment_digits(binary_img)

    # Define model and load weights
    model = MNISTModel()
    model.load_state_dict(torch.load('./models/handwriting_recognition_model.pt'))
    model.eval()

    # Define transformation for predictions
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    def predict_digits(digit_images):
        predictions = []
        # Predict each segmented digit
        for digit_img in digit_images:
            digit_tensor = transform(digit_img).unsqueeze(0)
            with torch.no_grad():
                output = model(digit_tensor)
                probabilities = nnF.softmax(output, dim=1)
                predicted_digit = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_digit].item()
            predictions.append((predicted_digit, confidence))
        return predictions

    # Run predictions and get top 6 digits by confidence
    predictions = predict_digits(digit_images)
    top_6_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:6]
    result_string = ''.join(str(pred[0]) for pred in top_6_predictions)

    # Display final predicted 6-digit number
    return result_string


img_path = './src/mbrimberry_files/Submissions/03 11 2024/Activity  474756 - 03 11 2024/Activity Packet/activity_1.png'
box_t = torch.tensor([[0.354167, 0.106061, 0.156863, 0.066288]])
print(process_image_to_digits(img_path, box_t))
